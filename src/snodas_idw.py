#!/usr/bin/env python3
##############################################################################
#
# U.S. Department of Commerce
# NOAA (National Oceanic and Atmospheric Administration)
# National Weather Service
# Office of Water Prediction
#
# Author:
#     Greg Fall, OWP (created)
#
##############################################################################
"""
Rewrite of IDL program idw.pro for generating SNODAS nudging layers.
"""
import sys
import os
import subprocess
import time
import logging
import argparse
import datetime as dt
from multiprocessing import Pool
#from pprint import pprint

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))
#import local_logger
import ops_logger
import geo_distance as geodst
import mn_geo as mng

import geopandas as gpd
import numpy as np
from numpy.polynomial import Polynomial
import psycopg2
from netCDF4 import Dataset

from tqdm import tqdm
import statsmodels.api as sm
from scipy import stats
# from scipy import optimize

import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature as cfNEF

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
#import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D
try:
    from adjustText import adjust_text
    ADJUST_TEXT_SUPPORTED = True
except ModuleNotFoundError:
    ADJUST_TEXT_SUPPORTED = False

# from pyproj import Proj
from pyproj.transformer import Transformer

try:
    import contextily as ctx
    GEO_TILE_ENGINE = 'ctx'
except ModuleNotFoundError:
    import cartopy.io.img_tiles as cimgt
    GEO_TILE_ENGINE = 'cimgt'

class TimerError(Exception):
    """
    A custom exception used to report errors in use of Timer class
    """

class Timer:
    """
    Class for wall time calculation and reporting.
    """
    def __init__(self):
        self._start_time = None
    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError('Timer is running. Use .stop() to stop it')
        self._start_time = time.perf_counter()
    def stop(self):
        """Stop the timer, and report the elapsed time"""
        logger = logging.getLogger()
        if self._start_time is None:
            raise TimerError('Timer is not running. Use .start() to start it')
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        message = f'Elapsed time: {elapsed_time:4f} seconds'
        logger.info(message)


class Error(Exception):
    """
    Base class for homemade exceptions.
    """

class GridInconsistency(Error):
    """
    Exception for inconsistent grid/array geometries.
    """

class IDW:
    """
    Parameters, process variables, and performance metrics for inverse
    distance weighted interpolation.
    """
    def __init__(self,
                 bbox_half_width_lat_deg,
                 bbox_half_width_lon_deg,
                 max_distance_km=75.0,
                 min_num_points=3,
                 max_num_points=50,
                 horizontal_power=1.1,
                 elevation_power=1.1,
                 elev_diff_floor_m=0.25,
                 horiz_dist_floor_km=0.25,
                 analysis_method='classic',
                 distance_method='geopy_great_circle',
                 aeqd_mosaic=None):
        self.bbox_half_width_lat_deg = bbox_half_width_lat_deg
        self.bbox_half_width_lon_deg = bbox_half_width_lon_deg
        self.max_distance_km = max_distance_km
        self.min_num_points = min_num_points
        self.max_num_points = max_num_points
        self.horizontal_power = horizontal_power
        self.elevation_power = elevation_power
        self.elev_diff_floor_m = elev_diff_floor_m
        self.horiz_dist_floor_km = horiz_dist_floor_km
        self.analysis_method = analysis_method
        self.distance_method = distance_method
        self.aeqd_mosaic = aeqd_mosaic
        self.num_distance_calc = 0
        self.distance_calc_total_wall_time = 0.0
        self.num_too_few_obs = 0

    def classic(self,
                target_lat,
                target_lon,
                target_elev,
                pt_is_real,
                pt_lat,
                pt_lon,
                pt_elev,
                pt_z_val):
        """
        Traditional IDW for SNODAS delta SWE.
        """

        # Calculate ellipsoid or great circle distances.
        start_time = time.perf_counter()
        num_pts = len(pt_lat)
        horiz_dist_km = \
            geodst.one_to_many_km(target_lat, target_lon,
                                  pt_lat, pt_lon,
                                  distance_method=self.distance_method,
                                  aeqd_mosaic=self.aeqd_mosaic)
        self.distance_calc_total_wall_time += time.perf_counter() - start_time
        self.num_distance_calc += num_pts

        # Will remove data outside the max_distance_km.
        cond = np.where(horiz_dist_km <= self.max_distance_km)[0]
        if len(cond) < self.min_num_points:
            return None

        # Check the range of pt_z_val in the neighborhood. The solution is
        # trivial if the range is zero.
        z_val = pt_z_val[cond]
        z_min = pt_z_val.min()
        z_max = pt_z_val.max()
        if z_max == z_min:
            return z_min

        # Adjust horizontal distances to the minimum allowed and truncate to
        # the neighborhood.
        horiz_dist_km = np.maximum(self.horiz_dist_floor_km,
                                   horiz_dist_km[cond])

        # Calculate elevation differences, adjusting to the minimum allowed as
        # needed, and truncate to the neighborhood.
        elev_diff_m = np.maximum(self.elev_diff_floor_m,
                                 np.abs(pt_elev[cond] - target_elev))

        # Only keep up to max_num_points of the nearest neighbors.
        # GF: I do not see the point of this, and suspect it was done in the
        # first place to avoid complicated memory management in IDL, not
        # because it has a real purpose.
        order = horiz_dist_km.argsort()[:self.max_num_points]
        horiz_dist_km = horiz_dist_km[order]
        elev_diff_m = elev_diff_m[order]
        z_val = z_val[order]

        # Perform the IDW calculation.
        weight = np.power(elev_diff_m, -self.elevation_power) * \
            (self.max_distance_km - horiz_dist_km) * \
            np.power(horiz_dist_km, -self.horizontal_power)
        z_val_interp = np.sum(weight * z_val) / np.sum(weight)

        return z_val_interp


    def with_elev_regression(self,
                             target_lat,
                             target_lon,
                             target_elev,
                             pt_is_real,
                             pt_lat,
                             pt_lon,
                             pt_elev,
                             pt_z_val,
                             polynomial_degree=1):
        """
        IDW with elevation regression included. The outputted value is a
        weighted combination of the IDW result and the result based on
        elevation regression, where the weight for the regression result is
        the R^2 value of the regression.
        """

        # Calculate ellipsoid or great circle distances.
        ref_time = time.perf_counter()
        num_pts = len(pt_lat)
        # horiz_dist_km = \
        #     np.array([distance.distance((target_lat, target_lon),
        #                                 (pt_lat[i], pt_lon[i])).km
        #               for i in range(num_pts)])
        # horiz_dist_km = geoglib_geodesic_km(target_lat, target_lon,
        #                                         pt_lat, pt_lon)
        # horiz_dist_km = geopy_gc_km_mp(target_lat, target_lon,
        #                                pt_lat, pt_lon,
        #                                num_processors=16)
        # horiz_dist_km = geopy_gc_km(target_lat, target_lon,
        #                             pt_lat, pt_lon)
        horiz_dist_km = \
            geodst.one_to_many_km(target_lat, target_lon,
                                  pt_lat, pt_lon,
                                  distance_method=self.distance_method,
                                  aeqd_mosaic=self.aeqd_mosaic)
        self.distance_calc_total_wall_time += time.perf_counter() - ref_time
        self.num_distance_calc += num_pts

        # Remove data outside the max_distance_km.
        # cond = np.where(horiz_dist_km <= self.max_distance_km)[0]
        # if len(cond) < self.min_num_points:
        #     return None
        cond = horiz_dist_km <= self.max_distance_km
        if np.sum(cond) < self.min_num_points:
            return None

        # Check the range of pt_z_val in the neighborhood. The solution is
        # trivial if the range is zero.
        z_val = pt_z_val[cond]
        z_min = pt_z_val.min()
        z_max = pt_z_val.max()
        if z_max == z_min:
            return z_min

        # Adjust horizontal distances to the minimum allowed and truncate to
        # the neighborhood.
        horiz_dist_km = np.maximum(self.horiz_dist_floor_km,
                                   horiz_dist_km[cond])

        # Calculate elevation differences.
        elev_diff_m = target_elev - pt_elev[cond]

        # Truncate remaining data.
        is_real = pt_is_real[cond]

        # Do not use pt_lat and pt_lon after this point without truncating
        # them with [cond].

        # Only keep up to max_num_points of the nearest neighbors.
        # GF: I do not see the point of this, and suspect it was done in the
        # first place to avoid complicated memory management in IDL, not
        # because it has a real purpose.
        order = horiz_dist_km.argsort()[:self.max_num_points]
        horiz_dist_km = horiz_dist_km[order]
        is_real = is_real[order]
        elev_diff_m = elev_diff_m[order]
        pt_elev = pt_elev[order]
        z_val = z_val[order]

        # Perform the "modified Shepard" IDW calculation.
        weight = \
            np.power((self.max_distance_km - horiz_dist_km) /
                     (self.max_distance_km * horiz_dist_km),
                     self.horizontal_power)
        z_val_idw = np.sum(weight * z_val) / np.sum(weight)

        # Elevation fit has to use binned averages, because there will often
        # be too many zeroes dominating the results otherwise.
        # z_mean, z_bin_edge, z_bin_index = \
        #     stats.binned_statistic(z_val, z_val,
        #                            bins=self.max_num_points,
        #                            statistic='mean')
        # z_bin_ctr = 0.5 * (z_bin_edge[1:] + z_bin_edge[:-1])
        if np.sum(is_real) < 5:
            r_squared = 0.0
            z_val_fit = 0.0
        else:
            real_pt_elev = pt_elev[is_real]
            #real_elev_diff = elev_diff_m[is_real]
            real_z_val = z_val[is_real]

            poly_elev = Polynomial.fit(real_pt_elev,
                                       real_z_val,
                                       polynomial_degree)
            z_fit = poly_elev(real_pt_elev)
            z_val_fit = poly_elev(target_elev)
            # Standard scores for observation points and fitted values.
            ssx = (real_z_val - np.mean(real_z_val)) / \
                np.std(real_z_val, ddof=1)
            ssy = (z_fit - np.mean(z_fit)) / np.std(z_fit, ddof=1)
            r_squared = (np.sum(ssx * ssy) / (len(real_z_val) - 1.0))**2
            # if random.randrange(0, 1000) == 500:
            #     print(f'fitted function: {poly_elev}')
            #     print(f'r\N{SUPERSCRIPT TWO} = {r_squared}')
            #     print(target_elev)
            #     print(z_val_idw)
            #     print(z_val_fit)
            #     plt.plot(real_pt_elev, real_z_val, 'ko', markersize = 3.0)
            #     plt.plot(real_pt_elev, z_fit, 'rx', markersize = 5.0)
            #     plt.plot(target_elev, z_val_fit, 'go', markersize = 5.0)
            #     xrange = plt.xlim()
            #     xshowfit = np.linspace(xrange[0], xrange[1], 101)
            #     yshowfit = poly_elev(xshowfit)
            #     plt.plot(xshowfit, yshowfit, 'r-')
            #     plt.xlabel('Elevation (m)')
            #     plt.ylabel('Delta SWE')
            #     plt.show()
            #     plt.plot(real_z_val, z_fit, 'bo', markersize = 2.0)
            #     plt.xlabel('Observed Delta SWE')
            #     plt.ylabel('Fitted Delta SWE')
            #     plt.show()
            #wr_squared = 0.0

        z_val_combo = \
            (1.0 - r_squared) * z_val_idw + r_squared * z_val_fit

        return z_val_combo


class IDWGrid:
    """
    Process variables and performance metrics for inverse distance weighted
    interpolation on a grid.
    """
    def __init__(self,
                 lat_axis,
                 lon_axis,
                 bbox_half_width_lat_deg,
                 bbox_half_width_lon_deg):
        self.lat_axis = lat_axis
        self.lon_axis = lon_axis
        self.bbox_half_width_lat_deg = bbox_half_width_lat_deg
        self.bbox_half_width_lon_deg = bbox_half_width_lon_deg
        self.output_grid_shape = (len(lat_axis), len(lon_axis))
        self.row = self.output_grid_shape[0]
        self.prev_row = self.output_grid_shape[0]
        self.col = self.output_grid_shape[1]
        self.prev_col = self.output_grid_shape[1]
        self.row_lat = np.nan
        self.nbrhood_bbox_min_lat = np.nan
        self.nbrhood_bbox_max_lat = np.nan
        self.col_lon = np.nan
        self.nbrhood_bbox_min_lon = np.nan
        self.nbrhood_bbox_max_lon = np.nan
        self.num_masked = 0
        self.num_too_few_obs = 0

    def set_bbox_lat_bounds(self, row):
        """
        Given a row index, update row/latitude attributes.
        """
        self.row = row
        self.row_lat = self.lat_axis[self.row]
        self.nbrhood_bbox_min_lat = \
            self.row_lat - self.bbox_half_width_lat_deg
        self.nbrhood_bbox_max_lat = \
            self.row_lat + self.bbox_half_width_lat_deg
        self.prev_row = self.row
        return self

    def set_bbox_lon_bounds(self, col):
        """
        Given a column index, update column/longitude attributes.
        """
        self.col = col
        self.col_lon = self.lon_axis[self.col]
        self.nbrhood_bbox_min_lon = \
            self.col_lon - self.bbox_half_width_lon_deg
        self.nbrhood_bbox_max_lon = \
            self.col_lon + self.bbox_half_width_lon_deg
        self.prev_col = self.col
        return self


# class AzimuthalEquidistantMosaic:
#     """
#     A mosaic of azimuthal equidistant map projections for estimating distances
#     between a central location and points in its surrounding neighborhood. The
#     purpose is to enable distance calculations more accurate than great circle
#     estimates, but faster than full ellipsoid calculations.
#     """
#     def __init__(self,
#                  max_lat,
#                  min_lat,
#                  min_lon,
#                  max_lon,
#                  tile_deg=4.0):
#         self.max_lat = max_lat
#         self.min_lat = min_lat
#         self.min_lon = min_lon
#         self.max_lon = max_lon
#         self.tile_deg = tile_deg
#         self.tile_ctr_lat = np.arange(max_lat - 0.5 * tile_deg,
#                                       min_lat - 0.500001 * tile_deg,
#                                       -tile_deg)
#         self.tile_rows = len(self.tile_ctr_lat)
#         self.tile_ctr_lon = np.arange(min_lon + 0.5 * tile_deg,
#                                       max_lon + 0.500001 * tile_deg,
#                                       tile_deg)
#         self.tile_cols = len(self.tile_ctr_lon)
#         self.projections = [[Proj('+proj=aeqd +lat_0={} + lon_0={} +units=m'.
#                                   format(self.tile_ctr_lat[j],
#                                          self.tile_ctr_lon[i]))
#                              for i in range(self.tile_cols)]
#                             for j in range(self.tile_rows)]

#     def get_proj(self, lat, lon):
#         """
#         Get the azimuthal equidistance projection corresponding to (lat, lon)
#         from a mosaic.
#         """
#         # ind_flt = (self.tile_ctr_lat[0] - lat) / self.tile_deg
#         # j = np.floor(ind_flt).astype(int) \
#         #     + (ind_flt - np.floor(ind_flt) + 0.5).astype(int)
#         j = coord_to_grid_ind(lat, self.tile_ctr_lat[0], 0, self.tile_deg,
#                               invert=True)
#         # ind_flt = (lon - self.tile_ctr_lon[0]) / self.tile_deg
#         # i = np.floor(ind_flt).astype(int) \
#         #     + (ind_flt - np.floor(ind_flt) + 0.5).astype(int)
#         i = coord_to_grid_ind(lon, self.tile_ctr_lon[0], 0, self.tile_deg)
#         return self.projections[j][i]


# def ax_draw_outline(fig, ax, edgecolor='blue'):
#     bbox = ax.get_tightbbox(fig.canvas.get_renderer())
#     x0, y0, width, height = \
#         bbox.transformed(fig.transFigure.inverted()).bounds
#     print('Axis outline rectangle: ', x0, y0, width, height)
#     fig.add_artist(plt.Rectangle((x0, y0), width, height,
#                                  edgecolor=edgecolor,
#                                  linewidth=1,
#                                  alpha=0.2,
#                                  fill=False))


# def geopy_gc_km(target_lat_deg,
#                 target_lon_deg,
#                 neighborhood_lats_deg,
#                 neighborhood_lons_deg):
#     """
#     Use geopy distance.distance.great_circle to calculate the distance between
#     (scalar) (target_lat_deg, target_lon_deg) and (arrays)
#     neighborhood_lats_deg and neighborhood_lons_deg.
#     The use of radius=6367.444657 is to better mimic the original idw.pro.
#     """
#     num_points = len(neighborhood_lats_deg)

#     # return np.array([distance.great_circle((target_lat_deg, target_lon_deg),
#     #                                        (neighborhood_lats_deg[i],
#     #                                         neighborhood_lons_deg[i])).km
#     # dist_ref = np.array([distance.geodesic((target_lat_deg, target_lon_deg),
#     #                                        (neighborhood_lats_deg[i],
#     #                                         neighborhood_lons_deg[i])).km
#     #                      for i in range(num_points)])

#     # dist_gc_geopy = \
#     #     np.array([distance.great_circle((target_lat_deg, target_lon_deg),
#     #                                     (neighborhood_lats_deg[i],
#     #                                      neighborhood_lons_deg[i])).km
#     #               for i in range(num_points)])
#     # TODO: remove radius keyword
#     dist_gc_gisrs = \
#         np.array([distance.great_circle((target_lat_deg, target_lon_deg),
#                                         (neighborhood_lats_deg[i],
#                                          neighborhood_lons_deg[i]),
#                                         radius=6367.444657).km
#                   for i in range(num_points)])

#     # return np.array([distance.great_circle((target_lat_deg, target_lon_deg),
#     #                                        (neighborhood_lats_deg[i],
#     #                                         neighborhood_lons_deg[i]),
#     #                                        radius=6367.444657).km
#     #                  for i in range(num_points)])
#     return dist_gc_gisrs


# def pyproj_aeqd_km(target_lat_deg,
#                    target_lon_deg,
#                    neighborhood_lats_deg,
#                    neighborhood_lons_deg,
#                    tile_lon,
#                    tile_lat,
#                    projections):
#     pass


def gen_delta_swe_mm_map(lon_lat_bbox,
                         pt_df,
                         fig_dpi=108,
                         target_fig_dim_in=10.0,
                         title='\N{GREEK CAPITAL LETTER DELTA}SWE',
                         keep_pt_crs=False):
    """
    For a defined bounding box, plot 'D_SWE_OM' values as circular markers on
    a map. Treat correct negatives (0 - 0) with different symbology.
    """
    # TODO: change order of lon_lat_bbox from lon1, lon2, lat1, lat2 to
    # lon1, lat1, lon2, lat2 which is more consistent with what comes back
    # from pyproj.transformer.Transformer.from_crs at least.

    logger = logging.getLogger()

    debug_graphics = False
    pt_crs = ccrs.PlateCarree()

    geo_scale = max([lon_lat_bbox[1] - lon_lat_bbox[0],
                     lon_lat_bbox[3] - lon_lat_bbox[2]])
    message = f'Geographic scale is {geo_scale} degrees.'
    logger.debug(message)

    # print(f'GEO_TILE_ENGINE is {GEO_TILE_ENGINE}')
    # Set up basemap/tiling to provide a background map.
    # Select basemap source and zoom level.
    if GEO_TILE_ENGINE == 'ctx':

        geo_tiler = ctx.providers.OpenTopoMap
        geo_tiler_crs = ccrs.epsg(3857)
        # Inventory of geo_scale and geo_tiler_zoom values for OpenTopoMap
        # using contextily:
        # geo_scale geo_tiler_zoom
        # --------- ---------------
        #      6.43               7
        #      7.77               7
        #      4.37               8
        #     15.38               6
        #      6.42               7, kind of pushing it
        geo_tiler_zoom = 6
        if geo_scale < 2:
            geo_tiler_zoom = 9
        elif geo_scale < 5:
            geo_tiler_zoom = 8
        elif geo_scale < 10:
            geo_tiler_zoom = 7
        elif geo_scale < 20:
            geo_tiler_zoom = 6
        else:
            geo_tiler_zoom = 5
        geo_tiler_alpha = 0.4

    elif GEO_TILE_ENGINE == 'cimgt':

        if keep_pt_crs:
            message = 'Overriding keep_pt_crs=True for cartopy.io.img_tiles'
            logger.warning(message)
            keep_pt_crs = False

        # See https://docs.mapbox.com/api/maps/styles/#mapbox-styles
        mapbox_token = 'pk.eyJ1Ijoibm9zdGljayIsImEiOiJja29vZGxjZmowOHNsMzFtcHR6OTk1cGxiIn0.__vFGgtuD71gdL7SqKa1_g'
        # mapbox_id='navigation-night-v1'
        # mapbox_id='dark-v11'
        # mapbox_id='streets-v12'
        # mapbox_id='light-v11'
        mapbox_id='outdoors-v12'
        geo_tiler_alpha = 0.5
        geo_tiler = cimgt.MapboxTiles(access_token=mapbox_token,
                                      map_id=mapbox_id)
        geo_tiler_crs = geo_tiler.crs
        if geo_scale > 20:
            geo_tiler_zoom = 4
        elif geo_scale > 10:
            geo_tiler_zoom = 5
        elif geo_scale > 5:
            geo_tiler_zoom = 6
        else:
            geo_tiler_zoom = 7

        # # For Google Tiles the scale should be one level higher than for
        # # Mapbox.
        # geo_tiler = cimgt.GoogleTiles(style='street')
        # geo_tiler_zoom += 1

        # # For OpenStreetMaps tiles the scale should be one level higher
        # # than for Mapbox.
        # geo_tiler = cimgt.OSM()
        # geo_tiler_zoom += 1

    else:
        msg = f'GEO_TILE_ENGINE value "{GEO_TILE_ENGINE}" is invalid.'
        logger.error(msg)
        return None

    # Set the image size, estimating that about 1.5 inches are needed for the
    # legend.
    geo_aspect = (lon_lat_bbox[1] - lon_lat_bbox[0]) \
        / (lon_lat_bbox[3] - lon_lat_bbox[2])
    message = f'Geographic aspect ratio is {geo_aspect} degrees.'
    logger.debug(message)

    # A basemap will be drawn in the background of the data using ctx
    # (contextily) or cigmt (cartopy.io.img_tiles) This section establishes
    # the coordinate reference system that will be used for the map.
    # If the point data CRS (lat/lon) is used, then the basemap background
    # will be reprojected. If the basemap CRS is used, then the points will
    # be (re)projected. Always use the basemap CRS if cigmt is used by
    # forcing keep_pt_crs = False in that case (see above).
    if keep_pt_crs:
        subplots_crs = pt_crs
        basemap_display_crs = pt_crs
    else:
        subplots_crs = geo_tiler_crs
        basemap_display_crs = geo_tiler_crs

    # Calculate the map aspect.
    transformer = Transformer.from_crs(
                                       pt_crs,
                                       subplots_crs,
                                       always_xy=True,
    )

    projected_bounds = \
        transformer.transform_bounds(lon_lat_bbox[0], lon_lat_bbox[2],
                                     lon_lat_bbox[1], lon_lat_bbox[3])
    map_aspect = \
        (projected_bounds[2] - projected_bounds[0]) \
        / (projected_bounds[3] - projected_bounds[1])
    message = f'Map aspect is {map_aspect}'
    logger.debug(message)

    # These are dimensions for the elements of the figure, in inches.
    fig_margin_l_in = 0.25
    fig_margin_b_in = 0.25
    fig_margin_r_in = 0.25
    fig_margin_t_in = 0.25
    fig_gap_in = 0.25
    map_title_h_in = 0.5
    target_leg_w_in = 0.2 * target_fig_dim_in
    target_leg_h_in = 0.4 * target_fig_dim_in

    # Calculate a guess at the figure dimensions. Since we cannot predict how
    # the legend will render, this is unlikely to match the result
    # perfectly, so the figure will be adjusted after the legend has been
    # rendered.
    map_scale_in = 0.0
    fig_w_in = 0.0
    fig_h_in = 0.0
    while (fig_w_in < target_fig_dim_in) and (fig_h_in < target_fig_dim_in):
        map_scale_in += 0.001
        map_w_in = map_scale_in * map_aspect
        map_h_in = map_w_in / map_aspect
        fig_w_in = fig_margin_l_in + map_w_in + fig_gap_in \
            + target_leg_w_in + fig_margin_r_in
        content_h_in = max([map_h_in + map_title_h_in, target_leg_h_in])
        fig_h_in = fig_margin_b_in + content_h_in + fig_margin_t_in

    map_fig_size = (fig_w_in, fig_h_in)
    logger.debug(f'Requested map figure dimensions are {fig_w_in:.3f} x ' +
                 f'{fig_h_in:.3f} inches.')

    # Locate the map and legend in normalized (figure) coordinates.
    map_fig_bbox = [fig_margin_l_in / fig_w_in,
                    fig_margin_b_in / fig_h_in,
                    (fig_w_in - fig_margin_r_in) / fig_w_in,
                    (fig_h_in - fig_margin_t_in) / fig_h_in]

    # Get the space between the axes in normalized (figure) coordinates.
    wspace = fig_gap_in / (0.5 * (map_w_in + target_leg_w_in))

    # Create the figure, using GridSpec to manage the layout of the axes.
    # "s_" stands for "scatter"
    s_fig, [s_geo_ax, s_leg_ax] = \
        plt.subplots(1, 2,
                     num=increment_figure(),
                     gridspec_kw={'left': map_fig_bbox[0],
                                  'right': map_fig_bbox[2],
                                  'bottom': map_fig_bbox[1],
                                  'top': map_fig_bbox[3],
                                  'wspace': wspace,
                                  'width_ratios': [map_w_in,
                                                   target_leg_w_in],
                                  },
                     figsize=map_fig_size,
                     dpi=fig_dpi,
                     subplot_kw={'projection': subplots_crs})
                     # subplot_kw=dict(projection=subplots_crs))

    # Erase the legend axis outline and set its margins to zero.
    s_leg_ax.axis('off')
    s_leg_ax.margins(0)

    # if debug_graphics:
    #     s_fig.patch.set_edgecolor('#ff000040')
    #     s_fig.patch.set_linewidth(8)

    # Establish the geographic extents of the map.
    s_geo_ax.set_extent(lon_lat_bbox, crs=pt_crs)

    # Add the basemap.
    # See matplotlib "interpolations for imshow" documentation and
    # rasterio "rasterio.enums.Resampling" documentation.
    if GEO_TILE_ENGINE == 'ctx':
        ctx.add_basemap(s_geo_ax, crs=basemap_display_crs,
                        source=geo_tiler,
                        zoom=geo_tiler_zoom,
                        alpha=geo_tiler_alpha,
                        attribution_size=6,
                        zorder=0)
    else: # GEO_TILE_ENGINE == 'cimgt'
        s_geo_ax.add_image(geo_tiler, geo_tiler_zoom,
                           alpha=geo_tiler_alpha,
                           zorder=0)
    # Create a very basic formatter for basemap values displayed at the
    # pointer location in interactive graphics.
    def format_cursor_data(img_val):
        return f'Basemap RGBA {img_val}'
    if len(s_geo_ax.get_images()) == 1:
        basemap = s_geo_ax.get_images()[0]
        basemap.format_cursor_data = format_cursor_data

    # Add state boundaries.
    s_geo_ax.add_feature(cfNEF(category='cultural',
                               name='admin_1_states_provinces_lakes',
                               scale='10m',
                               edgecolor='black',
                               facecolor='none',
                               linewidth=0.4,
                               alpha=0.33,
                               zorder=1))

    # Note here that we are reversing the sign of pt_df['D_SWE_OM'] throughout
    # this section.
    # Plotting the points of each color/marker with a separate scatter plot
    # makes the legend more manageable and gives us more control over the
    # symbology.
    cmap = mng.delta_swe_mm()
    legend_elements = []
    num_colors = cmap['colormap'].N
    for i in range(num_colors, -2, -1):
        if i == -1:
            # Out of bounds values below the range.
            if cmap['extend'] == 'min' or cmap['extend'] == 'both':
                range_str = f'<{cmap["col_levels"][0]}'
                subset = pt_df[-pt_df['D_SWE_OM'] * 1000 <
                                  cmap['col_levels'][0]]
            else:
                continue
        elif i == num_colors:
            # out of bounds values above the range.
            if cmap['extend'] == 'max' or cmap['extend'] == 'both':
                range_str = '\N{GREATER-THAN OR EQUAL TO}' + \
                            f'{cmap["col_levels"][num_colors]}'
                subset = pt_df[-pt_df['D_SWE_OM'] * 1000 >=
                                  cmap['col_levels'][num_colors-1]]
            else:
                continue # break
        else:
            # In-range values.
            range_str = f'{cmap["col_levels"][i]} to ' + \
                f'{cmap["col_levels"][i+1]}'

            if (cmap['col_levels'][i] == 0.0) or \
               ((cmap['col_levels'][i] * cmap['col_levels'][i+1]) < 0):

                # The range of D_SWE_OM represented by this color includes
                # zeroes. Eliminate correct negatives from this subset, since
                # they will be displayed with different symbology.
                # print('range includes zero')
                # print(f'# points including CN: {len(subset)}')
                subset = pt_df[(-pt_df['D_SWE_OM'] >=
                                cmap['col_levels'][i]) & \
                               (-pt_df['D_SWE_OM'] <
                                cmap['col_levels'][i+1]) & \
                               ((pt_df['OB_SWE'] > 0.0) |
                                (pt_df['MD_SWE'] > 0.0))]
                # print(f'# points excluding CN: {len(subset)}')
                # print(subset[['OB_SWE', 'MD_SWE', 'D_SWE_OM']])
            else:
                subset = pt_df[(-pt_df['D_SWE_OM'] * 1000 >=
                                cmap['col_levels'][i]) & \
                               (-pt_df['D_SWE_OM'] * 1000 <
                                cmap['col_levels'][i+1])]

        # test_val = cmap['col_levels'][i]
        # test_col = cmap['colormap'](test_val)

        legend_elements.append(Line2D([0], [0],
                                      markerfacecolor=cmap['colormap'](i),
                                      color='w',
                                      marker='o',
                                      markersize=6,
                                      markeredgecolor='black',
                                      markeredgewidth=0.2,
                                      label=range_str))
        s_geo_ax.scatter(subset['X'], subset['Y'],
                         c=-subset['D_SWE_OM'] * 1000,
                         s=32,
                         edgecolors='black',
                         linewidth=0.2,
                         cmap=cmap['colormap'],
                         norm=cmap['norm'],
                         transform=pt_crs,
                         label=range_str,
                         zorder=2)

    # Treat correct negatives differently from zeroes.

    # Find the indices associated with zero in cmap['col_levels'].
    if sum([lev == 0.0 for lev in cmap['col_levels']]) == 1:
        # Colormap includes a zero. Identify this index.
        i = cmap['col_levels'].index(0.0)
        if i == len(cmap['col_levels']) - 1:
            i = -1
    else:
        # Colormap does not include a zero. Find the index of the first in a
        # pair of adjacent values with zero between them.
        try:
            i = [pdct < 0 for pdct in
                 [cmap['col_levels'][j] * cmap['col_levels'][j+1]
                  for j in range(len(cmap['col_levels']) - 1)]].index(True)
        except ValueError:
            i = -1
    if i != -1:
        # Find correct negatives and associate them with a special symbol.
        range_str = '0 - 0'
        subset = pt_df[(-pt_df['D_SWE_OM'] >= cmap['col_levels'][i]) & \
                       (-pt_df['D_SWE_OM'] <  cmap['col_levels'][i+1]) & \
                       (pt_df['OB_SWE'] == 0.0) & \
                       (pt_df['MD_SWE'] == 0.0)]
        legend_elements.append(Line2D([0], [0],
                                      markerfacecolor=cmap['colormap'](i),
                                      color='w',
                                      marker='X',
                                      markersize=8,
                                      markeredgecolor='black',
                                      markeredgewidth=0.2,
                                      label=range_str))
        s_geo_ax.scatter(subset['X'], subset['Y'],
                         c=-subset['D_SWE_OM'],
                         s=48,
                         marker='X',
                         edgecolors='black',
                         linewidth=0.2,
                         cmap=cmap['colormap'],
                         norm=cmap['norm'],
                         transform=pt_crs,
                         label=range_str)

    s_geo_ax.set_title(title)

    s_leg = s_leg_ax.legend(handles=legend_elements,
                            loc='center right',
                            title='\N{GREEK CAPITAL LETTER DELTA}SWE (mm)',
                            fontsize='small',
                            bbox_to_anchor=(1, 0.5),
                            borderaxespad=0
                            )

    # if debug_graphics:
    #     ax_draw_outline(s_fig, s_geo_ax, edgecolor='red')
    #     ax_draw_outline(s_fig, s_leg_ax, edgecolor='green')

    # Get the position of the s_geo_ax (the map).
    geo_ax_x0, geo_ax_y0, geo_ax_w, geo_ax_h = s_geo_ax.get_position().bounds
    # geo_ax_aspect = geo_ax_w / geo_ax_h
    if debug_graphics:
        s_fig.add_artist(plt.Rectangle((geo_ax_x0, geo_ax_y0),
                                       geo_ax_w, geo_ax_h,
                                       linewidth=0,
                                       color='red',
                                       alpha=0.5))

    # Get the position of the s_leg_ax (the legend). Note that
    # s_leg_ax.get_position().bounds is not indicative of where the legend is
    # actually drawn. It is the bounds of the axes relative to which the
    # legend is drawn (see the bbox_to_anchor keyword in s_leg_ax.legend
    # above.
    leg_ax_x0, leg_ax_y0, leg_ax_w, leg_ax_h = s_leg_ax.get_position().bounds
    if debug_graphics:
        s_fig.add_artist(plt.Rectangle((leg_ax_x0, leg_ax_y0),
                                       leg_ax_w, leg_ax_h,
                                       linewidth=0,
                                       color='green',
                                       alpha=0.5))

    # Determine the figure dimensions as drawn.
    fig_width_in, fig_height_in = s_fig.get_size_inches()
    message = f'Actual map figure dimensions are {fig_width_in:.3f} x ' + \
              f'{fig_height_in:.3f} inches.'
    logger.debug(message)

    # Calculate the actual map dimensions.
    map_w_in = geo_ax_w * fig_width_in
    map_h_in = geo_ax_h * fig_height_in
    message = f'Actual map dimensions are {geo_ax_w * fig_width_in} x ' + \
              f'{geo_ax_h * fig_height_in}'
    logger.debug(message)

    # Get the dimensions of the map + title
    s_geo_ax_bbox = s_geo_ax.get_tightbbox(s_fig.canvas.get_renderer())
    x0, y0, width, height = \
        s_geo_ax_bbox.transformed(s_fig.transFigure.inverted()).bounds
    message = 'Map + title axis outline rectangle: ' + \
              f'{x0}, {y0}, {width}, {height}'
    logger.debug(message)
    map_plus_title_h_in = height * fig_height_in
    message = f'Map + title height in inches: {map_plus_title_h_in}'
    logger.debug(message)

    # Get the bounds of the rendered legend in pixels and convert them to
    # figure coordinates and physical dimensions. Unlike
    # s_leg_ax.get_position().bounds, these will indicate the actual position
    # of the legend.
    leg_x0_px, leg_y0_px, leg_w_px, leg_h_px = \
        s_leg.get_window_extent(s_fig.canvas.get_renderer()).bounds
    leg_x0_in = leg_x0_px / fig_dpi
    leg_w_in = leg_w_px / fig_dpi
    leg_y0_in = leg_y0_px / fig_dpi
    leg_h_in = leg_h_px / fig_dpi
    leg_x0 = leg_x0_in / map_fig_size[0]
    leg_w = leg_w_in / map_fig_size[0]
    leg_y0 = leg_y0_in / map_fig_size[1]
    leg_h = leg_h_in / map_fig_size[1]
    # if debug_graphics:
    #     s_fig.add_artist(plt.Rectangle((leg_x0, leg_y0),
    #                                    leg_w, leg_h,
    #                                    linewidth=0,
    #                                    color='orange',
    #                                    alpha=0.5))
    # We can verify the legend size in inches.
    logger.debug(f'Target legend dimensions: {target_leg_w_in} x ' +
                 f'{target_leg_h_in} inches')
    logger.debug(f'Actual legend dimensions: {leg_w_in} x ' +
                 f'{leg_h_in} inches')

    # While it might be tempting, now that we know where the legend ended up,
    # to resize the figure using s_fig.set_size_inches(full_width_in,
    # full_height_in), we would also need to use
    # matplotlib.gridspec.GridSpecBase.set_width_ratios and adjust the 'left',
    # 'right', 'bottom', and 'top' grid positions.
    # Just get a look at that stuff.
    # print('Map figure bbox used to create figure: ', map_fig_bbox)
    # print(dir(s_fig))
    # print('--')
    # print(s_fig._gridspecs[0])
    # print(s_fig._gridspecs[0].get_width_ratios())

    # Option 1: recalculate the dimensions of the figure from scratch using
    # the now-known legend dimensions. This is probably a bad idea!
    # map_scale_in = 0.0
    # fig_w_in = 0.0
    # fig_h_in = 0.0
    # while (fig_w_in < target_fig_dim_in) and (fig_h_in < target_fig_dim_in):
    #     map_scale_in += 0.001
    #     map_w_in = map_scale_in * map_aspect
    #     map_h_in = map_w_in / map_aspect
    #     fig_w_in = fig_margin_l_in + map_w_in + fig_gap_in \
    #         + leg_w_in + fig_margin_r_in
    #     content_h_in = max([map_h_in + map_title_h_in, leg_h_in])
    #     fig_h_in = fig_margin_b_in + content_h_in + fig_margin_t_in
    # Option 2: recalculate the dimensions of the figure to accommodate the
    # map and legend as they have been drawn.
    fig_w_in = \
        fig_margin_l_in + map_w_in + fig_gap_in + leg_w_in + fig_margin_r_in

    # content_h_in = max([map_h_in + map_title_h_in, leg_h_in])
    content_h_in = max([map_plus_title_h_in, leg_h_in])
    fig_h_in = fig_margin_b_in + content_h_in + fig_margin_t_in

    logger.debug(f'New map figure dimensions are {fig_w_in:.3f} x ' +
                 f'{fig_h_in:.3f} inches.')

    # To resize the figure and get what we want, we will need to reposition
    # the geoaxes. Note that the bottom position is based on the height of the
    # map including the title, but the height is based only on the height of
    # the map. This is intentional.
    s_geo_ax_pos = [fig_margin_l_in / fig_w_in,
                    0.5 * (fig_h_in - map_plus_title_h_in) / fig_h_in,
                    map_w_in / fig_w_in,
                    map_h_in / fig_h_in]

    s_geo_ax.set_position(s_geo_ax_pos)
    s_fig.set_size_inches(fig_w_in, fig_h_in)
    # This stops working sometime between matplotlib 3.4.3 and 3.7.2. I
    # cannot figure out a way to do it right now.
    # s_fig._gridspecs[0].set_width_ratios([map_w_in, leg_w_in])
    # print(s_fig._gridspecs[0].get_width_ratios())
    # plt.draw()

    message = f'Map should be {map_w_in} x {map_h_in} inches'
    logger.debug(message)
    message = f'Legend should be {leg_w_in} x {leg_h_in} inches'
    logger.debug(message)

    # Get the bounds of the rendered legend in pixels and convert them to
    # figure coordinates and physical dimensions. Unlike
    # s_leg_ax.get_position().bounds, these will indicate the actual position
    # of the legend.
    leg_x0_px, leg_y0_px, leg_w_px, leg_h_px = \
        s_leg.get_window_extent(s_fig.canvas.get_renderer()).bounds
    leg_x0_in = leg_x0_px / fig_dpi
    leg_w_in = leg_w_px / fig_dpi
    leg_y0_in = leg_y0_px / fig_dpi
    leg_h_in = leg_h_px / fig_dpi
    leg_x0 = leg_x0_in / map_fig_size[0]
    leg_w = leg_w_in / map_fig_size[0]
    leg_y0 = leg_y0_in / map_fig_size[1]
    leg_h = leg_h_in / map_fig_size[1]
    if debug_graphics:
        s_fig.add_artist(plt.Rectangle((leg_x0, leg_y0),
                                       leg_w, leg_h,
                                       linewidth=0,
                                       color='orange',
                                       alpha=0.5))
    # We can verify the legend size in inches.
    logger.debug(f'Target legend dimensions: {target_leg_w_in} x ' +
                 f'{target_leg_h_in} inches')
    logger.debug(f'Actual legend dimensions: {leg_w_in} x ' +
                 f'{leg_h_in} inches')

    # Get the bounds of the rendered geoaxes in figure coordinates and convert
    # them to physical dimensions.
    geo_x0, geo_y0, geo_w, geo_h = s_geo_ax.get_position().bounds
    if debug_graphics:
        s_fig.add_artist(plt.Rectangle((geo_x0, geo_y0),
                                       geo_w, geo_h,
                                       linewidth=0,
                                       color='blue',
                                       alpha=0.5))

    # geo_x0_in = geo_x0 * map_fig_size[0]
    # geo_w_in = geo_w * map_fig_size[0]
    # geo_y0_in = geo_y0 * map_fig_size[1]
    # geo_h_in = geo_h * map_fig_size[1]

    # TODO:
    # Get the bounds of the title on the geoaxes. Need to include space for
    # this when resizing the figure.

    # TODO:
    # For the full_width_in, I need to first reset the position of the
    # s_geo_ax based on what tight_layout did so when the new figure is drawn
    # it is not centered and there is room for the legend.

    return s_fig


def parse_args():
    """
    Parse command line arguments.
    """
    version = '0.2'
    description = 'Interpolate snowpack differences to a raster.'
    parser = argparse.ArgumentParser(
        description=description
        )
    parser.add_argument('-v', '--version',
                        action='version',
                        version=f'%(prog)s {version}')
    parser.add_argument('-c', '--cross_validate',
                        action='store_true',
                        dest='cross_validate',
                        help='Estimate errors using cross validation')
    parser.add_argument('-d', '--dry_run',
                        action='store_true',
                        dest='dry_run',
                        help='Exit after cross validation to estimate errors')
    parser.add_argument('-n', '--no_layer',
                        action='store_true',
                        dest='no_layer',
                        help='Exit before creating/replacing an output layer')
    parser.add_argument('-b', '--no_database',
                        action='store_true',
                        dest='no_database',
                        help='Run without accessing the database')
    parser.add_argument('-g', '--gui',
                        action='store_true',
                        dest='gui',
                        help='Display graphics during program execution')
    parser.add_argument('-q', '--quick_look',
                        action='store_true',
                        dest='quick_look',
                        help='View points and nudging layer ' + \
                             '(if one exists) and exit')
    parser.add_argument('-k', '--skip_map',
                        action='store_true',
                        dest='skip_map',
                        help='Skip generating a map of observation points')
    parser.add_argument('-f', '--file_location',
                        nargs=1,
                        dest='file_location',
                        metavar='{directory}',
                        help='Set directory location of input/output data')
    parser.add_argument('start_date_YmdH',
                        type=str,
                        metavar='{start date %YmdH}',
                        help='Input data start date/time as YYYYmmddHH')
    parser.add_argument('stop_date_YmdH',
                        type=str,
                        metavar='{stop date %YmdH}',
                        help='Input data stop date/time as YYYYmmddHH')
    parser.add_argument('-i', '--idw_params',
                        nargs=5,
                        dest='idw_params',
                        metavar=('{min. # points}',
                                 '{max. # points}',
                                 '{horiz. power}',
                                 '{elev. power}',
                                 '{max. distance, km}'),
                        help='Use the supplied IDW parameters')
    parser.add_argument('region_name',
                        type=str,
                        metavar='{region name}',
                        help='Region name, ' +
                        'typically "east", "west", "central", or "us".')

    # Arguments below added for compatibility with ops_logger.
    # Originally that has -v (--verbose), -q (--quiet), -s (--syslog),
    # -l (--log) -o (--stdout), -e (--stderr)
    # I am already using -v and -q, but -s, -l, -o, and -e are available.
    # Log message filtering by level.
    parser.add_argument('-V',
                        '--verbose',
                        action='store_true',
                        dest='verbose',
                        help='Display all info messages')
    parser.add_argument('-Q',
                        '--quiet',
                        action='store_true',
                        dest='quiet',
                        help='Display only error messages')

    # Additional logging options.
    parser.add_argument('-s',
                        '--syslog',
                        dest='log_facility',
                        action='store',
                        type=str,
                        help='Route logging messages to syslog facility FACILITY',
                        metavar='FACILITY')
    parser.add_argument('-l',
                        '--log',
                        dest='log_file',
                        action='store',
                        type=str,
                        help='Route logging messages to log file LOGFILE',
                        metavar='LOGFILE')
    parser.add_argument('-o',
                        '--stdout',
                        action='store_true',
                        dest='stdout',
                        help='Route logging messages to stdout')
    parser.add_argument('-e',
                        '--stderr',
                        action='store_true',
                        dest='stderr',
                        help='Route logging messages to stderr')
    args = parser.parse_args()
    return args


def check_valid_YmdH(date_YmdH):
    """
    Verify that a date/time formatted as %Y%m%d%H is valid.
    """
    try:
        dt.datetime.strptime(date_YmdH, '%Y%m%d%H')
    except ValueError:
        return False
    return True


def ops_config():
    """
    Test the system and infer whether or not it is configured for SNODAS
    operations, based on the presence/absence/validity of a few key
    environment variables.
    """
    logger = logging.getLogger()
    nsa_prefix = os.getenv('NSA_PREFIX')
    if nsa_prefix is None:
        return False
    if not os.path.isdir(nsa_prefix):
        message = 'NSA_PREFIX={nsa_prefix} defined, but not a directory. ' + \
                  'Assuming this an offline system not configured for ' + \
                  'operations.'
        logger.warning(message)
    pg_host = os.getenv('PGHOST')
    if pg_host is None:
        return False
    pg_db = os.getenv('PGDATABASE')
    if pg_db is None:
        return False
    gisrs_schema = os.getenv('GISRS_SCHEMA')
    if gisrs_schema is None:
        return False

    return True


def check_args(args):
    """
    Check the validity of command line arguments.
    """
    logger = logging.getLogger()
    all_args_ok = True
    original_cv = args.cross_validate

    if args.dry_run and not args.cross_validate:
        message = 'No [-c, --cross_validate] argument provided, but ' + \
            'assumed because [-d, --dry_run] has been indicated.'
        logger.warning(message)
        args.cross_validate = True

    if not check_valid_YmdH(args.start_date_YmdH):
        logger.error('Invalid start date/time "%s"', args.start_date_YmdH)
        all_args_ok = False

    if not check_valid_YmdH(args.stop_date_YmdH):
        logger.error('Invalid stop date/time "%s"', args.stop_date_YmdH)
        all_args_ok = False

    if args.quick_look:
        # Generate warnings about any incompatible arguments.
        if original_cv \
            or args.dry_run \
            or args.no_layer \
            or args.idw_params:
            irrelevant_args = []
            if original_cv:
                irrelevant_args.append('-c')
                args.cross_validate = False
            if args.dry_run:
                irrelevant_args.append('-d')
                args.dry_run = False
            if args.no_layer:
                irrelevant_args.append('-n')
                args.no_layer = False
            if args.skip_map:
                irrelevant_args.append('-k')
                args.skip_map = False
            if args.idw_params:
                irrelevant_args.append('-i')
                args.idw_params = None
            message = 'Ignoring argument/s ' + \
                      f'{", ".join(irrelevant_args)} ' + \
                      'not relevant to the -q option.'
            logger.warning(message)

    if not ops_config() and args.file_location is None:
        message = 'This appears to be a non-operational system; ' + \
                  'the [-f, --file_location] argument must be set.'
        logger.error(message)
        all_args_ok = False

    if ops_config() and args.file_location is not None:
        message = 'This appears to be an operational system, but ' + \
                  f'all data will be read from {args.file_location[0]} ' + \
                  'rather than operational locations.'
        logger.warning(message)

    if not ops_config() and not args.no_database:
        message = 'No [-b, --no_database] argument provided, but ' + \
                  'assumed because this system appears to be ' + \
                  'non-operational.'
        # ALERT - WE CAN"T DO THIS!
        args.no_database = True
        logger.warning(message)

    if args.dry_run:
        if args.no_layer:
            message = 'Ignoring argument -n not relevant to the -d option.'
            logger.warning(message)

    if args.idw_params is not None:

        # Check the validity of the idw_params while converting them to the
        # expected types; store as individual args attributes.
        try:
            args.min_num_points = int(args.idw_params[0])
            if (args.min_num_points < 2 or args.min_num_points > 10):
                message = 'Minimum # points must be in [2, 10]; ' + \
                          f'{args.min_num_points} is invalid.'
                logger.error(message)
                args.min_num_points = 2 # just so we can finish check_args
                all_args_ok = False
        except ValueError:
            message = 'IDW parameter {min. # points}: ' + \
                      f'invalid int value "{args.idw_params[0]}"'
            logger.error(message)
            args.min_num_points = 2
            all_args_ok = False

        try:
            args.max_num_points = int(args.idw_params[1])
            if args.max_num_points == -1:
                args.max_num_points = None
            elif (args.max_num_points < args.min_num_points or
                  args.max_num_points > 100):
                message = 'Maximum # points must be -1 or in ' + \
                          f'[{args.min_num_points}, 100]; ' \
                          f'{args.max_num_points} is invalid.'
                logger.error(message)
                all_args_ok = False
        except ValueError:
            message = 'IDW parameter {max. # points}: ' + \
                      f'invalid int value "{args.idw_params[1]}"'
            logger.error(message)
            all_args_ok = False

        try:
            args.horizontal_power = float(args.idw_params[2])
            if (args.horizontal_power < 0.0 or args.horizontal_power > 10.0):
                message = 'Power for horizontal distance must be in ' + \
                          f'[0.0, 10.0]; {args.horizontal_power} is invalid.'
                logger.error(message)
                all_args_ok = False
        except ValueError:
            message = 'IDW parameter {horiz. power}: ' + \
                      f'invalid float value "{args.idw_params[2]}"'
            logger.error(message)
            all_args_ok = False

        try:
            args.elevation_power = float(args.idw_params[3])
            if (args.elevation_power < 0.0 or args.elevation_power > 10.0):
                message = 'Power for elevation (vertical) difference ' + \
                          'must be in [0.0, 10.0]; ' + \
                          f'{args.elevation_power} is invalid.'
                logger.error(message)
                all_args_ok = False
        except ValueError:
            message = 'IDW parameter {elev. power}: ' + \
                      f'invalid float value "{args.idw_params[3]}"'
            logger.error(message)
            all_args_ok = False

        try:
            args.max_distance_km = float(args.idw_params[4])
            if (args.max_distance_km < 1.0 or args.max_distance_km > 250.0):
                message = 'Maximum distance over which to interpolate ' + \
                          'must be in [1.0, 250.0]; ' + \
                          f'{args.max_distance_km} is invalid.'
                logger.error(message)
                all_args_ok = False
        except ValueError:
            message = 'IDW parameter {max. distance, km}: ' + \
                      f'invalid float value "{args.idw_params[4]}"'
            logger.error(message)
            all_args_ok = False

    else:
        # Create empty placeholders for typed values of args.idw_params.
        args.min_num_points = None
        args.max_num_points = None
        args.horizontal_power = None
        args.elevation_power = None
        args.max_distance_km = None

    if not all_args_ok:
        return False

    return True, args


def read_idw_points(point_data_path):
    """
    Read data from observing stations and other discrete point locations that
    will be analyized to generate a grid of SWE adjustments.
    """
    pt_df = gpd.read_file(point_data_path)

    # Replace the missing value set by get_assim_data.pro with NaN.
    pt_df = pt_df.replace(99999, np.nan)
    pt_df = pt_df.replace(-99999, np.nan)

    # Rename columns to match expected conventions.
    pt_df.rename({'STATION_I': 'STATION_ID',
                  'STATION_T': 'STATION_TY',
                  'STATION_E': 'STATION_EL',
                  'DEM_ELE': 'DEM_ELEVAT',
                  'FOREST_': 'FOREST_DEN',
                  'TRUE_FL': 'TRUE_FLAG',
                  'OB_SWE_': 'OB_SWE_T',
                  'MD_SWE_': 'MD_SWE_T',
                  'D_SWE_M': 'D_SWE_MM',
                  'OB_DEPTH_': 'OB_DEPTH_T',
                  'MD_DEPTH_': 'MD_DEPTH_T',
                  'D_DEPTH_M': 'D_DEPTH_MM',
                  'OB_DENS': 'OB_DENSITY',
                  'MD_DENS': 'MD_DENSITY',
                  'D_DEN_M': 'D_DEN_MM',
                  'D_S_P_O': 'D_S_P_OM',
                  'D_NSP_O': 'D_NSP_OM',
                  'OB_NSN_': 'OB_NSN_P',
                  'MD_NSN_': 'MD_NSN_P'},
                  axis='columns',
                  errors='ignore',
                  inplace=True)

    return pt_df


def verify_get_assim_data_conventions(pt_df):
    """
    Confirm that the station data being analyzed follows the conventions
    the get_assim_data program is expected to follow.
    """
    logger = logging.getLogger()
    # df_sub = pt_df.filter(items=['OB_SWE',
    #                              'OB_DEPTH',
    #                              'OB_DENSITY',
    #                              'TRUE_FLAG',
    #                              'MD_SWE',
    #                              'MD_DEPTH',
    #                              'MD_DENSITY',
    #                              'D_SWE_OM',
    #                              'D_DEPTH_OM']).copy()

    # If OB_DENSITY is NaN, and both OB_SWE and OB_DEPTH are valid and
    # nonzero, then it must be that one or the other is derived using
    # MD_DENSITY. Verify this is the case.
    # This command inspires a "NumExpr defaulting to 8 threads" message.
    not_true = \
        (pt_df['OB_SWE'].notna()) & \
        (pt_df['OB_SWE'] > 0) & \
        (pt_df['OB_DEPTH'].notna()) & \
        (pt_df['OB_DEPTH'] > 0) & \
        (pt_df['OB_DENSITY'].isna())
    sim_ob_df = pt_df[not_true].copy()

    # # Make sure TRUE_FLAG is not set to 1 for any of these.
    # if (sim_ob_df['TRUE_FLAG'] == 1).sum() != 0:
    #     logger.error('OB_SWE - OB_DEPTH - TRUE_FLAG inconsistency.')
    #     return None

    # Make sure TRUE_FLAG is set to -1 for all of these.
    if (sim_ob_df['TRUE_FLAG'] == -1).sum() != len(sim_ob_df):
        logger.error('OB_SWE - OB_DEPTH - TRUE_FLAG inconsistency.')
        return None

    # Verify that the "observed" density derived from OB_SWE and OB_DEPTH
    # matches (roughly) the MD_DENSITY.
    ob_density_derived = sim_ob_df['OB_SWE'] / sim_ob_df['OB_DEPTH'] * 1000
    max_density_diff = \
        (ob_density_derived - sim_ob_df['MD_DENSITY']).abs().max()
    if max_density_diff > 1.0:
        # print(max_density_diff)
        # print(ob_density_derived)
        # print(sim_ob_df['MD_DENSITY'])
        message = 'Inconsistent OB_SWE - OB_DEPTH - MD_DENSITY for ' + \
            'data with TRUE_FLAG == -1.'
        logger.error(message)
        return None

    # Check the "not use model snow density if the model SWE is small" oddity
    # from idw.pro. This is just a case of something that probably belongs in
    # get_assim_data.pro. The logic that follows seems to assume that the
    # existing D_SWE_OM value was generated from a simulated OB_SWE derived
    # from a real OB_DEPTH using MD_DENSITY. It effectively backs out the
    # observed depth implicitly by deriving D_DEPTH_OM (though why this is
    # necessary is a mystery since D_DEPTH_OM is already in the shapefile,
    # albeit with the incorrect sign). Then it implicitly recalculates OB_SWE
    # using 250 km/m instead of MD_DENSITY, but does so by directly
    # calculating a new D_SWE_OM.
    snow_density_cap = 250.0
    iffy_md_density = \
        (pt_df['TRUE_FLAG'] == -1) & \
        (pt_df['MD_SWE'] < 0.01) & \
        (pt_df['MD_DENSITY'] > snow_density_cap)
    if iffy_md_density.sum() != 0:
        message = 'Limiting MD_DENSITY and recalculating deltas for ' + \
                  f'{iffy_md_density.sum()} simulated (depth-based) ' + \
                  'SWE observations.'
        logger.info(message)

        # Recalculate D_DEPTH_OM, but with the correct sign
        # (get_assim_data.pro reverses the sign, an old mistake).
        delta_depth = \
            1000.0 * \
            pt_df[iffy_md_density]['D_SWE_OM'] / \
            pt_df[iffy_md_density]['MD_DENSITY']

        pt_df.loc[iffy_md_density, 'D_SWE_OM'] = \
            snow_density_cap * delta_depth / 1000.0

    return pt_df


def get_gisrs_layer_data_file_path(layer_name):
    """
    Get the path to the data file for a GISRS layer.
    """
    logger = logging.getLogger()
    # Open the operations database.
    pg_host = os.getenv('PGHOST')
    if pg_host is None:
        logger.error('Missing environment variable PGHOST.')
        #pg_host = 'opps-db'
        return None
    pg_db = os.getenv('PGDATABASE')
    if pg_db is None:
        logger.error('Missing environment variable PGDATABASE.')
        #pg_db = 'operations'
        return None
    gisrs_schema = os.getenv('GISRS_SCHEMA')
    if gisrs_schema is None:
        logger.error('Missing environment variable GISRS_SCHEMA.')
        #gisrs_schema = 'gisrs'
        return None
    conn_string = f'host=\'{pg_host}\' dbname={pg_db}'
    conn = psycopg2.connect(conn_string)
    conn.set_client_encoding('utf-8')
    cursor = conn.cursor()
    # Get the table where the layer resides.
    sql_cmd = \
        'SELECT tabname::varchar ' + \
        f'FROM {gisrs_schema}.product ' + \
        f'WHERE layer = \'{layer_name}\';'
    logger.debug('psql command: "%s"', sql_cmd)
    cursor.execute(sql_cmd)
    table_name = cursor.fetchall()
    if len(table_name) == 0:
        return None
    if len(table_name) != 1:
        logger.error('Multiple tables found for product "%s".',
                     layer_name)
        return None
    table_name = table_name[0][0]
    # TODO: arguments to queries should be given as parameters
    sql_cmd = \
        'SELECT data_file_pathname::varchar ' + \
        f'FROM {gisrs_schema}.{table_name} ' + \
        f'WHERE gis_layer = \'{layer_name}\';'
    logger.debug('psql command: %s', sql_cmd)
    cursor.execute(sql_cmd)
    layer_data_file_path = cursor.fetchall()
    if len(layer_data_file_path) == 0:
        return None
    if len(layer_data_file_path) != 1:
        logger.error('Multiple path names found for layer "%s".',
                     layer_name)
        return None
    layer_data_file_path = layer_data_file_path[0][0]
    cursor.close()
    conn.close()
    return os.path.normpath(layer_data_file_path)


def write_nc_nudging_raster(north_down_grid,
                            lat_axis,
                            lat_res,
                            lon_axis,
                            lon_res,
                            long_name,
                            units,
                            start_datetime,
                            stop_datetime,
                            create_module,
                            create_comment,
                            file_path,
                            color_table_path=None,
                            missing_value=-99999.0):
    """
    Write a geographic longitude/latitude raster dataset for SNODAS
    assimilation (nudging) to a netCDF file.
    The grid orientation should be north-down, the lat_axis should be
    consistent with that (i.e. in descending order), and the lat_res should
    NOT be negative.
    """
    # logger = logging.getLogger()

    if not np.ma.isMaskedArray(north_down_grid):
        raise TypeError('The input grid must be of the ' +
                        'np.ma.MaskedArray class.')
    # Set masked values to a more conventional no-data value, in case they
    # are NaN. Use a copy to protect the contents of the original array.
    north_down_grid = np.ma.copy(north_down_grid)
    masked_ind = np.where(np.ma.getmaskarray(north_down_grid) == True)
    north_down_grid[masked_ind] = missing_value
    north_down_grid = np.ma.masked_where(north_down_grid == missing_value,
                                         north_down_grid)
    # print(f'data type: {north_down_grid.dtype}')

    num_rows, num_cols = north_down_grid.shape
    if len(lat_axis) != num_rows:
        raise GridInconsistency('The latitude axis is inconsistent with ' +
                                'the data array rows.')
    if len(lon_axis) != num_cols:
        raise GridInconsistency('The longitude axis is inconsistent with ' +
                                'the data array columns.')

    nc_dataset = Dataset(file_path, 'w', format='NETCDF4')
    lat_dim = nc_dataset.createDimension('lat', num_rows)
    lon_dim = nc_dataset.createDimension('lon', num_cols)
    nv_dim = nc_dataset.createDimension('nv', 2)

    lat_var = nc_dataset.createVariable('lat', 'f8', ('lat'))
    lat_var.bounds = 'lat_bounds'
    lat_var.long_name = 'latitude'
    lat_var.units = 'degrees_north'
    lat_var.standard_name = 'latitude'
    lat_var.resolution = lat_res
    lat_var.origin_offset = 0.5 * lat_res
    lat_var[:] = lat_axis

    lon_var = nc_dataset.createVariable('lon', 'f8', ('lon'))
    lon_var.bounds = 'lon_bounds'
    lon_var.long_name = 'longitude'
    lon_var.units = 'degrees_east'
    lon_var.standard_name = 'longitude'
    lon_var.resolution = lon_res
    lon_var.origin_offset = 0.5 * lon_res
    lon_var[:] = lon_axis

    lat_bounds_var = nc_dataset.createVariable('lat_bounds', 'f8',
                                               ('lat', 'nv'))
    lat_bounds_var[:] = np.stack((lat_axis + 0.5 * lat_res,
                                  lat_axis - 0.5 * lat_res)).T

    lon_bounds_var = nc_dataset.createVariable('lon_bounds', 'f8',
                                               ('lon', 'nv'))
    lon_bounds_var[:] = np.stack((lon_axis - 0.5 * lon_res,
                                  lon_axis + 0.5 * lon_res)).T

    data_var = nc_dataset.createVariable('Data', 'f4', ('lat', 'lon'),
                                         # compression='zlib',
                                         zlib=True,
                                         fill_value=missing_value)
    data_var.no_data_value = missing_value
    data_var.scale_factor = 1.0
    data_var.add_offset = 0.0
    data_var.grid_mapping = 'crs'
    data_var.institution = 'NOHRSC'
    data_var.satellite_data = 'no'
    data_var.long_name = long_name
    data_var.thematic = 'no'
    data_var.units = units
    data_var.gisrs_product_code = np.int32(-1)
    data_var.minimum_data_value = north_down_grid.min()
    data_var.maximum_data_value = north_down_grid.max()
    data_var.start_date = start_datetime.strftime('%Y-%m-%d %H:%M:%S')
    data_var.stop_date = stop_datetime.strftime('%Y-%m-%d %H:%M:%S')
    if color_table_path is not None:
        data_var.number_of_color_tables = np.int32(1)
        data_var.color_table_descriptor = 'Default'
        data_var.color_table_file = color_table_path
    else:
        data_var.number_of_color_tables = np.int32(0)
    data_var.standard_name = 'Not applicable'
    data_var.data_are_elevations = 'no'
    data_var[:] = north_down_grid

    crs_var = nc_dataset.createVariable('crs', 'i4')
    crs_var.grid_mapping_name = 'latitude_longitude'
    crs_var.longitude_of_prime_meridian = 0.0
    crs_var.horizontal_datum = 'WGS84'
    wgs84_semi_major_axis_m = 6378137.0
    wgs84_semi_minor_axis_m = 6356752.314245
    wgs84_inverse_flattening = \
        1.0 / (1.0 - wgs84_semi_minor_axis_m / wgs84_semi_major_axis_m)
    crs_var.semi_major_axis = wgs84_semi_major_axis_m
    crs_var.semi_minor_axis = wgs84_semi_minor_axis_m
    crs_var.inverse_flattening = wgs84_inverse_flattening

    create_date_str = dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    create_module_str = \
        f'{create_date_str} created by module: {create_module}'
    nc_dataset.setncattr_string('history', create_module_str)
    create_comment_str = \
        f'{create_date_str} created comment: {create_comment}'
    nc_dataset.setncattr_string('comment', create_comment_str)
    nc_dataset.setncattr_string('format_version',
                                'NOHRSC NetCDF faster file v1.2')
    nc_dataset.setncattr_string('Conventions', 'CF-1.6')
    nc_dataset.setncattr_string('references', 'Not applicable')
    nc_dataset.setncattr_string('source', 'Not applicable')
    nc_dataset.setncattr_string('title', 'Not applicable')

    nc_dataset.close()

    # return


def get_nc_geo_raster(file_path, var_name):
    """
    Read a geographic longitude/latitude raster dataset from a NetCDF file.
    """
    logger = logging.getLogger()

    nc_dataset = Dataset(file_path)
    nc_var = nc_dataset[var_name]

    # Get geographic axes, limits, and resolution.
    #print(nc_dataset['lat'].get_dims())
    for dim in nc_dataset.dimensions:
        # print(dim)
        # print(nc_dataset.dimensions[dim].size)
        # print(nc_dataset.dimensions[dim].name)
        # if (nc_dataset.dimensions[dim].name == 'lat' or
        #     nc_dataset.dimensions[dim].name == 'Lat' or
        #     nc_dataset.dimensions[dim].name == 'latitude' or
        #     nc_dataset.dimensions[dim].name == 'Latitude'):
        if nc_dataset.dimensions[dim].name in \
            ('lat', 'Lat', 'latitude', 'Latitude'):
            lat_name = nc_dataset.dimensions[dim].name
            lat_dim_size = nc_dataset.dimensions[dim].size
        # elif(nc_dataset.dimensions[dim].name == 'lon' or
        #      nc_dataset.dimensions[dim].name == 'Lon' or
        #      nc_dataset.dimensions[dim].name == 'longitude' or
        #      nc_dataset.dimensions[dim].name == 'Longitude'):
        elif nc_dataset.dimensions[dim].name in \
            ('lon', 'Lon', 'longitude', 'Longitude'):
            lon_name = nc_dataset.dimensions[dim].name
            lon_dim_size = nc_dataset.dimensions[dim].size
        else:
            pass

    lat_axis = nc_dataset[lat_name][:]
    if len(lat_axis) != lat_dim_size:
        logger.error('Latitude coordinate variable mismatch.')
        return None, None, None
    lon_axis = nc_dataset[lon_name][:]
    if len(lon_axis) != lon_dim_size:
        logger.error('Longitude coordinate variable mismatch.')
        return None, None, None

    min_lon = nc_dataset[lon_name + '_bounds'][:,:][0,0]
    min_lon_ctr = lon_axis[0]
    max_lon_ctr = lon_axis[-1]
    max_lon = nc_dataset[lon_name + '_bounds'][:,:][-1,1]
    max_lat = nc_dataset[lat_name + '_bounds'][:,:][0,0]
    max_lat_ctr = lat_axis[0]
    min_lat_ctr = lat_axis[-1]
    min_lat = nc_dataset[lat_name + '_bounds'][:,:][-1,1]
    if min_lat_ctr > max_lat_ctr:
        logger.error('No support for north-up rasters.')
        return None, None, None

    nc_header = \
        {'number_of_columns': lon_dim_size,
         'lon_resolution': nc_dataset[lon_name].getncattr('resolution'),
         'min_lon': min_lon,
         'min_lon_ctr': min_lon_ctr,
         'max_lon_ctr': max_lon_ctr,
         'max_lon': max_lon,
         'number_of_rows': lat_dim_size,
         'lat_resolution': nc_dataset[lat_name].getncattr('resolution'),
         'max_lat': max_lat,
         'max_lat_ctr': max_lat_ctr,
         'min_lat_ctr': min_lat_ctr,
         'min_lat': min_lat,
         'no_data_value': nc_var.getncattr('no_data_value')}

    # if nc_var.dtype is not np.dtype('float32'):
    #     print(nc_var.dtype.name)
    #     logger.error('"%s" variable from %s is of unexpected type.',
    #                  var_name, file_path)
    #     return None, None
    # print(file_path, type(nc_var[:]), nc_var[:][0,0])

    return nc_var, nc_header, nc_dataset


def coord_to_grid_ind(coord, benchmark_coord, benchmark_ind, resolution,
                      invert=False):
    """
    General x-to-column, y-to-row function for numpy arrays. Use invert=True
    for row-to-y in north-down orientations.
    """
    # Calculate the floating point index relative to the benchmark (reference
    # cell center) coordinate.
    if invert:
        ind = (benchmark_coord - coord) / resolution
    else:
        ind = (coord - benchmark_coord) / resolution
    return benchmark_ind + \
        np.floor(ind).astype(int) + (ind - np.floor(ind) + 0.5).astype(int)


def idw_for_cross_val_loc(IDW,
                          ind,
                          pt_is_real,
                          pt_lat,
                          pt_lon,
                          pt_dem_elev,
                          pt_ind,
                          pt_z_val):
    """
    Perform one iteration of IDW for cross validation output.
    """

    # Only perform cross validation at real observation points.
    if not pt_is_real[ind]:
        return None

    # Find all points in a bounding box approximately (slightly larger than)
    # the size of the neighborhood defined by the point location and
    # max_distance_km, approximated by IDW.bbox_half_width_[lat,lon]_deg
    nbrhood_bbox_min_lat = pt_lat[ind] - IDW.bbox_half_width_lat_deg
    nbrhood_bbox_max_lat = pt_lat[ind] + IDW.bbox_half_width_lat_deg
    nbrhood_bbox_min_lon = pt_lon[ind] - IDW.bbox_half_width_lon_deg
    nbrhood_bbox_max_lon = pt_lon[ind] + IDW.bbox_half_width_lon_deg
    in_nbrhood_bbox_tf = \
        (pt_lat >= nbrhood_bbox_min_lat) & \
        (pt_lat <= nbrhood_bbox_max_lat) * \
        (pt_lon >= nbrhood_bbox_min_lon) & \
        (pt_lon <= nbrhood_bbox_max_lon) & \
        (pt_ind != ind)
    num_bbox_neighbors = in_nbrhood_bbox_tf.sum()
    # box_time_elapsed += time.perf_counter() - t1
    if num_bbox_neighbors < IDW.min_num_points:
        IDW.num_too_few_obs += 1
        return None

    # Extract bounding box data into new arrays.
    nbrhood_bbox_pt_is_real = pt_is_real[in_nbrhood_bbox_tf]
    nbrhood_bbox_pt_lat = pt_lat[in_nbrhood_bbox_tf]
    nbrhood_bbox_pt_lon = pt_lon[in_nbrhood_bbox_tf]
    nbrhood_bbox_pt_dem_elev = pt_dem_elev[in_nbrhood_bbox_tf]
    nbrhood_bbox_pt_z_val = pt_z_val[in_nbrhood_bbox_tf]

    if IDW.analysis_method == 'classic':
        IDWMethod = IDW.classic
    elif IDW.analysis_method == 'with_elev_regression':
        IDWMethod = IDW.with_elev_regression
    else:
        return None

    z_val_interp = \
        IDWMethod(pt_lat[ind],
                  pt_lon[ind],
                  pt_dem_elev[ind],
                  nbrhood_bbox_pt_is_real,
                  nbrhood_bbox_pt_lat,
                  nbrhood_bbox_pt_lon,
                  nbrhood_bbox_pt_dem_elev,
                  nbrhood_bbox_pt_z_val)

    return z_val_interp


def round_half_up(x):
    """
    Rounding function to use if banker rounding (the default in python) is not
    desired.
    From user GaloisPlusPlus at StackOverflow:
    https://stackoverflow.com/questions/28617841/rounding-to-nearest-int-with-numpy-rint-not-consistent-for-5
    """
    round_lambda = lambda z: (int(z > 0) - int(z < 0)) * int(abs(z) + 0.5)
    if isinstance(x, (np.ndarray, np.generic)):
        return np.vectorize(round_lambda)(x)
    return round_lambda(x)


def multi_ols_reg(pt_dataframe,
                  y_col,
                  x_cols,
                  x_col_names=None,
                  alpha=0.1):
    """
    Perform multiple regression of pt_dataframe[y_col] with
    pt_dataframe[x_cols] as multiple independent (exogenous) variables using
    the statsmodels package.
    """

    # logger = logging.getLogger()

    if x_col_names is None:
        x_col_names = x_cols
    ols_y = pt_dataframe[y_col]
    ols_x = pt_dataframe[x_cols]
    x_cols = ['const'] + x_cols
    x_col_names = ['intercept'] + x_col_names

    # Perform the initial fit.
    ols_result = sm.OLS(ols_y, ols_x).fit()

    # Refine the regression, eliminating variables with P(>|t|) less than
    # 0.1, starting from the largest, not including the intercept.
    coefs = ols_result.params
    p_vals = ols_result.pvalues # Indexed by x_col_names
    max_p_val_col = p_vals[1:].idxmax() # This is from x_col_names
    # print(x_col_names)
    # print(p_vals)
    # print(max_p_val_col)

    # As long as >1 independent variables remain and the maximum
    # p-value is still >= alpha (commonly 0.05, but we default to 0.1 here),
    # eliminate the independent variable with the highest p_vals.
    while p_vals[max_p_val_col] >= alpha and len(p_vals) > 2:

        # Eliminate the variable with the largest p-value (the variable most
        # likely to not have influence).
        #message = f'Removing "{max_p_val_col}" from the regression.'
        #logger.info(message)
        remove_ind = x_cols.index(max_p_val_col)
        del x_cols[remove_ind]
        del x_col_names[remove_ind]

        # Remove the intercept.
        x_cols = x_cols[1:]
        x_col_names = x_col_names[1:]

        # Put the intercept back.
        ols_x = pt_dataframe[x_cols]
        ols_x = sm.add_constant(ols_x)
        x_cols = ['const'] + x_cols
        x_col_names = ['intercept'] + x_col_names
        ols_result = sm.OLS(ols_y, ols_x).fit()
        # print(ols_result.summary())
        coefs = ols_result.params
        p_vals = ols_result.pvalues
        max_p_val_col = p_vals[1:].idxmax()
        # print(x_col_names)
        # print(p_vals)
        # print(max_p_val_col)

    return x_cols, coefs, ols_result


def get_delta_swe_mm_symbol(delta_swe_obs_minus_mdl_mm):
    """
    Define a symbol (marker, markersize, and markercolor) to represent
    a value of ΔSWE.
    """
    marker='o'
    markersize=5.0
    markercolor='red'
    if delta_swe_obs_minus_mdl_mm < -500:
        markercolor='#980064' # 152 0 100
    elif delta_swe_obs_minus_mdl_mm < -250:
        markercolor='#cf004b' # 207 0 75
    elif delta_swe_obs_minus_mdl_mm < -100:
        markercolor='#f67100' # 246 113 0
    elif delta_swe_obs_minus_mdl_mm < -50:
        markercolor='#f67100' # 255 169 0
    elif delta_swe_obs_minus_mdl_mm < -10:
        markercolor='#ffd817' # 255 216 23
    elif delta_swe_obs_minus_mdl_mm < -5:
        markercolor='#ffff7e' # 255 255 126
    elif delta_swe_obs_minus_mdl_mm < -1:
        markercolor='#ffffbe' # 255 255 190
    elif delta_swe_obs_minus_mdl_mm < 1:
        markercolor='#e6e6ff' # 230 230 255
    elif delta_swe_obs_minus_mdl_mm < 5:
        markercolor='#beffff' # 190 255 255
    elif delta_swe_obs_minus_mdl_mm < 10:
        markercolor='#7effff' # 126 255 255
    elif delta_swe_obs_minus_mdl_mm < 50:
        markercolor='#17d8ff' # 23 216 255
    elif delta_swe_obs_minus_mdl_mm < 100:
        markercolor='#00a9ff' # 0 169 255
    elif delta_swe_obs_minus_mdl_mm < 250:
        markercolor='#0071f6' # 0 113 246
    elif delta_swe_obs_minus_mdl_mm < 500:
        markercolor='#4b00f6' # 75 0 246
    else:
        markercolor='#640098' # 100 0 152
    return marker, markersize, markercolor


def idw_cross_validate(pt_df,
                       z_val_column_name,
                       max_distance_km,
                       min_num_points,
                       max_num_points,
                       horizontal_power,
                       elevation_power,
                       analysis_method='classic',
                       distance_method='geopy_great_circle',
                       aeqd_mosaic=None,
                       disable_progress=False):
    """
    Perform cross validation to estimate interpolation errors.
    The use of radius=6367.444657 is to better mimic the original idw.pro.
    """
    # Set up "neighborhood" bounding boxes in latitude and longitude by
    # estimating "km per degree" in the latitudinal and longitudinal
    # directions. These should be chosen so that the half-width of the
    # bounding box is never less than the neighborhood radius in either
    # direction. Consequently, the minimum viable "km per degree" estimate for
    # the region should be used.

    # The minimum km per degree latitude occurs at the equator, making that a
    # safe choice. We might also use the minimum (absolute) latitude in the
    # data, but the difference will be minimal, since km per degree latitude
    # on the WGS84 ellipsoid changes little with latitude:
    #
    # Latitude  km per degree latitude
    # --------  ----------------------
    #     0     110.574
    #    15     110.649
    #    30     110.852
    #    45     110.132
    #    60     111.413
    #    75     111.618
    #    90     111.694

    logger = logging.getLogger()

    if aeqd_mosaic is not None:
        if distance_method != 'azimuthal_equidistant_mosaic':
            msg = 'Unnecessary setting of keyword ' + \
                  '"aeqd_mosaic" with ' + \
                  f'distance_method set to "{distance_method}"'
            logger.warning(msg)
        else:
            if aeqd_mosaic is None:
                msg = 'Missing required instance of ' +  \
                      'class AzEqdMosaic'
                return None

    # Pull point data from the DataFrame. Only include REAL observation
    # points. Those with NULL STATION_ID columns are artificial zeroes used to
    # force the result to zero at the boundary of the region of interest.
    pt_is_real = pt_df['STATION_ID'].notna().to_numpy()
    pt_lat = pt_df['Y'].to_numpy()
    pt_lon = pt_df['X'].to_numpy()
    pt_dem_elev = pt_df['DEM_ELEVAT'].to_numpy()
    pt_z_val = pt_df[z_val_column_name].to_numpy()
    num_points = len(pt_df)
    pt_ind = np.arange(num_points)

    # Get reference coordinates for reference "km_per_deg" values.
    min_point_abs_lat = np.min(np.abs(pt_lat))
    max_point_abs_lat = np.max(np.abs(pt_lat))
    min_point_lon = np.min(pt_lon)
    max_point_lon = np.max(pt_lon)
    mid_point_lon = 0.5 * (min_point_lon + max_point_lon)

    # km_per_deg_lat_ref = distance.great_circle((0.0005, 0.0),
    #                                            (-0.0005, 0.0),
    #                                            radius=6367.444657).km * 1000.0
    km_per_deg_lat_ref = \
        geodst.one_to_one_km(min_point_abs_lat - 0.001, mid_point_lon,
                             min_point_abs_lat, mid_point_lon,
                             distance_method=distance_method,
                             aeqd_mosaic=aeqd_mosaic) * 1000.0
    message = f'Reference km per degree latitude: {km_per_deg_lat_ref:.4f}'
    logger.debug(message)

    # km_per_deg_lon_ref = distance.great_circle((max_point_abs_lat, 0.0),
    #                                            (max_point_abs_lat, 1.0),
    #                                            radius=6367.444657).km
    km_per_deg_lon_ref = \
        geodst.one_to_one_km(max_point_abs_lat, mid_point_lon - 0.5,
                             max_point_abs_lat, mid_point_lon + 0.5,
                             distance_method=distance_method,
                             aeqd_mosaic=aeqd_mosaic)
    message = 'Reference km per degree longitude ' + \
              f'(applied at {max_point_abs_lat:.4f} degrees): ' + \
              f'{km_per_deg_lon_ref:.4f}'
    logger.debug(message)

    bbox_half_width_lat_deg = max_distance_km / km_per_deg_lat_ref
    bbox_half_width_lon_deg = max_distance_km / km_per_deg_lon_ref

    # # Calculate a bounding box neighborhood for each point location.
    # nbrhood_bbox_min_lat = pt_lat - bbox_half_width_lat_deg
    # nbrhood_bbox_max_lat = pt_lat + bbox_half_width_lat_deg
    # nbrhood_bbox_min_lon = pt_lon - bbox_half_width_lon_deg
    # nbrhood_bbox_max_lon = pt_lon + bbox_half_width_lon_deg

    # Initialize an instance of the IDW class.
    this_idw = IDW(bbox_half_width_lat_deg,
                   bbox_half_width_lon_deg,
                   max_distance_km=max_distance_km,
                   min_num_points=min_num_points,
                   max_num_points=max_num_points,
                   horizontal_power=horizontal_power,
                   elevation_power=elevation_power,
                   analysis_method=analysis_method,
                   distance_method=distance_method,
                   aeqd_mosaic=aeqd_mosaic)

    # Perform cross validation.
    z_val_interp = np.full(len(pt_df), np.nan, dtype=np.float64)

    # for ind, point in tqdm(enumerate(pt_df.itertuples()), total=len(pt_df)):
    # message = 'Performing cross validation for "{}" interpolation'.\
    #     format(z_val_column_name)
    # logger.info(message)
    for ind in tqdm(range(num_points),
                    disable=disable_progress,
                    desc='   Validate'):
        z_val_interp[ind] = \
            idw_for_cross_val_loc(this_idw,
                                  ind,
                                  pt_is_real,
                                  pt_lat,
                                  pt_lon,
                                  pt_dem_elev,
                                  pt_ind,
                                  pt_z_val)

    message = 'Wall time for selecting neighborhoods by distance: ' + \
              f'{this_idw.distance_calc_total_wall_time}'
    logger.debug(message)
    message = f'Number of distance calculations: {this_idw.num_distance_calc}'
    logger.debug(message)
    return z_val_interp


def plot_idw_cross_val_results(observed_vals,
                               analysis_vals,
                               site_ids,
                               title,
                               x_axis_label,
                               y_axis_label,
                               min_num_points,
                               max_num_points,
                               horizontal_power,
                               elevation_power,
                               max_distance_km,
                               cv_r_squared,
                               cv_rmse_mm,
                               cv_mae_mm,
                               line_fit=False):
    """
    Plot the results of a leave-one-out cross validation of the IDW process.
    """
    # logger = logging.getLogger()

    # Perform simple linear regression of the cross validation results
    # to provide a way of assessing the results in addition to
    # cv_r_squared.
    # ols_x = observed_vals
    # ols_y = analysis_vals
    # ols_x = sm.add_constant(ols_x)
    # result = sm.OLS(ols_y, ols_x).fit()
    if line_fit:
        result, x2, y2, sd = binned_xy_wls(analysis_vals, observed_vals,
                                           num_bins=10,
                                           bin_padding=1.0,
                                           match_ranges=True,
                                           max_acceptable_bin_size=10.0)

    # Perform a fit using scipy.optimize. This is only here until I feel that
    # I know how to use statsmodels properly, though there is no major reason
    # for me to favor one over the other.
    # ocf_params = optimize.curve_fit(reg_line, x2, y2, [0.0, 1.0], sigma=sd)
    # print(f'curve_fit returns parameters of {ocf_params}')
    # print(result.params)

    # Generate a scatter plot of cross-validation results.

    # Set margins for the scatter plot.
    ax_w_in = 6.0
    ax_h_in = 6.0
    fig_margin_l_in = 1.25
    fig_margin_b_in = 0.75
    fig_margin_r_in = 0.5
    fig_margin_t_in = 0.75
    fig_w_in = fig_margin_l_in + ax_w_in + fig_margin_r_in
    fig_h_in = fig_margin_b_in + ax_h_in + fig_margin_t_in
    fig_bbox = [fig_margin_l_in / fig_w_in,
                fig_margin_b_in / fig_h_in,
                (fig_w_in - fig_margin_r_in) / fig_w_in,
                (fig_h_in - fig_margin_t_in) / fig_h_in]

    # Create the plot.
    cv_fig, cv_ax = plt.subplots(figsize=(fig_w_in, fig_h_in),
                                 num=increment_figure(),
                                 dpi=108,
                                 gridspec_kw={'left':fig_bbox[0],
                                              'bottom':fig_bbox[1],
                                              'right':fig_bbox[2],
                                              'top':fig_bbox[3]},
                                 )

    plt.grid(True, color='lightgrey', alpha=0.5)
    cv_ax.plot(observed_vals, analysis_vals, 'o',
               color='DodgerBlue', markersize=3.0,
               zorder=1)

    if line_fit:
        cv_ax.errorbar(x2, y2, yerr=sd,
                       ecolor='LightGray', capsize=4, linestyle='none',
                       marker='s', markersize=6,
                       markeredgecolor='DarkGray',
                       markerfacecolor='LightGray',
                       alpha=0.5, zorder=0)
    cv_ax.set_aspect('equal')
    xymin = np.min([cv_ax.get_xlim(), cv_ax.get_ylim()])
    xymax = np.max([cv_ax.get_xlim(), cv_ax.get_ylim()])
    cv_ax.set_xlim(xymin, xymax)
    cv_ax. set_ylim(xymin, xymax)
    cv_ax.set_xlabel(x_axis_label)
    cv_ax.set_ylabel(y_axis_label)
    cv_ax.set_title(title)

    # Create the legend.
    num_worst = 10
    handles = [patches.Rectangle((0,0), 1, 1, alpha=0)] * 7
    labelcolors = ['black'] * 7
    labelcolors[5] = 'red'
    labels = [f'{len(analysis_vals):,} points',
              f'"{min_num_points} {max_num_points} ' + \
              f'{horizontal_power} {elevation_power} ' + \
              f'{max_distance_km}"',
              f'R\u00b2 = {cv_r_squared:.3f}',
              f'RMSE = {cv_rmse_mm:.3f} mm',
              f'MAE = {cv_mae_mm:.3f} mm',
              f'IDs of {num_worst} worst results']
    if line_fit:
        labels.append(f'Fit: Y = {result.params[1]:.3f}X + ' + \
                      f'{result.params[0]:.3f}')
        cv_ax.plot(np.array(cv_ax.get_xlim()),
                   np.array(cv_ax.get_xlim()) * result.params[1] \
                   + result.params[0],
                   color='black', linestyle='--', alpha=0.2)
    cv_ax.plot(np.array(cv_ax.get_xlim()),
               np.array(cv_ax.get_xlim()),
               color='black', alpha=0.2)
    cv_ax.legend(handles, labels,
                 loc='best',
                 labelcolor=labelcolors,
                 handlelength=0,
                 handletextpad=0,
                 fontsize='small')

    # Get moments and identify the points where we have the worst
    # results. The distance from the 1:1 line to any (x,y) point is
    # sqrt(2) / 2 * sqrt((x-y)^2), so (x-y)^2 is a straightforward metric
    # here.
    vario = (observed_vals - analysis_vals)**2
    vario_order = np.argsort(vario)
    texts=[]
    for i in vario_order[-num_worst:]:
        texts.append(cv_ax.text(observed_vals[i],
                                analysis_vals[i],
                                site_ids[i],
                                color='r',
                                zorder=2))
    # TODO: get adjustText working on AWS.
    if ADJUST_TEXT_SUPPORTED:
        adjust_text(texts, ax=cv_ax)

    return cv_fig, cv_ax


def binned_xy_ols(y, x,
                  num_bins=20,
                  bin_padding=0.0,
                  match_ranges=False,
                  max_acceptable_bin_size=None):
    """
    Perform linear regression of y vs. x, but bin the x and y values to
    limit the effect of clustered data.
    """
    if match_ranges:
        combo = np.concatenate((x,y))
        range_min = combo.min() - bin_padding
        range_max = combo.max() + bin_padding
        bin_size = (range_max - range_min) / num_bins
        if max_acceptable_bin_size is not None:
            while bin_size > max_acceptable_bin_size:
                num_bins += 1
                bin_size = (range_max - range_min) / num_bins
        if bin_padding == 0.0:
            # Add one bin_size bin.
            num_bins += 1
            range_min = range_min - 0.5 * bin_size
            range_max = range_max + 0.5 * bin_size
        num_bins = [num_bins, num_bins]
        bin_range = [[range_min, range_max], [range_min, range_max]]
    else:
        x_range_min = x.min() - bin_padding
        x_range_max = x.max() + bin_padding
        x_num_bins = num_bins
        x_bin_size = (x_range_max - x_range_min) / x_num_bins
        y_range_min = y.min() - bin_padding
        y_range_max = y.max() + bin_padding
        y_num_bins = num_bins
        y_bin_size = (y_range_max - y_range_min) / y_num_bins
        if max_acceptable_bin_size is not None:
            while x_bin_size > max_acceptable_bin_size:
                x_num_bins += 1
                x_bin_size = (x_range_max - x_range_min) / x_num_bins
            while y_bin_size > max_acceptable_bin_size:
                y_num_bins += 1
                y_bin_size = (y_range_max - y_range_min) / y_num_bins
        if bin_padding == 0.0:
            # Add one x_bin_size and one y_bin_size bin.
            x_num_bins += 1
            x_range_min = x_range_min - 0.5 * x_bin_size
            x_range_max = x_range_max + 0.5 * x_bin_size
            y_num_bins += 1
            y_range_min = y_range_min - 0.5 * y_bin_size
            y_range_max = y_range_max + 0.5 * y_bin_size
        num_bins = [x_num_bins, y_num_bins]
        bin_range = [[x_range_min, x_range_max], [y_range_min, y_range_max]]
    x2, x2_edge, y2_edge, x_bin_number = \
        stats.binned_statistic_2d(x, y, x, statistic='mean',
                                  bins=num_bins,
                                  range=bin_range)
    y2, x2_edge, y2_edge, y_bin_number = \
        stats.binned_statistic_2d(x, y, y, statistic='mean',
                                  bins=num_bins,
                                  range=bin_range)
    condition = ~np.isnan(x2) & ~np.isnan(y2)
    x2 = x2[condition]
    y2 = y2[condition]
    # x2 = sm.add_constant(x2)
    result = sm.OLS(y2, x2).fit()

    return result


def binned_xy_wls(y, x,
                  num_bins=20,
                  bin_padding=0.0,
                  match_ranges=False,
                  max_acceptable_bin_size=None):
    """
    Perform linear regression of y vs. x, but bin the x and y values to
    limit the effect of clustered data, and use weighted least squares,
    with weights equal to 1 / variance in the y variable for each bin.
    """
    if match_ranges:
        combo = np.concatenate((x,y))
        range_min = combo.min() - bin_padding
        range_max = combo.max() + bin_padding
        bin_size = (range_max - range_min) / num_bins
        if max_acceptable_bin_size is not None:
            while bin_size > max_acceptable_bin_size:
                num_bins += 1
                bin_size = (range_max - range_min) / num_bins
        if bin_padding == 0.0:
            # Add one bin_size bin.
            num_bins += 1
            range_min = range_min - 0.5 * bin_size
            range_max = range_max + 0.5 * bin_size
        num_bins = [num_bins, num_bins]
        bin_range = [[range_min, range_max], [range_min, range_max]]
    else:
        x_range_min = x.min() - bin_padding
        x_range_max = x.max() + bin_padding
        x_num_bins = num_bins
        x_bin_size = (x_range_max - x_range_min) / x_num_bins
        y_range_min = y.min() - bin_padding
        y_range_max = y.max() + bin_padding
        y_num_bins = num_bins
        y_bin_size = (y_range_max - y_range_min) / y_num_bins
        if max_acceptable_bin_size is not None:
            while x_bin_size > max_acceptable_bin_size:
                x_num_bins += 1
                x_bin_size = (x_range_max - x_range_min) / x_num_bins
            while y_bin_size > max_acceptable_bin_size:
                y_num_bins += 1
                y_bin_size = (y_range_max - y_range_min) / y_num_bins
        if bin_padding == 0.0:
            # Add one x_bin_size and one y_bin_size bin.
            x_num_bins += 1
            x_range_min = x_range_min - 0.5 * x_bin_size
            x_range_max = x_range_max + 0.5 * x_bin_size
            y_num_bins += 1
            y_range_min = y_range_min - 0.5 * y_bin_size
            y_range_max = y_range_max + 0.5 * y_bin_size
        num_bins = [x_num_bins, y_num_bins]
        bin_range = [[x_range_min, x_range_max], [y_range_min, y_range_max]]
    x2, x2_edge, y2_edge, x_bin_number = \
        stats.binned_statistic_2d(x, y, x, statistic='mean',
                                  bins=num_bins,
                                  range=bin_range)
    y2, x2_edge, y2_edge, y_bin_number = \
        stats.binned_statistic_2d(x, y, y, statistic='mean',
                                  bins=num_bins,
                                  range=bin_range)
    sd, x2_edge, y2_edge, s_bin_number = \
        stats.binned_statistic_2d(x, y, y, statistic='std',
                                  bins=num_bins,
                                  range=bin_range)

    # Do not allow the value used to calculate the weight to be less than
    # 10% of the mean y value.
    sd_floor = 0.1 * y2
    sd = np.maximum(sd, sd_floor)

    # There could still be zeroes in sd. Reset them to the minimum nonzero
    # value.
    sd = np.maximum(sd, np.min(sd[sd > 0]))

    wt = 1.0 / sd**2.0

    # condition = ~np.isnan(x2) & ~np.isnan(y2) & ~np.isnan(wt)
    condition = np.isfinite(x2) & np.isfinite(y2) & np.isfinite(wt)
    x2 = x2[condition]
    y2 = y2[condition]
    sd = sd[condition]
    wt = wt[condition]

    X2 = sm.add_constant(x2)
    result = sm.WLS(y2, X2, wt).fit()

    # result = sm.GLM(y2, x2, w = wt).fit_constrained()
    # Potentially use scipy.optimize.minimize instead of statsmodels, though
    # the only constraint I might want is no intercept, which is easily
    # accomplished by not applying add_constant to the statsmodels API method.

    return result, x2, y2, sd


def auto_gen_assim_params(args,
                          pt_df,
                          event_threshold_m=1.0e-4,
                          analysis_method='classic',
                          distance_method='geopy_great_circle',
                          aeqd_mosaic=None,
                          disable_progress=False):
    """
    Automatically generate SNODAS IDW assimilation parameters with a brute
    force search of parameter space.
    """

    # Parameters will be stored as if they were command line arguments.
    # global args

    logger = logging.getLogger()

    # Remove correct negatives from pt_df.
    event_threshold_m = 1.0e-4
    pt_no_cn = pt_df[(pt_df['OB_SWE'] > event_threshold_m) | \
                     (pt_df['MD_SWE'] > event_threshold_m)]

    # Set up arrays of parameters. Manually scanning parameter space at low
    # resolution is what we have, for now.
    min_num_points = np.array([3,5,10])
    max_num_points = np.append(np.linspace(50, 100, 3, dtype='int'), -1)
    horiz_power = np.linspace(1.0, 2.0, 3)
    elev_power = np.linspace(0.0, 2.0, 5)
    max_distance_km = np.linspace(50.0, 250.0, 9)
    num_tests \
        = len(min_num_points) \
        * len(max_num_points) \
        * len(horiz_power) \
        * len(elev_power) \
        * len(max_distance_km)
    cv_proportion = np.full(num_tests, np.nan, dtype=np.float64)
    # cv_semivariance = np.full(num_tests, np.nan, dtype=np.float64)
    # cv_fitted_slope = np.full(num_tests, np.nan, dtype=np.float64)
    cv_r_squared = np.full(num_tests, np.nan, dtype=np.float64)
    cv_rmse_mm = np.full(num_tests, np.nan, dtype=np.float64)
    cv_mae_mm = np.full(num_tests, np.nan, dtype=np.float64)
    delta_swe_om = pt_no_cn['D_SWE_OM'].to_numpy()
    # delta_swe_range = delta_swe_om.max() - delta_swe_om.min()
    # print(f'\N{GREEK CAPITAL LETTER DELTA}SWE range: {delta_swe_range}')
    # print(np.log10(delta_swe_range))
    # max_cv_r_squared = -1.0e10
    for param_ind in tqdm(range(num_tests),
                          disable=disable_progress):
        i, j, k, l, m = np.unravel_index(param_ind,
                                         (len(min_num_points),
                                          len(max_num_points),
                                          len(horiz_power),
                                          len(elev_power),
                                          len(max_distance_km)))
        delta_swe_cv = idw_cross_validate(pt_no_cn,
                                          'D_SWE_OM',
                                          max_distance_km[m],
                                          min_num_points[i],
                                          max_num_points[j],
                                          horiz_power[k],
                                          elev_power[l],
                                          analysis_method=analysis_method,
                                          distance_method=distance_method,
                                          aeqd_mosaic=aeqd_mosaic,
                                          disable_progress=True)

        # Evaluate errors in delta_swe_cv.
        ind = np.where(np.isfinite(delta_swe_om) & \
                       np.isfinite(delta_swe_cv))[0]
        # For now only consider parameters that give a result for all sites.
        cv_proportion[param_ind] = float(len(ind)) / float(len(pt_no_cn))
        if cv_proportion[param_ind] == 0.0:
            continue

        # # Get the semivariance of the cross-validation points. The distance
        # # from the 1:1 line of delta_swe_cv vs. delta_swe_om to any (x,y)
        # # point is sqrt(2) * sqrt((x-y)^2), so the mean value of (x-y)^2, also
        # # known as the semivariance, is a straightforward metric for how well
        # # the current parameters reproduce the data.
        # squared_error = (delta_swe_om[ind] - delta_swe_cv[ind])**2
        # cv_semivariance[param_ind] = squared_error.mean()

        # # Perform a simple linear regression, so the slope can be used as the
        # # key metric.
        # # ols_x = delta_swe_om[ind]
        # # ols_y = delta_swe_cv[ind]
        # # ols_x = sm.add_constant(ols_x)
        # # result = sm.OLS(ols_y, ols_x).fit()
        # # cv_fitted_slope[param_ind] = result.params[1]
        # ols_x = delta_swe_om[ind]
        # ols_y = delta_swe_cv[ind]
        # # ols_x = sm.add_constant(ols_x)
        # result = sm.OLS(ols_y, ols_x).fit()
        # cv_fitted_slope[param_ind] = result.params[0]

        # Perform a linear regression using binned statistics, to effectively
        # decluster the cross-validation data, typically done to prevent
        # small values from dominating the regression.
        # result = binned_xy_ols(delta_swe_cv[ind], delta_swe_om[ind],
        #                        num_bins=20,
        #                        bin_padding=1.0e-3,
        #                        match_ranges=True,
        #                        max_acceptable_bin_size=5.0e-3)
        # cv_fitted_slope[param_ind] = result.params[0]

        # Generate stats based on residuals.
        cv_residual = delta_swe_om[ind] - delta_swe_cv[ind]
        ss_res = np.sum(cv_residual**2)
        cv_rmse_mm[param_ind] = np.sqrt(np.mean(ss_res)) * 1000
        cv_mae_mm[param_ind] = np.mean(np.abs(cv_residual)) * 1000
        ss_tot = np.sum((delta_swe_om[ind] - np.mean(delta_swe_om[ind]))**2)
        cv_r_squared[param_ind] = 1.0 - ss_res / ss_tot
        # Perform a linear least squares fit.
        # X = np.column_stack((delta_swe_om))
        # X = sm.add_constant(X)
        #mod = sm.OLS(

    # Mask results wherever the stats are not well-behaved.
    # condition = ~np.isfinite(cv_fitted_slope) \
    #     | (cv_fitted_slope <= 0) \
    #     | np.isnan(cv_r_squared) \
    #     | ~np.isfinite(cv_semivariance)
    condition = ~np.isfinite(cv_r_squared)
    # cv_semivariance = np.ma.masked_where(condition, cv_semivariance)
    # cv_fitted_slope = np.ma.masked_where(condition, cv_fitted_slope)
    cv_rmse_mm = np.ma.masked_where(condition, cv_rmse_mm)
    cv_mae_mm = np.ma.masked_where(condition, cv_mae_mm)
    cv_r_squared = np.ma.masked_where(condition, cv_r_squared)
    cv_proportion = np.ma.masked_where(condition, cv_proportion)

    if cv_rmse_mm.count() == 0:
        logger.error('Failed to cross-validate IDW parameters.')
        return None, None, None
    elif cv_rmse_mm.count() < num_tests:
        message = 'Successfully cross-validated ' + \
                  f'{cv_rmse_mm.count()} of {num_tests} ' + \
                  'parameter combinations.'
        logger.info(message)

    # Consider cv_proportion. If at least 25% of the cross-validation results
    # provide a value for each location (have cv_proportion == 1.0), then
    # limit the results to those. Otherwise, limit the results to those cases
    # for which cv_proportion is at or above its median value.
    # half_of_cv_results = num_tests // 2
    # print(np.where(np.ma.getmaskarray(cv_r_squared) == False)[0])
    # print(len(np.where(np.ma.getmaskarray(cv_r_squared) == False)[0]))
    # print(np.where(cv_proportion == 1.0)[0])
    if (cv_proportion == 1.0).sum() >= num_tests // 4:
        cv_proportion_threshold = 1.0
    else:
        cv_proportion_threshold = np.ma.median(cv_proportion)
    if cv_proportion_threshold != 1.0:
        message = 'Only considering parameter combinations for which ' + \
                  'valid IDW results are generated for at least ' + \
                  f'{cv_proportion_threshold * 100:.1f}% of locations.'
        logger.info(message)
    else:
        message = 'Only considering parameter combinations for which ' + \
                  'valid IDW results are generated for all locations.'
        logger.info(message)

    # Mask arrays wherever cv_proportion < cv_proportion_threshold
    condition = cv_proportion < cv_proportion_threshold
    # cv_semivariance = np.ma.masked_where(condition, cv_semivariance)
    # cv_fitted_slope = np.ma.masked_where(condition, cv_fitted_slope)
    cv_rmse_mm = np.ma.masked_where(condition, cv_rmse_mm)
    cv_mae_mm = np.ma.masked_where(condition, cv_mae_mm)
    cv_r_squared = np.ma.masked_where(condition, cv_r_squared)

    # Decide which parameters are "best".
    # best_fit_param_ind = np.argmax(cv_r_squared)
    # best_fit_param_ind = np.argmin(np.abs(np.log(cv_fitted_slope)))
    # best_fit_param_ind = np.argmin(cv_semivariance)
    # Use the lowest cv_rmse_mm value to decide which parameters are
    # "best".
    best_fit_param_ind = np.argmin(cv_rmse_mm)
    i, j, k, l, m = np.unravel_index(best_fit_param_ind,
                                     (len(min_num_points),
                                      len(max_num_points),
                                      len(horiz_power),
                                      len(elev_power),
                                      len(max_distance_km)))

    # Update args with the selected parameters.
    args.min_num_points = min_num_points[i]
    args.max_num_points = max_num_points[j]
    args.horizontal_power = horiz_power[k]
    args.elevation_power = elev_power[l]
    args.max_distance_km = max_distance_km[m]

    return \
        cv_r_squared[best_fit_param_ind], \
        cv_rmse_mm[best_fit_param_ind], \
        cv_mae_mm[best_fit_param_ind], \
        args


def idw_for_grid_loc_init(point_is_real,
                          point_lat,
                          point_lon,
                          point_dem_elev,
                          point_z_val,
                          this_idw,
                          this_idw_grid):
    global P_POINT_IS_REAL
    global P_POINT_LAT
    global P_POINT_LON
    global P_POINT_DEM_ELEV
    global P_POINT_Z_VAL
    global P_THIS_IDW
    global P_THIS_IDW_GRID

    P_POINT_IS_REAL = point_is_real
    P_POINT_LAT = point_lat
    P_POINT_LON = point_lon
    P_POINT_DEM_ELEV = point_dem_elev
    P_POINT_Z_VAL = point_z_val
    P_THIS_IDW = this_idw
    P_THIS_IDW_GRID = this_idw_grid


def idw_for_grid_loc(ind, cell_elev):
    """
    Perform one iteration of IDW for gridded (lon/lat) output.
    """

    # Globals created by idw_lon_lat_grid.
    # global POINT_IS_REAL
    # global POINT_LAT
    # global POINT_LON
    # global POINT_DEM_ELEV
    # global POINT_Z_VAL
    # global this_idw
    # global this_idw_grid

    global num_filtered

    row, col = np.unravel_index(ind, P_THIS_IDW_GRID.output_grid_shape)
    if row != P_THIS_IDW_GRID.prev_row:
        P_THIS_IDW_GRID.set_bbox_lat_bounds(row)
    if col != P_THIS_IDW_GRID.prev_col:
        P_THIS_IDW_GRID.set_bbox_lon_bounds(col)

    # Find all points in a bounding box approximately (slightly larger than)
    # the size of the neighborhood defined by the grid location and
    # max_distance_km, approximated by
    # P_THIS_IDW_GRID.nbrhood_bbox_[min,max]_[lat,lon].
    in_nbrhood_bbox_tf = \
        (P_POINT_LAT >= P_THIS_IDW_GRID.nbrhood_bbox_min_lat) & \
        (P_POINT_LAT <= P_THIS_IDW_GRID.nbrhood_bbox_max_lat) * \
        (P_POINT_LON >= P_THIS_IDW_GRID.nbrhood_bbox_min_lon) & \
        (P_POINT_LON <= P_THIS_IDW_GRID.nbrhood_bbox_max_lon)
    num_bbox_neighbors = in_nbrhood_bbox_tf.sum()
    # box_time_elapsed += time.perf_counter() - t1
    if num_bbox_neighbors < P_THIS_IDW.min_num_points:
        P_THIS_IDW_GRID.num_too_few_obs += 1
        return None

    # Extract bounding box data into new arrays.
    nbrhood_bbox_pt_is_real = P_POINT_IS_REAL[in_nbrhood_bbox_tf]
    nbrhood_bbox_pt_lat = P_POINT_LAT[in_nbrhood_bbox_tf]
    nbrhood_bbox_pt_lon = P_POINT_LON[in_nbrhood_bbox_tf]
    nbrhood_bbox_pt_dem_elev = P_POINT_DEM_ELEV[in_nbrhood_bbox_tf]
    nbrhood_bbox_pt_z_val = P_POINT_Z_VAL[in_nbrhood_bbox_tf]

    if P_THIS_IDW.analysis_method == 'classic':
        IDWMethod = P_THIS_IDW.classic
    elif P_THIS_IDW.analysis_method == 'with_elev_regression':
        IDWMethod = P_THIS_IDW.with_elev_regression
    else:
        return None

    z_val_interp = \
        IDWMethod(P_THIS_IDW_GRID.row_lat,
                  P_THIS_IDW_GRID.col_lon,
                  cell_elev,
                  nbrhood_bbox_pt_is_real,
                  nbrhood_bbox_pt_lat,
                  nbrhood_bbox_pt_lon,
                  nbrhood_bbox_pt_dem_elev,
                  nbrhood_bbox_pt_z_val)

    # this_idw_grid
    # this_idw
    return z_val_interp


def idw_lon_lat_grid(pt_df,
                     z_val_column_name,
                     max_distance_km,
                     min_num_points,
                     max_num_points,
                     horizontal_power,
                     elevation_power,
                     dem_elev_grid,
                     proc_region_grid,
                     lon_axis,
                     lat_axis,
                     analysis_method='classic',
                     distance_method='geopy_great_circle',
                     aeqd_mosaic=None,
                     disable_progress=False):
    """
    Inverse distance weighted (IDW) interpolation on a longitude/latitude
    grid.
    """

    # These globals are created here and accessed as read-only by
    # idw_for_grid_loc, with two exceptions: this_idw has its method "classic"
    # called in idw_for_grid_loc, and this_idw_grid is updated by its own
    # methods and has at least one of its attributes adjusted in
    # idw_for_grid_loc. Therefore those two are left uncapitalized.
    # global POINT_IS_REAL
    # global POINT_LAT
    # global POINT_LON
    # global POINT_DEM_ELEV
    # global POINT_Z_VAL
    # global this_idw
    # global this_idw_grid

    logger = logging.getLogger()

    chunksize = 512

    if proc_region_grid.shape != dem_elev_grid.shape:
        logger.error('Input grids are different shapes.')
        return None

    num_rows_out, num_cols_out = dem_elev_grid.shape

    if lon_axis.shape[0] != num_cols_out:
        # logger.error('Longitude axis is inconsistent with input grids.')
        # return None
        raise GridInconsistency
    if lat_axis.shape[0] != num_rows_out:
        # logger.error('Latitude axis is inconsistent with input grids.')
        # return None
        raise GridInconsistency

    # Set up "neighborhood" bounding boxes in latitude and longitude by
    # estimating "km per degree" in the latitudinal and longitudinal
    # directions. These should be chosen so that the half-width of the
    # bounding box is never less than the neighborhood radius in either
    # direction. Consequently, the minimum viable "km per degree" estimate for
    # the region should be used.

    # The minimum km per degree latitude occurs at the equator, making that a
    # safe choice. We might also use the minimum (absolute) latitude in the
    # data, but the difference will be minimal, since km per degree latitude
    # on the WGS84 ellipsoid changes little with latitude:
    #
    # Latitude  km per degree latitude
    # --------  ----------------------
    #     0     110.574
    #    15     110.649
    #    30     110.852
    #    45     110.132
    #    60     111.413
    #    75     111.618
    #    90     111.694

    # Pull point data from the DataFrame.
    POINT_IS_REAL = pt_df['STATION_ID'].notna().to_numpy()
    POINT_LAT = pt_df['Y'].to_numpy()
    POINT_LON = pt_df['X'].to_numpy()
    POINT_DEM_ELEV = pt_df['DEM_ELEVAT'].to_numpy()
    POINT_Z_VAL = pt_df[z_val_column_name].to_numpy()

    # max_idw_abs_lat = max([np.max(np.abs(lat_axis)), pt_df['Y'].abs().max()])
    # max_idw_abs_lat = max([np.max(np.abs(lat_axis)),
    #                        np.max(np.abs(POINT_LAT))])
    # Neighborhoods are defined relative to grid locations, so only those are
    # used to determine reference "km_per_deg" values.
    min_idw_abs_lat = np.min(np.abs(lat_axis))
    max_idw_abs_lat = np.max(np.abs(lat_axis))
    mid_point_lon = 0.5 * (lon_axis[0] + lon_axis[-1])

    # km_per_deg_lat_ref = distance.great_circle((0.0005, 0.0),
    #                                            (-0.0005, 0.0),
    #                                            radius=6367.444657).km *
    #                                            1000.0
    km_per_deg_lat_ref = \
        geodst.one_to_one_km(min_idw_abs_lat - 0.001, mid_point_lon,
                             min_idw_abs_lat, mid_point_lon,
                             distance_method=distance_method,
                             aeqd_mosaic=aeqd_mosaic) * 1000.0
    message = 'Reference km per degree latitude: {:.4f}'. \
        format(km_per_deg_lat_ref)
    logger.debug(message)

    # km_per_deg_lon_ref = distance.great_circle((max_idw_abs_lat, 0.0),
    #                                            (max_idw_abs_lat, 1.0),
    #                                            radius=6367.444657).km
    # distance.distance((max_idw_abs_lat, 0.0),
    #                   (max_idw_abs_lat, 1.0)).km
    km_per_deg_lon_ref = \
        geodst.one_to_one_km(max_idw_abs_lat, mid_point_lon - 0.5,
                             max_idw_abs_lat, mid_point_lon + 0.5,
                             distance_method=distance_method,
                             aeqd_mosaic=aeqd_mosaic)
    message = 'Reference km per degree longitude ' + \
        '(applied at {:.4f} degrees): {:.4f}'. \
        format(max_idw_abs_lat, km_per_deg_lon_ref)
    logger.debug(message)

    bbox_half_width_lat_deg = max_distance_km / km_per_deg_lat_ref
    bbox_half_width_lon_deg = max_distance_km / km_per_deg_lon_ref

    # Identify grid locations where interpolation will be performed.
    message = f'Output grid shape: ({num_rows_out},{num_cols_out})'
    logger.debug(message)

    num_dem_valid = dem_elev_grid.count()
    if num_dem_valid != num_rows_out * num_cols_out:
        dem_num_missing = num_rows_out * num_cols_out - num_dem_valid
        message = 'DEM grid includes {} missing values.'. \
            format(dem_num_missing)
        logger.warning(message)

    # Create a processing mask that combines the processing region and DEM.
    process_mask = np.ma.getmaskarray(proc_region_grid) \
                   | np.ma.getmaskarray(dem_elev_grid)

    # Initialize an instance of the IDW class.
    this_idw = IDW(bbox_half_width_lat_deg,
                   bbox_half_width_lon_deg,
                   max_distance_km=max_distance_km,
                   min_num_points=min_num_points,
                   max_num_points=max_num_points,
                   horizontal_power=horizontal_power,
                   elevation_power=elevation_power,
                   analysis_method=analysis_method,
                   distance_method=distance_method,
                   aeqd_mosaic=aeqd_mosaic)

    this_idw_grid = IDWGrid(lat_axis,
                            lon_axis,
                            bbox_half_width_lat_deg,
                            bbox_half_width_lon_deg)

    # Keep track of the time taken for generating in_nbrhood_box_tf.
    box_time_elapsed = 0.0

    output_grid = np.full(dem_elev_grid.shape, np.nan, dtype=np.float64)

    # Stuff that can be in an obvious list of tuples for multiprocessing:
    # 1. ind
    # 2. replace process_mask with sampled values for all
    # 3. replace dem_elev_grid with sampled values for all
    # Those will give me three things in a tuple, and a list of those tuples
    # for the whole grid would be the iterated thing.
    # That leaves this_idw, this_idw_grid, and the 4 point data arrays.
    # Try making them global.

    # Create 1-D versions...
    process_mask = process_mask.flatten()
    cell_index = np.arange(num_rows_out * num_cols_out)
    dem_elev_grid = dem_elev_grid.flatten()
    output_grid = output_grid.flatten()

    # Extract non-masked cells.
    process_ind = np.where(process_mask == False)[0]
    cell_index_process_subset = cell_index[process_ind]
    dem_elev_grid_process_subset = dem_elev_grid[process_ind]
    #output_grid_process_subset = output_grid[process_ind]

    args = list(zip(cell_index_process_subset,
                    dem_elev_grid_process_subset))
    message = f'Interpolating "{z_val_column_name}" on the output grid...'
    logger.info(message)
    with Pool(8,
              initializer=idw_for_grid_loc_init,
              initargs=(POINT_IS_REAL,
                        POINT_LAT,
                        POINT_LON,
                        POINT_DEM_ELEV,
                        POINT_Z_VAL,
                        this_idw,
                        this_idw_grid)) as pool:
        output_grid_process_subset = \
            np.array(pool.starmap(idw_for_grid_loc,
                                  tqdm(args,
                                       total=len(process_ind),
                                       disable=disable_progress,
                                       desc='Interpolate'),
                                  chunksize=chunksize),
                     np.float64)

    output_grid[process_ind] = output_grid_process_subset
    output_grid = output_grid.reshape(num_rows_out, num_cols_out)

    message = 'Wall time for selecting neighborhoods by distance: ' + \
        f'{this_idw.distance_calc_total_wall_time}'
    logger.debug(message)
    message = 'Number of distance calculations: {this_idw.num_distance_calc}'
    logger.debug(message)

    return output_grid


def raster_filter_for_grid_loc_init(num_input_rows,
                                    num_input_cols,
                                    nbc,
                                    padded_input,
                                    padded_elev):
    """
    Initializer of global variables used by raster_filter_for_grid_loc
    when it is called by multiprocessing functions.
    """
    global P_NUM_INPUT_ROWS
    global P_NUM_INPUT_COLS
    global P_NBC
    global P_PADDED_INPUT
    global P_PADDED_ELEV

    P_NUM_INPUT_ROWS = num_input_rows
    P_NUM_INPUT_COLS = num_input_cols
    P_NBC = nbc
    P_PADDED_INPUT = padded_input
    P_PADDED_ELEV = padded_elev


def raster_filter_for_grid_loc(ind,
                               num_points_in_cell,
                               input_grid_val,
                               input_grid_elev):
    """
    Perform RASTER_FILTER filtering of one IDW output grid cell.
    This returns a tuple, so the results need to get handled properly, perhaps
    using "list(zip(*", if this function is called by pool.starmap, as is
    intended.
    """
    # Globals created by idw_pro_raster_filter:
    # global num_input_rows
    # global num_input_cols
    # global nbc
    # global padded_input
    # global padded_elev

    cell_filtered = False

    # if cell_is_masked:
    #     return cell_filtered, None

    input_row, input_col = \
        np.unravel_index(ind, (P_NUM_INPUT_ROWS, P_NUM_INPUT_COLS))
    padded_row = input_row + P_NBC
    padded_col = input_col + P_NBC
    subgrid = P_PADDED_INPUT[padded_row - P_NBC:padded_row + P_NBC + 1,
                             padded_col - P_NBC:padded_col + P_NBC + 1]
    if subgrid.count() < 3:
        return cell_filtered, input_grid_val

    subgrid_mean = np.mean(subgrid)
    cell_val_var = abs(input_grid_val - subgrid_mean)
    subgrid_std = np.std(subgrid)
    if cell_val_var > 1.5 * subgrid_std:
        # "High uncertainty" case; which I would state as "high doubt".
        # num_filtered += 1
        cell_filtered = True
        if num_points_in_cell > 0:
            # Mostly preserve the value/s from nearby station/s.
            input_grid_val = \
                0.125 * subgrid_mean + \
                0.875 * P_PADDED_INPUT[padded_row, padded_col]
            # if abs(input_grid_val) > 0.001:
            #     print('high doubt, but preserve original (12.5% weight) ' +
            #           'for {},{}: {}'.
            #           format(input_row, input_col, input_grid_val))
        else:
            input_grid_val = subgrid_mean
            # if abs(input_grid_val) > 0.001 and \
            #    input_row % 20 == 0 and \
            #    input_col % 20 == 0:
            #     print('high doubt, convert to the subgrid mean ' +
            #           'for {},{}: {}'.
            #           format(input_row, input_col, input_grid_val))

    elif cell_val_var > 1.0 * subgrid_std:

        # "Low uncertainty" case, which I would state as "moderate doubt".
        # num_filtered += 1
        cell_filtered = True
        # num_points_in_cell = ((pt_grid_col == input_col) &
        #                       (pt_grid_row == input_row)).sum()
        if num_points_in_cell > 0:
            # Mostly preserve the value/s from nearby station/s.
            input_grid_val = \
                0.125 * subgrid_mean + \
                0.875 * P_PADDED_INPUT[padded_row, padded_col]
            # if abs(input_grid_val) > 0.001:
            #     print('some doubt, but preserve original (12.5% weight) ' +
            #           'for {},{}: {}'.
            #           format(input_row, input_col, input_grid_val))
        else:
            # Check to see how typical the cell elevation is for its local
            # neighborhood.
            elev_subgrid = \
                P_PADDED_ELEV[padded_col - P_NBC:padded_col + P_NBC + 1,
                              padded_row - P_NBC:padded_row + P_NBC + 1]
            subgrid_elev_mean = np.mean(elev_subgrid)
            cell_elev_var = abs(input_grid_elev - subgrid_elev_mean)
            subgrid_elev_std = np.std(elev_subgrid)
            if cell_elev_var > 0.5 * subgrid_elev_std:
                # Unusual cell elevation; preserve a bit of the original
                # output value.
                input_grid_val = \
                    0.75 * subgrid_mean + \
                    0.25 * P_PADDED_INPUT[padded_row, padded_col]
                # if abs(input_grid_val) > 0.001 and \
                #    input_row % 10 == 0 and \
                #    input_col % 10 == 0:
                #     print('some doubt, but preserve original (25% weight) ' +
                #           'for {},{}: {}'.
                #           format(input_row, input_col, input_grid_val))
            else:
                input_grid_val = subgrid_mean
                # if abs(input_grid_val) > 0.001 and \
                #    input_row % 10 == 0 and \
                #    input_col % 10 == 0:
                #     print('some doubt, convert to the subgrid mean ' +
                #           'for {},{}: {}'.
                #           format(input_row, input_col, input_grid_val))
    else:
        pass
    # if input_row == sample_row and input_col == sample_col:
    #     print('({},{}) after: {}'.format(input_row, input_col,
    #                                      input_grid_val))
    return cell_filtered, input_grid_val


def idw_pro_raster_filter(input_grid,
                          elev_grid,
                          pt_grid_row,
                          pt_grid_col,
                          pt_df,
                          num_buffer_cells=3,
                          idw_pro_style=False,
                          multiprocess=False,
                          disable_progress=False):
    """
    Smooth/filter IDW results that differ significantly from other grid
    locations in their immediate neighborhood.
    Because the IDL procedure updates its own input grid as it goes, we need
    to replicate its logic in a way that is not necessarily all that pythonic
    or efficient. This is what happens with the idw_pro_style=True option. If
    the default idw_pro_style=False is used instead, then the process uses a
    separate output grid, which makes more sense.
    """

    logger = logging.getLogger()

    # These globals are created or passed here and accessed as read-only
    # by raster_filter_for_grid_loc.
    # global num_input_rows
    # global num_input_cols
    # global nbc
    # global padded_input
    # global padded_elev

    num_input_rows, num_input_cols = input_grid.shape
    num_dem_rows, num_dem_cols = elev_grid.shape
    if num_input_rows != num_dem_rows or \
       num_input_cols != num_dem_cols:
        raise GridInconsistency

    nbc = num_buffer_cells

    chunksize = 512

    # Pull point data from the DataFrame.
    # pt_lat = pt_df['Y'].to_numpy()
    # pt_lon = pt_df['X'].to_numpy()
    pt_dem_elev = pt_df['DEM_ELEVAT'].to_numpy()

    # Generate a grid that identifies how many point locations fall in each
    # cell of input_grid (usually zero)
    num_points_in_cell = np.zeros((num_input_rows, num_input_cols),
                                  dtype=np.int8)
    for rc in list(zip(pt_grid_row, pt_grid_col)):
        num_points_in_cell[rc] += 1

    # Create padded versions of the input_grid and elev_grid.
    padded_input = np.pad(input_grid, nbc, mode='edge')
    padded_elev = np.ma.masked_invalid(np.pad(elev_grid, nbc,
                                                 mode='empty'))
    # Erase padded values in the corners to match the behavior of
    # raster_filter.pro.
    padded_input[0:nbc,0:nbc] = np.nan
    padded_input[-nbc:,0:nbc] = np.nan
    padded_input[0:nbc,-nbc:] = np.nan
    padded_input[-nbc:,-nbc:] = np.nan
    padded_input = np.ma.masked_invalid(padded_input)

    # padded_num_rows = num_input_rows + 2 * nbc
    # padded_num_cols = num_input_cols + 2 * nbc

    # This algorithm should read from padded_input but only modify input_grid,
    # even though the original raster_filter.pro did not do it that way.
    output_grid = np.full(num_input_rows * num_input_cols,
                          np.nan, dtype=np.float64)
    input_mask = np.ma.getmaskarray(input_grid).flatten()
    logger.info('Running raster smoothing filter...')
    if multiprocess:
        if idw_pro_style:
            message = '"idw_pro_style=True" is ignored when ' + \
                '"multiprocessor=True" is set.'
            logger.warning(message)

        cell_index = np.arange(num_input_rows * num_input_cols)
        process_ind = np.where(input_mask == False)[0]
        # cell_index_process_subset = cell_index[process_ind]
        args = list(zip(cell_index[process_ind],
                        num_points_in_cell.flatten()[process_ind],
                        input_grid.flatten()[process_ind],
                        elev_grid.flatten()[process_ind]))
        with Pool(8,
                  initializer=raster_filter_for_grid_loc_init,
                  initargs=(num_input_rows,
                            num_input_cols,
                            nbc,
                            padded_input,
                            padded_elev)) as pool:
            filter_out = \
                pool.starmap(raster_filter_for_grid_loc,
                             tqdm(args,
                                  total=len(process_ind),
                                  disable=disable_progress,
                                  desc='     Filter'),
                             chunksize=chunksize)
        # Unzip the list of tuples to a 2-element list of tuples.
        filter_out = list(zip(*filter_out))
        filtered_tf = np.array(filter_out[0])
        num_filtered = filtered_tf.sum()
        output_grid_process_subset = np.array(filter_out[1], dtype=np.float64)
        output_grid[process_ind] = output_grid_process_subset
        output_grid = output_grid.reshape(num_input_rows, num_input_cols)
        output_grid = np.ma.masked_invalid(output_grid)

    else:
        num_filtered = 0
        if not idw_pro_style:
            output_grid = input_grid.copy()
        for ind in tqdm(range(num_input_rows * num_input_cols),
                        disable=disable_progress,
                        desc='     Filter'):
            if input_mask[ind]:
                continue

            input_row, input_col = np.unravel_index(ind, input_grid.shape)
            padded_row = input_row + nbc
            padded_col = input_col + nbc
            subgrid = padded_input[padded_row - nbc:padded_row + nbc + 1,
                                   padded_col - nbc:padded_col + nbc + 1]
            if subgrid.count() < 3:
                continue
            subgrid_mean = np.mean(subgrid)
            cell_val_var = abs(input_grid[input_row, input_col] - subgrid_mean)
            subgrid_std = np.std(subgrid)
            if cell_val_var > 1.5 * subgrid_std:
                # "High uncertainty" case; which I would state as "high
                # doubt".
                num_filtered += 1
                if num_points_in_cell[input_row, input_col] > 0:
                    # Mostly preserve the value/s from nearby station/s.
                    if idw_pro_style:
                        padded_input[padded_row, padded_col] = \
                            0.125 * subgrid_mean + \
                            0.875 * padded_input[padded_row, padded_col]
                    else:
                        output_grid[input_row, input_col] = \
                            0.125 * subgrid_mean + \
                            0.875 * padded_input[padded_row, padded_col]
                    # if abs(input_grid[input_row, input_col]) > 0.001:
                    #     print('high doubt, but preserve original ' +
                    #           '(12.5% weight) ' +
                    #           'for {},{}: {}'.
                    #           format(input_row, input_col,
                    #                  input_grid[input_row, input_col]))
                else:
                    if idw_pro_style:
                        padded_input[padded_row, padded_col] = subgrid_mean
                    else:
                        output_grid[input_row, input_col] = subgrid_mean
                    # if abs(input_grid[input_row, input_col]) > 0.001 and \
                    #    input_row % 20 == 0 and \
                    #    input_col % 20 == 0:
                    #     print('(sp) high doubt, convert to the subgrid mean ' +
                    #           'for {},{}: {}'.
                    #           format(input_row, input_col,
                    #                  input_grid[input_row, input_col]))

            elif cell_val_var > 1.0 * subgrid_std:
                # "Low uncertainty" case, which I would state as "moderate
                # doubt".
                num_filtered += 1
                # num_points_in_cell = ((pt_grid_col == input_col) &
                #                       (pt_grid_row == input_row)).sum()
                if num_points_in_cell[input_row, input_col] > 0:
                    # Mostly preserve the value/s from nearby station/s.
                    if idw_pro_style:
                        padded_input[padded_row, padded_col] = \
                            0.125 * subgrid_mean + \
                            0.875 * padded_input[padded_row, padded_col]
                    else:
                        output_grid[input_row, input_col] = \
                            0.125 * subgrid_mean + \
                            0.875 * padded_input[padded_row, padded_col]
                else:
                    # Check to see how typical the cell elevation is for its
                    # local neighborhood.
                    elev_subgrid = \
                        padded_elev[padded_col - nbc:padded_col + nbc + 1,
                                    padded_row - nbc:padded_row + nbc + 1]
                    subgrid_elev_mean = np.mean(elev_subgrid)
                    cell_elev_var = \
                        abs(elev_grid[input_row, input_col] -
                            subgrid_elev_mean)
                    subgrid_elev_std = np.std(elev_subgrid)
                    if cell_elev_var > 0.5 * subgrid_elev_std:
                        # Unusual cell elevation; preserve a bit of the
                        # original output value.
                        if idw_pro_style:
                            padded_input[padded_row, padded_col] = \
                                0.75 * subgrid_mean + \
                                0.25 * padded_input[padded_row, padded_col]
                        else:
                            output_grid[input_row, input_col] = \
                                0.75 * subgrid_mean + \
                                0.25 * padded_input[padded_row, padded_col]
                        # if abs(input_grid[input_row, input_col]) > 0.001 and \
                        #    input_row % 10 == 0 and \
                        #    input_col % 10 == 0:
                        #     print('(sp) some doubt, but preserve original ' +
                        #           '(25% weight) ' +
                        #           'for {},{}: {}'.
                        #           format(input_row, input_col,
                        #                  input_grid[input_row, input_col]))
                    else:
                        if idw_pro_style:
                            padded_input[padded_row, padded_col] = subgrid_mean
                        else:
                            output_grid[input_row, input_col] = subgrid_mean
                        # if abs(input_grid[input_row, input_col]) > 0.001 and \
                           # input_row % 10 == 0 and \
                           # input_col % 10 == 0:
                           #  print('(sp) some doubt, convert to the subgrid ' +
                           #        'mean for {},{}: {}'.
                           #        format(input_row, input_col,
                           #               input_grid[input_row, input_col]))
            else:
                # if abs(input_grid[input_row, input_col]) > 0.001 and \
                #    input_row % 100 == 0 and \
                #    input_col % 100 == 0:
                #     print('no filter for {},{}: {}'.
                #           format(input_row, input_col,
                #                  input_grid[input_row, input_col]))
                pass
        if idw_pro_style:
            output_grid = padded_input[nbc:-nbc, nbc:-nbc]

    message = f'Filtered {num_filtered} of {input_grid.count()} cells.'
    logger.info(message)

    return output_grid


def increment_figure():
    """
    Increase the figure number in matplotlib.pyplot by one.
    """
    if len(plt.get_fignums()) == 0:
        fig_num = 1
    else:
        fig_num = max(plt.get_fignums()) + 1

    return fig_num


# def reg_line(x, intercept, slope):
#     return slope * x + intercept


# def origin_line(x, slope):
#     return slope * x


def main():
    """
    Python port of idw.pro, SNODAS nudging layer creator.
    This program performs inverse distance weighted interpolation of point
    data to generate a nudging layer for SNODAS assimilation.
    """

    mpl.use('TkAgg')
    # mpl.style.use('fast')

    # Make command line argument global so the instance can be adjusted in
    # auto_gen_assim_params.
    # global args

    # Process the command line.
    args = parse_args()
    if args is None:
        # logger.error('Failed to parse command line.')
        sys.exit('Failed to parse command line.')

    if args.quiet or \
        args.log_facility or \
        args.log_file or \
        args.stderr or \
        not sys.stdout.isatty():
        disable_progress=True
    else:
        disable_progress=False

    # args_no_database = True

    # Initialize the logger.
    #logger = local_logger.init(logging.WARNING)
    # if sys.stdout.isatty():
    #     logger.setLevel(logging.INFO)
    ops_logger.init_logging()
    ops_logger.set_logging_options(args)
    logger = logging.getLogger()

    # Make the logging format more user-friendly for interactive use.
    if sys.stdout.isatty() and sys.stderr.isatty():
        # print(logger.handlers[0].formatter._fmt)
        logger.info('Simplifying logging format.')
        logging_format = '%(levelname)s (%(funcName)s): %(message)s'
        formatter = logging.Formatter(logging_format)
        logger.handlers[0].setFormatter(formatter)

    # Check the validity of the command-line arguments.
    args_ok, args = check_args(args)
    if not args_ok:
        sys.exit(1)

    # Initialize graphics. These are the first parameters to adjust to change
    # the size/scale of graphics.
    # mpl.rcParams['font.family'] = 'Calibri'
    # mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.size'] = 12
    target_fig_dim_in = 10.0

    # Accommodate idw.pro (legacy) methods for setting output grid
    # geometry. This should be True as long as we are comparing the output of
    # this IDW process to an existing nudging layer (i.e., during inital
    # development). The idw.pro approach to defining output grid bounds is
    # peculiar and we do not wish to continue it in the future, but we are
    # bound to its idiosyncracies as long as we are looking at existing
    # nudging layers.
    accommodate_idw_pro = False

    # Set the method that will be used to perform distance calculations.
    #distance_method = 'gisrs_great_circle'
    distance_method = 'azimuthal_equidistant_mosaic'
    # distance_method = 'geopy_geodesic'
    # distance_method_pool_size = 0

    # Currently not using the method_func business, instead passing the
    # analysis_method string as a keyword for initializing instances of the
    # IDW class in idw_cross_validate and idw_lon_lat_grid.
    analysis_method = 'classic'
    #analysis_method = 'with_elev_regression'
    if analysis_method == 'classic':
        pass
        # method_func = classic
    elif analysis_method == 'with_elev_regression':
        pass
        # method_func = with_elev_regression
    else:
        message = f'Analysis method "{analysis_method}" is not supported.'
        logger.error(message)
        sys.exit(1)

    start_datetime = dt.datetime.strptime(args.start_date_YmdH, '%Y%m%d%H')
    stop_datetime = dt.datetime.strptime(args.stop_date_YmdH, '%Y%m%d%H')

    if args.file_location is None:
        nsa_prefix = os.getenv('NSA_PREFIX')
        if nsa_prefix is None:
            message = 'Missing environment variable NSA_PREFIX. ' + \
                      'Will try "/operations".'
            logger.warning(message)
            nsa_prefix = '/operations'

    # Set up the output directory for saving figures and any other files.

    # output_dir = os.path.normpath(os.path.join(os.path.dirname(__file__),
    #                                            args.stop_date_YmdH[:-2]))
    if args.file_location is not None:
        output_dir = args.file_location[0]
    else:
        output_dir = \
            os.path.normpath(os.path.join(nsa_prefix,
                                          'ssm',
                                          'assimilation',
                                          'snodas_idw_output',
                                          f'{args.stop_date_YmdH[0:4]}',
                                          f'{args.stop_date_YmdH[4:6]}',
                                          f'{args.stop_date_YmdH[6:8]}'))
    try:
        os.makedirs(output_dir, mode=0o775, exist_ok=True)
        subprocess.run(['chmod', 'g+s', output_dir], check=True)
    except:
        logger.warning(f'Unable to create output directory {output_dir}. ' +
                       f'Will use {os.path.dirname(__file__)}')
        output_dir = os.path.dirname(__file__)
    mpl.rcParams['savefig.directory'] = output_dir

    # Find the input shapefile.
    if args.file_location is not None:
        point_data_dir = args.file_location[0]
    else:
        point_data_dir = os.path.join(nsa_prefix,
                                      'ssm',
                                      'assimilation',
                                      'point_data_shape_new')
    if not os.path.isdir(point_data_dir):
        logger.error('Shapefile directory %s not found.', point_data_dir)
        sys.exit(1)
    point_data_filename = 'ssm1054_md_based_' + \
        f'{args.start_date_YmdH}_{args.stop_date_YmdH}_{args.region_name}'
    point_data_path = \
        os.path.join(point_data_dir, point_data_filename + '.shp')
    if not os.path.exists(point_data_path):
        logger.error('Input shapefile %s not found.', point_data_path)
        sys.exit(1)
    logger.info('Input shapefile path is %s', point_data_path)

    # Read the input shapefile.
    pt_df = read_idw_points(point_data_path)

    # Verify that the essential conventions of get_assim_data.pro have been
    # followed.
    pt_df = verify_get_assim_data_conventions(pt_df)
    if pt_df is None:
        logger.error('Failed to verify get_assim_data conventions.')
        sys.exit(1)

    # Check ranges of relevant columns of pt_df.
    logger.info('MD_SWE range (mm): ' +
                f'[{1000 *pt_df["MD_SWE"].min():.1f}, ' +
                f'{1000 *pt_df["MD_SWE"].max():.1f}]')
    logger.info('MD_DENSITY range (kg m-3): ' +
                f'[{pt_df["MD_DENSITY"].min():.1f}, ' +
                f'{pt_df["MD_DENSITY"].max():.1f}]')
    logger.info('D_SWE_OM range (mm): ' +
                f'[{1000 * pt_df["D_SWE_OM"].min():.1f}, ' +
                f'{1000 * pt_df["D_SWE_OM"].max():.1f}]')
    # # Clean pt_df so all the columns we need are valid.
    # pt_df = clean_pt_df(pt_df)
    message = 'Y (latitude) range: ' + \
              f'[{pt_df["Y"].min()}, {pt_df["Y"].max()}]'
    logger.info(message)
    message = 'X (longitude) range: ' + \
              f'[{pt_df["X"].min()}, {pt_df["X"].max()}]'
    logger.info(message)

    # Get the DEM layer.
    if args.file_location is None and not args.no_database:

        # This is an operations system, so look for the DEM as a GISRS layer.
        dem_elev_layer_name = 'Terrain (GMTED 2010) (30sec) (elevation)'
        dem_elev_path = get_gisrs_layer_data_file_path(dem_elev_layer_name)
        if dem_elev_path is None:
            message = 'Failed to locate layer ' + \
                      f'"{dem_elev_layer_name}".'
            logger.error(message)
            sys.exit(1)
    else:

        if args.file_location is None:

            # This is an operations system and args.file_location was not set,
            # but args.no_database is True. Use the standard location of
            # "common_raster" in operations.
            common_raster_dir = os.path.join(nsa_prefix,
                                             'gisrs',
                                             'data',
                                             'common',
                                             'raster')

        else:

            # This may or may not be an operations system, but
            # args.file_location was set.
            # First look for a "resources" directory relative to the program
            # location.
            common_raster_dir = \
                os.path.normpath(os.path.join(os.path.dirname(__file__),
                                              '..', 'resources'))
            if not os.path.isdir(common_raster_dir):

                # Fall back on args.file_location.
                common_raster_dir = args.file_location[0]

        if not os.path.isdir(common_raster_dir):
            message = f'Common raster directory {common_raster_dir} ' + \
                      'not found.'
            logger.error(message)
            sys.exit(1)
        dem_elev_file_name = 'Terrain_GMTED_2010_30sec.ea.nc'
        dem_elev_path = os.path.join(common_raster_dir, dem_elev_file_name)
        if not os.path.exists(dem_elev_path):
            message = f'DEM file {dem_elev_path} not found.'
            logger.error(message)
            sys.exit(1)

    message = f'Input DEM elevation path is {dem_elev_path}'
    logger.info(message)
    dem_elev_var, dem_elev_hdr, dem_elev = \
        get_nc_geo_raster(dem_elev_path, 'Data')

    # Get DEM column and row (nearest neighbor) indices for the points.
    pt_dem_col = coord_to_grid_ind(pt_df['X'].to_numpy(),
                                   dem_elev_hdr['min_lon_ctr'],
                                   0,
                                   dem_elev_hdr['lon_resolution'])
    pt_dem_row = coord_to_grid_ind(pt_df['Y'].to_numpy(),
                                   dem_elev_hdr['max_lat_ctr'],
                                   0,
                                   dem_elev_hdr['lat_resolution'],
                                   invert=True)

    # Verify that the points fall within the bounds of the DEM. Get rid of any
    # points that are out of bounds.
    point_in_bounds = \
        np.where((pt_dem_col >= 0) &
                 (pt_dem_col < dem_elev_hdr['number_of_columns']) &
                 (pt_dem_row >= 0) &
                 (pt_dem_row < dem_elev_hdr['number_of_rows']))[0]

    in_bounds_count = len(point_in_bounds)
    if in_bounds_count == 0:
        message = 'All point locations are out of bounds on the DEM grid.'
        logger.error(message)
        sys.exit(1)
    if in_bounds_count < len(pt_df):
        message = \
            f'{in_bounds_count} of {len(pt_df)} points ' + \
            'are inside the DEM bounds; ' + \
            f'{len(pt_df) - in_bounds_count} are out of bounds; ' + \
            'these will not be be used.'
        logger.warning(message)
        pt_dem_col = pt_dem_col[point_in_bounds]
        pt_dem_row = pt_dem_row[point_in_bounds]
        pt_df = pt_df.iloc[point_in_bounds].reset_index(drop=True)
    else:
        logger.info('Sampling %d locations from the DEM grid.',
                    in_bounds_count)

    # Get the "benchmark" (row/column center) coordinates that will describe
    # the bounds of the DEM grid after it is clipped to the limits needed to
    # accommodate assimilation.
    if accommodate_idw_pro:

        # Processing bounds are based on padding the point coordinates
        # bounding box with two DEM rows/columns, and DEM subgrid rows and
        # columns are derived from that.
        output_max_lat_ctr = \
            pt_df['Y'].max() + 2.0 * dem_elev_hdr['lat_resolution']
        output_min_lat_ctr = \
            pt_df['Y'].min() - 2.0 * dem_elev_hdr['lat_resolution']
        output_min_lon_ctr = \
            pt_df['X'].min() - 2.0 * dem_elev_hdr['lon_resolution']
        output_max_lon_ctr = \
            pt_df['X'].max() + 2.0 * dem_elev_hdr['lon_resolution']
        dem_row_limits = \
            coord_to_grid_ind(np.array([output_max_lat_ctr,
                                        output_min_lat_ctr]),
                              dem_elev_hdr['max_lat_ctr'],
                              0,
                              dem_elev_hdr['lat_resolution'],
                              invert=True)
        dem_col_limits = \
            coord_to_grid_ind(np.array([output_min_lon_ctr,
                                        output_max_lon_ctr]),
                              dem_elev_hdr['min_lon_ctr'],
                              0,
                              dem_elev_hdr['lon_resolution'])

    else:

        # Processing bounds are based on the DEM geometry, with limits set so
        # that all points will fall within the implied DEM subgrid.
        dem_row_limits = [pt_dem_row.min(), pt_dem_row.max()]
        dem_col_limits = [pt_dem_col.min(), pt_dem_col.max()]
        output_max_lat_ctr = \
            dem_elev_hdr['max_lat_ctr'] - \
            dem_elev_hdr['lat_resolution'] * dem_row_limits[0]
        output_min_lat_ctr = \
            dem_elev_hdr['max_lat_ctr'] - \
            dem_elev_hdr['lat_resolution'] * dem_row_limits[1]
        output_min_lon_ctr = \
            dem_elev_hdr['min_lon_ctr'] + \
            dem_elev_hdr['lon_resolution'] * dem_col_limits[0]
        output_max_lon_ctr = \
            dem_elev_hdr['min_lon_ctr'] + \
            dem_elev_hdr['lon_resolution'] * dem_col_limits[1]

    dem_num_rows = np.diff(dem_row_limits)[0] + 1
    dem_num_cols = np.diff(dem_col_limits)[0] + 1

    if distance_method == 'azimuthal_equidistant_mosaic':
        # Set up a mosaic of azimuthal equidistant projections.
        aeqd_mosaic = geodst.AzEqdMosaic(output_max_lat_ctr +
                                         0.5 * dem_elev_hdr['lat_resolution'],
                                         output_min_lat_ctr -
                                         0.5 * dem_elev_hdr['lat_resolution'],
                                         output_min_lon_ctr -
                                         0.5 * dem_elev_hdr['lon_resolution'],
                                         output_max_lon_ctr +
                                         0.5 * dem_elev_hdr['lon_resolution'])
    else:
        aeqd_mosaic = None

    # Place the DEM raster in memory. It and the processing region raster will
    # be used together in the interpolation process.
    dem_elev_grid = dem_elev_var[:]
    dem_elev.close()

    # Sample elevations for point locations from the DEM grid.
    pt_df['DEM_ELEVAT'] = dem_elev_grid[pt_dem_row, pt_dem_col]

    # TODO: make sure DEM_ELEVAT values are all legit.

    # Subset the DEM grid. The original idw.pro expanded the coordinate limits
    # by two DEM rows/columns, but this seems to have been more due to an
    # abundance of caution than anything truly needed.
    dem_elev_grid_clipped = \
        dem_elev_grid[dem_row_limits[0]: dem_row_limits[1] + 1,
                      dem_col_limits[0]: dem_col_limits[1] + 1]

    # Find the processing region layer.
    if not args.no_database and args.file_location is None:
        proc_region_layer_name = 'ssm_process_region_{}_{}_swe_{}'. \
            format(args.start_date_YmdH, args.stop_date_YmdH,
                   args.region_name)
        proc_region_path = \
            get_gisrs_layer_data_file_path(proc_region_layer_name)
        if proc_region_path is None:
            message = f'Failed to locate layer "{proc_region_layer_name}.'
            logger.error(message)
            sys.exit(1)
    else:
        if args.file_location is not None:
            proc_region_dir = args.file_location[0]
        else:
            proc_region_dir = os.path.join(nsa_prefix,
                                           'ssm',
                                           'data',
                                           'snow_model',
                                           'raster',
                                           'sm_swe_correct')
        if not os.path.isdir(proc_region_dir):
            message = f'Processing region directory {proc_region_dir} ' + \
                      'not found.'
            logger.error(message)
            sys.exit(1)
        proc_region_file_name = \
            'ssm_process_region_' + \
            f'{args.start_date_YmdH}_{args.stop_date_YmdH}_swe_' + \
            f'{args.region_name}.nc'
        proc_region_path = \
            os.path.join(proc_region_dir, proc_region_file_name)
        if not os.path.exists(proc_region_path):
            message  = f'Processing region path {proc_region_path} ' + \
                       'not found.'
            logger.error(message)
            sys.exit(1)

    message = f'Input processing region path is {proc_region_path}'
    logger.info(message)

    # Read the processing region layer.
    proc_region_var, proc_region_hdr, proc_region = \
        get_nc_geo_raster(proc_region_path, 'Data')

    # Get processing region locations needed to match the DEM after it is
    # subsetted (though this has not been done yet).
    proc_region_row_limits = \
        coord_to_grid_ind(np.array([output_max_lat_ctr, output_min_lat_ctr]),
                          proc_region_hdr['max_lat_ctr'],
                          0,
                          proc_region_hdr['lat_resolution'],
                          invert=True)
    proc_region_num_rows = np.diff(proc_region_row_limits)[0] + 1
    proc_region_col_limits = \
        coord_to_grid_ind(np.array([output_min_lon_ctr, output_max_lon_ctr]),
                          proc_region_hdr['min_lon_ctr'],
                          0,
                          proc_region_hdr['lon_resolution'])
    proc_region_num_cols = np.diff(proc_region_col_limits)[0] + 1

    # Verify that the processing region and DEM subgrids will be consistent.
    if proc_region_col_limits[0] < 0 or \
       proc_region_col_limits[1] >= proc_region_hdr['number_of_columns'] or \
       proc_region_num_cols != dem_num_cols or \
       proc_region_row_limits[0] < 0 or \
       proc_region_row_limits[1] >= proc_region_hdr['number_of_rows'] or \
       proc_region_num_rows != dem_num_rows:
        logger.error('DEM and processing region coverages are inconsistent.')
        sys.exit(1)

    # Subset the processing region grid.
    proc_region_grid_clipped = \
        proc_region_var[:][proc_region_row_limits[0]:
                           proc_region_row_limits[1] + 1,
                           proc_region_col_limits[0]:
                           proc_region_col_limits[1] + 1]
    proc_region.close()

    # Set up axes for plotting.
    proc_region_lat_axis = \
        np.linspace(proc_region_hdr['max_lat_ctr'],
                    proc_region_hdr['min_lat_ctr'],
                    proc_region_hdr['number_of_rows']) \
                    [proc_region_row_limits[0]: proc_region_row_limits[1] + 1]

    proc_region_lon_axis = \
        np.linspace(proc_region_hdr['min_lon_ctr'],
                    proc_region_hdr['max_lon_ctr'],
                    proc_region_hdr['number_of_columns']) \
                    [proc_region_col_limits[0]:proc_region_col_limits[1] + 1]

    bbox = [proc_region_lon_axis[0] - 0.5 * proc_region_hdr['lon_resolution'],
            proc_region_lon_axis[-1] + 0.5 * proc_region_hdr['lon_resolution'],
            proc_region_lat_axis[-1] - 0.5 * proc_region_hdr['lat_resolution'],
            proc_region_lat_axis[0] + 0.5 * proc_region_hdr['lat_resolution']]

    # The dem_elev_grid_clipped and proc_region_grid_clipped establish the
    # output grid geometry. Calculate point locations within that output
    # grid.
    pt_output_row = pt_dem_row - dem_row_limits[0]
    pt_output_col = pt_dem_col - dem_col_limits[0]

    if args.gui:
        # Enable interactive plots.
        plt.ion()

    # Create a subset of pt_df that eliminates correct negatives.
    event_threshold_m = 1.0e-4
    pt_no_cn = pt_df[(pt_df['OB_SWE'] > event_threshold_m) | \
                     (pt_df['MD_SWE'] > event_threshold_m)]

    # Create a subset of pt_df that removes artificial zeroes. Simulated
    # OB_SWE derived from OB_DEPTH and MD_DENSITY *are still included* in
    # pt_real, so "pt_is_real" does not imply that OB_SWE itself is real.
    pt_is_real = pt_df['STATION_ID'].notna() & \
        pt_df['OB_SWE'].notna() & \
        pt_df['MD_SWE'].notna() & \
        pt_df['D_SWE_OM'].notna() & \
        pt_df['X'].notna() & \
        pt_df['Y'].notna()
    pt_real = pt_df[pt_is_real]
    # print(len(pt_real))
    # sys.exit(1)

    # Inventory STATION_TY from the pt_real subset.
    logger.info('Counts of snow data values by STATION_TY column:')
    logger.info(pt_real['STATION_TY'].value_counts())
    message = f'Total # of actual observing stations: {len(pt_real)}'
    logger.info(message)

    # Display the points.
    timer = Timer()
    if not args.skip_map:
        timer.start()
        message = 'Generating a map of \N{GREEK CAPITAL LETTER DELTA}SWE points...'
        logger.info(message)
        delta_swe_map_title = \
            '\N{GREEK CAPITAL LETTER DELTA}SWE for ' + \
            f'{stop_datetime.strftime("%Y-%m-%d")}'
        s_fig = gen_delta_swe_mm_map(bbox,
                                     pt_real,
                                     target_fig_dim_in=target_fig_dim_in,
                                     title=delta_swe_map_title,
                                     keep_pt_crs=True)
        if s_fig is None:
            sys.exit(1)

        output_img = \
            'snodas_assim_points_' + \
            f'{args.stop_date_YmdH}_{args.region_name}.png'
        output_path = os.path.join(output_dir, output_img)
        s_fig.savefig(output_path)
        message = 'Saved \N{GREEK CAPITAL LETTER DELTA}SWE points map ' + \
                  f'to {output_path}'
        logger.info(message)

        if args.gui:
            plt.pause(1.0)

        timer.stop()

    # plt.pause(1.0) updates/displays graphics and runs the event loop very
    # briefly. Would plt.draw(block=False) do the same?
    # plt.pause(1.0)
    # plt.ioff()
    # plt.show(block=False)
    # sys.exit(0)

    # Read and display the real nudging layer produced by IDL.
    if not args.no_database and args.file_location is None:
        nudging_layer_name = f'ssm1054_{args.stop_date_YmdH}'
        nudging_layer_path = \
            get_gisrs_layer_data_file_path(nudging_layer_name)
        if nudging_layer_path is None:
            # Try adding the region.
            message = f'Failed to locate layer "{nudging_layer_name}.'
            logger.warning(message)
            nudging_layer_name = \
                nudging_layer_name + f'_{args.region_name}'
            nudging_layer_path = \
                get_gisrs_layer_data_file_path(nudging_layer_name)
            if nudging_layer_path is None:
                message = f'Failed to locate layer "{nudging_layer_name}"".'
                logger.error(message)
    else:
        nudging_layer_path = None
        if args.file_location is not None:
            nudging_dir = args.file_location[0]
        else:
            nudging_dir = os.path.join(nsa_prefix,
                                       'ssm',
                                       'data',
                                       'snow_model',
                                       'raster',
                                       'sm_swe_correct')
        if not os.path.isdir(nudging_dir):
            message = f'Nudging layer directory {nudging_dir} not found.'
            logger.error('nudging layer directory %s not found.',
                         proc_region_dir)
        else:
            nudging_layer_name = f'ssm1054_{args.stop_date_YmdH}'
            nudging_file_name = f'{nudging_layer_name}.nc'
            nudging_layer_path = os.path.join(nudging_dir, nudging_file_name)
            if not os.path.exists(nudging_layer_path):
                # Try including the region.
                message = f'Failed to locate {nudging_layer_path}.'
                logger.warning(message)
                nudging_file_name = \
                    f'ssm1054_{args.stop_date_YmdH}_{args.region_name}.nc'
                nudging_layer_path = \
                    os.path.join(nudging_dir, nudging_file_name)
                if not os.path.exists(nudging_layer_path):
                    message = f'Also failed to locate {nudging_layer_path}.'
                    logger.error(message)
                    nudging_layer_path = None

    if nudging_layer_path is not None:

        if nudging_layer_path[-3:] != '.nc':
            message = \
                f'Non-NetCDF files like {nudging_layer_path} not supported.'
            logger.warning(message)
            nudging_layer_path = os.path.join('/net', 'tmp',
                                              nudging_layer_name + '.nc')
            if not os.path.exists(nudging_layer_path):
                nudging_layer_path = None

    if nudging_layer_path is not None:

        # Read and plot nudging layer.
        message = 'Checking results against nudging data in ' + \
                  f'{nudging_layer_path}.'
        logger.info(message)

        nudging_var, nudging_hdr, _ = \
            get_nc_geo_raster(nudging_layer_path, 'Data')

        nudging_row_limits = \
            coord_to_grid_ind(np.array([output_max_lat_ctr,
                                        output_min_lat_ctr]),
                              nudging_hdr['max_lat_ctr'],
                              0,
                              nudging_hdr['lat_resolution'],
                              invert=True)
        nudging_num_rows = np.diff(nudging_row_limits)[0] + 1

        nudging_col_limits = \
            coord_to_grid_ind(np.array([output_min_lon_ctr,
                                        output_max_lon_ctr]),
                              nudging_hdr['min_lon_ctr'],
                              0,
                              nudging_hdr['lon_resolution'])
        nudging_num_cols = np.diff(nudging_col_limits)[0] + 1

        # The DEM, processing region, and nudging layer are in general not
        # consistently aligned, and occasionally as a consequence
        # nudging_row_limits and/or nudging_col_limits will result in an
        # inconsistent shape. That gets handled here.
        if nudging_num_rows != dem_num_rows and \
           abs(nudging_num_rows - dem_num_rows) == 1 and \
           (nudging_row_limits[0] != 0 or \
            nudging_row_limits[1] != nudging_hdr['number_of_rows'] - 1):

            # Since nudging_num_rows and dem_num_rows differ only by one, add
            # or remove a nudging row to give them consistent row counts.
            old_nudging_row_limits = nudging_row_limits.copy()
            if nudging_num_rows > dem_num_rows:
                # Remove the last nudging row. This is an arbitrary choice.
                nudging_row_limits[1] = nudging_row_limits[1] - 1
            else:
                # Add a nudging row at the start. This is an arbitrary choice.
                nudging_row_limits[0] = nudging_row_limits[0] - 1
            # Prevent the first nudging_row_limits value from being -1.
            if nudging_row_limits[0] == -1:
                nudging_row_limits = nudging_row_limits + 1
            nudging_num_rows = np.diff(nudging_row_limits)[0] + 1
            message = 'Adjusted nudging row range from ' + \
                f'{old_nudging_row_limits} to {nudging_row_limits}.'
            logger.warning(message)

        if nudging_num_cols != dem_num_cols and \
           abs(nudging_num_cols - dem_num_cols) == 1 and \
           (nudging_col_limits[0] != 0 or \
            nudging_col_limits[1] != nudging_hdr['number_of_cols'] - 1):

            # Since nudging_num_cols and dem_num_cols differ only by one, add
            # or remove a nudging column to give them consistent column
            # counts.
            old_nudging_col_limits = nudging_col_limits.copy()
            if nudging_num_cols > dem_num_cols:
                # Remove the last nudging column. This is an arbitrary choice.
                nudging_col_limits[1] = nudging_col_limits[1] - 1
            else:
                # Add a nudging column at the start. This is an arbitrary
                # choice.
                nudging_col_limits[0] = nudging_col_limits[0] - 1
            # Prevent the first nudging_col_limits value from being -1.
            if nudging_col_limits[0] == -1:
                nudging_col_limits = nudging_col_limits + 1
            nudging_num_cols = np.diff(nudging_col_limits)[0] + 1
            message = 'Adjusted nudging column range from ' + \
                f'{old_nudging_col_limits} to {nudging_col_limits}.'
            logger.warning(message)

        # Confirm that nudging_row_limits and nudging_col_limits will result
        # in a nudging grid shape consistent with that of the DEM subgrid
        # a.k.a. output grid.
        if nudging_col_limits[0] < 0 or \
           nudging_col_limits[1] >= nudging_hdr['number_of_columns'] or \
           nudging_num_cols != dem_num_cols or \
           nudging_row_limits[0] < 0 or \
           nudging_row_limits[1] >= nudging_hdr['number_of_rows'] or \
           nudging_num_rows != dem_num_rows:
            message = 'Output and nudging layer coverages are inconsistent'
            logger.error(message)
            sys.exit(1)

        nudging_grid = nudging_var[:][nudging_row_limits[0]:
                                      nudging_row_limits[1] + 1,
                                      nudging_col_limits[0]:
                                      nudging_col_limits[1] + 1]

        if nudging_var.units not in ('meters', 'm'):
            message = f'Nudging layer units are "{nudging_var.units}"; ' + \
                      '"meters" or "m" expected.'
            logger.error(message)
            sys.exit(1)
        message = 'nudging grid range (mm): ' + \
                  f'[{1000 * np.min(nudging_grid):.4f}, ' + \
                  f'{1000 * np.max(nudging_grid):.4f}]'
        logger.info(message)

        nudging_lon_axis = np.linspace(nudging_hdr['min_lon_ctr'],
                                       nudging_hdr['max_lon_ctr'],
                                       nudging_hdr['number_of_columns']) \
                                       [nudging_col_limits[0]:
                                        nudging_col_limits[1] + 1]
        nudging_lat_axis = np.linspace(nudging_hdr['max_lat_ctr'],
                                       nudging_hdr['min_lat_ctr'],
                                       nudging_hdr['number_of_rows']) \
                                       [nudging_row_limits[0]:
                                        nudging_row_limits[1] + 1]

        # Display the nudging layer.
        color_ramp = mng.delta_swe_mm()
        colorbar_units = 'mm'
        title = f'idw.pro nudging layer\n{nudging_layer_name}'

        # "n_" for "nudging"
        n_fig, n_geo_axes, _ = \
            mng.mpl_map_2023(nudging_grid * 1000,
                             nudging_lat_axis,
                             nudging_hdr['lat_resolution'],
                             nudging_lon_axis,
                             nudging_hdr['lon_resolution'],
                             'SNODAS_nudging',
                             color_ramp,
                             colorbar_units,
                             title,
                             cbar_scale=0.67,
                             fig_num=increment_figure())

        output_img = \
            os.path.splitext(os.path.basename(nudging_layer_path))[0] + \
            '.png'
        output_path = os.path.join(output_dir, output_img)
        n_fig.savefig(output_path)
        message = f'Saved nudging layer image to {output_path}'
        logger.info(message)

        # plt.draw()
        # plt.ioff()
        # plt.show()
        # sys.exit(0)

    if args.gui:
        plt.pause(1.0)
    #print(f'plt.isinteractive: {plt.isinteractive()}')
    #plt.pause(0.1)
    #print('ok')

    if args.quick_look:
        logger.info('Quick look finished.')
        if args.gui:
            # for i in range(10):
            #     plt.pause(1.0)
            #     print(f'pause {i+1}')
            logger.info('Close plot windows to quit.')
            plt.ioff()
            plt.show()
        sys.exit(0)

    if args.idw_params is None:

        message = 'No interpolation parameters were provided ' + \
                  'in the command line.'
        logger.info(message)
        message = 'Using cross validation to determine a set of ' + \
                  '"best" parameters.'
        logger.info(message)

        cv_r_squared, cv_rmse_mm, cv_mae_mm, args = \
            auto_gen_assim_params(args,
                                  pt_df,
                                  analysis_method=analysis_method,
                                  distance_method=distance_method,
                                  aeqd_mosaic=aeqd_mosaic,
                                  disable_progress=disable_progress)

        # print('best R\N{SUPERSCRIPT TWO} result:')
        logger.info('Stats for selected parameters:')
        message = f'R\N{SUPERSCRIPT TWO}: {cv_r_squared:.3f}'
        logger.info(message)
        # print('proportion of points predicted: ' +
        #       f'{cv_proportion[best_fit_param_ind]}')
        message = 'RMSE: ' + f'{cv_rmse_mm:.1f} mm'
        logger.info(message)
        message = 'MAE: ' + f'{cv_mae_mm:.1f} mm'
        logger.info(message)
        message = f'# points [{args.min_num_points}, {args.max_num_points}]'
        logger.info(message)
        message = f'horiz. power {args.horizontal_power}'
        logger.info(message)
        message = f'elev. power {args.elevation_power}'
        logger.info(message)
        message = f'max. distance {args.max_distance_km} km'
        logger.info(message)

    # plt.pause(1.0)

    if args.cross_validate:

        # Perform cross validation IDW to estimate errors.

        logger.info('Estimating errors via cross validation...')

        timer.start()
        delta_swe_cv = idw_cross_validate(pt_no_cn,
                                          'D_SWE_OM',
                                          args.max_distance_km,
                                          args.min_num_points,
                                          args.max_num_points,
                                          args.horizontal_power,
                                          args.elevation_power,
                                          analysis_method=analysis_method,
                                          distance_method=distance_method,
                                          aeqd_mosaic=aeqd_mosaic,
                                          disable_progress=disable_progress)
        timer.stop()

        # Evaluate errors in delta_swe_cv.
        delta_swe_om = pt_no_cn['D_SWE_OM'].to_numpy()
        delta_swe_low_limit = 0.0
        ind = np.where(np.isfinite(delta_swe_cv) & \
                       np.isfinite(delta_swe_om) & \
                       ((np.abs(delta_swe_om) > delta_swe_low_limit) | \
                        (np.abs(delta_swe_cv) > delta_swe_low_limit)))[0]
        message = 'Cross-validation (CV) stats for delta SWE ' + \
            f'exceeding +/-{delta_swe_low_limit} m:'
        logger.info(message)
        message = 'Absolute CV \N{GREEK CAPITAL LETTER DELTA}SWE ' + \
            f'exceeds {delta_swe_low_limit} ' + \
            f'for {len(ind)} of {len(pt_df)} points.'
        logger.info(message)
        # print(len(np.where(np.isfinite(delta_swe_cv))[0]))
        # print(len(np.where(np.isfinite(delta_swe_om))[0]))
        cv_error = delta_swe_cv[ind] - delta_swe_om[ind]
        # print(delta_swe_cv[ind[10]], delta_swe_om[ind[10]])
        # print(np.median(delta_swe_om[ind]))
        message = f'CV minimum error: {cv_error.min() * 1000:.2f} mm'
        logger.info(message)
        message = f'CV median error: {np.median(cv_error) * 1000:.2f} mm'
        logger.info(message)
        message = f'CV maximum error: {cv_error.max() * 1000:.2f} mm'
        logger.info(message)
        message = f'CV mean error: {np.mean(cv_error) * 1000:.2f} mm'
        logger.info(message)
        message = f'CV std. dev. in error: {np.std(cv_error) * 1000:.2f} mm'
        logger.info(message)
        cv_rmse_mm = np.sqrt(np.mean(cv_error**2)) * 1000
        message = f'CV RMSE: {cv_rmse_mm:.2f} mm'
        logger.info(message)
        cv_mae_mm = np.mean(np.abs(cv_error)) * 1000
        message = f'CV MAE: {cv_mae_mm:.2f} mm'
        logger.info(message)
        # Use standard scores to calculate correlation.
        # ssx = (delta_swe_om[ind] - np.mean(delta_swe_om[ind])) / \
        #       np.std(delta_swe_om[ind], ddof=1)
        # ssy = (delta_swe_cv[ind] - np.mean(delta_swe_cv[ind])) / \
        #       np.std(delta_swe_cv[ind], ddof=1)
        # cv_r_squared = (np.sum(ssx * ssy) / (len(ind) - 1.0))**2
        # Use the textbook definition for R squared.
        ss_res = np.sum((delta_swe_om[ind] - delta_swe_cv[ind])**2)
        ss_tot = np.sum((delta_swe_om[ind] - np.mean(delta_swe_om[ind]))**2)
        cv_r_squared = 1.0 - ss_res / ss_tot
        message = f'CV R\N{SUPERSCRIPT TWO} = {cv_r_squared:.5f}'
        logger.info(message)

        cv_title = 'IDW Cross-Validation Results for\n' + \
                   f'{stop_datetime.strftime("%Y-%m-%d")}'
        cv_fig, cv_ax = \
            plot_idw_cross_val_results(-delta_swe_om[ind] * 1000,
                                       -delta_swe_cv[ind] * 1000,
                                       pt_no_cn['STATION_ID'].to_numpy()[ind],
                                       cv_title,
                                       'Observed ' + \
                                       '\N{GREEK CAPITAL LETTER DELTA}' + \
                                       'SWE (mm)',
                                       'Analysis ' + \
                                       '\N{GREEK CAPITAL LETTER DELTA}' + \
                                       'SWE (mm)',
                                       args.min_num_points,
                                       args.max_num_points,
                                       args.horizontal_power,
                                       args.elevation_power,
                                       args.max_distance_km,
                                       cv_r_squared,
                                       cv_rmse_mm,
                                       cv_mae_mm)

        output_img = f'snodas_idw_cv_{args.stop_date_YmdH}_' + \
            f'{args.region_name}_' + \
            f'{args.min_num_points}_{args.max_num_points}_' + \
            f'{args.horizontal_power}_{args.elevation_power}_' + \
            f'{args.max_distance_km:.0f}.png'
        output_path = os.path.join(output_dir, output_img)
        cv_fig.savefig(output_path)
        message = f'Saved cross-validation plot to {output_path}'
        logger.info(message)

        if args.gui:
            plt.pause(1.0)

    if args.dry_run:
        logger.info('LOOCV a.k.a. "jacknife" cross-validation finished.')
        if args.gui:
            logger.info('Close plot windows to quit.')
            plt.ioff()
            plt.show()
        sys.exit(0)

    # Perform basic IDW interpolation on the grid. Output grid parameters are
    # based on the DEM.
    timer.start()

    output_lat_res = dem_elev_hdr['lat_resolution']
    output_lon_res = dem_elev_hdr['lon_resolution']
    output_num_rows = dem_row_limits[1] - dem_row_limits[0] + 1
    output_num_cols = dem_col_limits[1] - dem_col_limits[0] + 1
    message = f'Output grid shape: ({output_num_rows},{output_num_cols})'
    logger.info(message)
    if accommodate_idw_pro:
        # To match the way it is done in idw.pro, output grid dimensions and
        # lon/lat axes are based on minimum/maximum point coordinates combined
        # with the DEM resolution and two-cell padding. One possible
        # complication here is caused by the fact that
        # output_[min,max]_[lon,lat]_ctr are used to determine the subgrids of
        # the DEM and processing region used here, and in that context are
        # treated as coordinates for finding nearest neighbors, but in
        # idw.pro, when WRITE_ARC_RASTER is called, they are used to establish
        # EDGE coordinates xllcorner and yllcorner in the Arc/Info header. We
        # mimic that here, even though it is weird and confusing.
        # To follow that approach, note that due to the way they are
        # calculated, the ranges of output_[min,max]_[lon,at]_ctr are in NO
        # WAY expected to be consistent with an integer number of output grid
        # cells in the idw.pro-friendly framework. That has to be fixed here,
        # and that is accomplished by basing the output_lat_axis and
        # output_lon_axis on output_min_lat_ctr and
        # output_min_lat_ctr, while the former output_max_lat_ctr and
        # output_max_lon_ctr (used previously to choose row/column ranges to
        # subset from the DEM and processing region layer) are ignored.
        output_min_lat_ctr = \
            output_min_lat_ctr + 0.5 * dem_elev_hdr['lat_resolution']
        output_max_lat_ctr = \
            output_min_lat_ctr + \
            (output_num_rows - 1) * dem_elev_hdr['lat_resolution']
        output_min_lon_ctr = \
            output_min_lon_ctr + 0.5 * dem_elev_hdr['lon_resolution']
        output_max_lon_ctr = \
            output_min_lon_ctr + \
            (output_num_cols - 1) * dem_elev_hdr['lon_resolution']

    output_lat_axis = np.linspace(output_max_lat_ctr, output_min_lat_ctr,
                                  output_num_rows)
    output_lon_axis = np.linspace(output_min_lon_ctr, output_max_lon_ctr,
                                  output_num_cols)

    message = 'IDW parameters: ' + \
              f'{args.min_num_points} ' + \
              f'{args.max_num_points} ' + \
              f'{args.horizontal_power} ' + \
              f'{args.elevation_power} ' + \
              f'{args.max_distance_km}'
    logger.info(message)

    idw_result = idw_lon_lat_grid(pt_df,
                                  'D_SWE_OM',
                                  args.max_distance_km,
                                  args.min_num_points,
                                  args.max_num_points,
                                  args.horizontal_power,
                                  args.elevation_power,
                                  dem_elev_grid_clipped,
                                  proc_region_grid_clipped,
                                  output_lon_axis,
                                  output_lat_axis,
                                  analysis_method=analysis_method,
                                  distance_method=distance_method,
                                  aeqd_mosaic=aeqd_mosaic,
                                  disable_progress=disable_progress)
    timer.stop()

    if args.gui:
        plt.pause(1.0)

    idw_result = np.ma.masked_invalid(idw_result)

    logger.info('IDW result range (mm): ' +
                f'{np.min(idw_result) * 1000:.2f}, ' +
                f'{np.max(idw_result) * 1000:.2f}')

    # Filter the data.
    idw_pro_style = False
    multiprocess = True
    skip_filter = False
    if not skip_filter:

        timer.start()
        idw_result = idw_pro_raster_filter(idw_result,
                                           dem_elev_grid_clipped,
                                           pt_output_row,
                                           pt_output_col,
                                           pt_df,
                                           idw_pro_style=idw_pro_style,
                                           multiprocess=multiprocess,
                                           disable_progress=disable_progress)
        timer.stop()
        logger.info('IDW (filtered) result range (mm): ' +
                    f'{np.min(idw_result) * 1000:.2f}, ' +
                    f'{np.max(idw_result) * 1000:.2f}')
    else:
        logger.warning('Skipping the traditional raster filter.')

    # if multiprocess:
    #     pkl_file='foo_mp.pkl'
    # else:
    #     pkl_file='foo_sp.pkl'
    # with open(pkl_file, 'wb') as f:
    #     pkl.dump(idw_result, f)
    # print('stored idw output in {}'.format(pkl_file))

    if args.gui:
        plt.pause(1.0)

    # Display the IDW result.
    color_ramp = mng.delta_swe_mm()
    colorbar_units = 'mm'
    title = f'IDW result:\n{stop_datetime.strftime("%Y-%m-%d")}'

    # "r_" for "result"
    r_fig, r_geo_axes, _ = \
        mng.mpl_map_2023(idw_result * 1000,
                         output_lat_axis,
                         output_lat_res,
                         output_lon_axis,
                         output_lon_res,
                         'SNODAS_nudging',
                         color_ramp,
                         colorbar_units,
                         title,
                         cbar_scale=0.67,
                         fig_num=increment_figure(),
                         debug_graphics=False)

    # Draw the points invisibly to help with legend placement.
    r_geo_axes.scatter(pt_df['X'], pt_df['Y'], s=0)

    params_text = \
        [f'analysis method: {analysis_method}',
         f'distance method: {distance_method}',
         f'min # points: {args.min_num_points}; ' + \
         f'max # points: {args.max_num_points}',
         f'horiz. power: {args.horizontal_power:.1f}']
    if analysis_method == 'classic':
        params_text[3] = params_text[3] + \
            f'; elev. power: {args.elevation_power:.1f}'
        params_text.append(
            f'search distance: {args.max_distance_km:.1f} km')

        handles = [patches.Rectangle((0,0), 0, 0)] * len(params_text)
        r_leg = r_geo_axes.legend(handles,
                                  params_text,
                                  loc='best',
                                  handlelength=0, handletextpad=0,
                                  fontsize='x-small',
                                  labelcolor='white',
                                  labelspacing=0.4,
                                  frameon=False)
        # Redraw the legend labels atop a black stroke. This is really slow.
        for text in r_leg.texts:
            text.set_path_effects(
                [path_effects.Stroke(linewidth=3,
                                     foreground='black',
                                     alpha=0.5),
                 path_effects.Normal()])

    if args.gui:
        plt.pause(1.0)

    output_img = f'snodas_idw_nudging_{args.stop_date_YmdH}_' + \
                 f'{args.region_name}_' + \
                 f'{args.min_num_points}_{args.max_num_points}_' + \
                 f'{args.horizontal_power}_{args.elevation_power}_' + \
                 f'{args.max_distance_km:.0f}.png'
    output_path = os.path.join(output_dir, output_img)
    r_fig.savefig(output_path)
    message = f'Saved IDW result to {output_path}'
    logger.info(message)

    if nudging_layer_path is not None:

        # Examine differences between the IDW result and the nudging layer.
        # print(type(idw_result))
        # print(type(nudging_grid))
        # print(np.min(nudging_grid), np.max(nudging_grid))
        idw_diff = idw_result - nudging_grid
        # Are there any idw_result values that are unmasked but NaN?
        message = 'Summary of differences between IDW result ' + \
                  f'and existing layer "{nudging_layer_name}":'
        logger.info(message)
        message = f'Minimum difference: {idw_diff.min() * 1000:.2f} mm'
        logger.info(message)
        message = 'Median difference: ' + \
                  f'{np.ma.median(idw_diff) * 1000:.2f} mm'
        logger.info(message)
        message = f'Maximum difference: {idw_diff.max() * 1000:.2f} mm'
        logger.info(message)
        message = f'Mean difference: {np.mean(idw_diff) * 1000:.2f} mm'
        logger.info(message)
        message = 'Std. dev. in difference: ' + \
                  f'{np.std(idw_diff) * 1000:.2f} mm'
        logger.info(message)
        message = 'RMS difference: ' + \
                  f'{np.sqrt(np.mean(np.power(idw_diff,2))) * 1000:.2f} mm'
        logger.info(message)
        message = 'Mean absolute difference: ' + \
                  f'{np.mean(np.abs(idw_diff)) * 1000:.2f} mm'
        logger.info(message)

        color_ramp = mng.delta_swe_mm()
        colorbar_units = 'mm'
        title = f'IDW result difference from\n{nudging_layer_name}'
        # "d_" for "difference"
        d_fig, d_geo_axes, _ = \
            mng.mpl_map_2023(idw_diff * 1000,
                             output_lat_axis,
                             output_lat_res,
                             output_lon_axis,
                             output_lon_res,
                             'SNODAS_nudging_diff',
                             color_ramp,
                             colorbar_units,
                             title,
                             cbar_scale=0.67,
                             fig_num=increment_figure())

        if args.gui:
            plt.pause(1.0)

        output_img = f'snodas_idw_nudging_diff_{args.stop_date_YmdH}_' + \
                     f'{args.region_name}_' + \
                     f'{args.min_num_points}_{args.max_num_points}_' + \
                     f'{args.horizontal_power}_{args.elevation_power}_' + \
                     f'{args.max_distance_km:.0f}.png'
        output_path = os.path.join(output_dir, output_img)
        d_fig.savefig(output_path)
        message = 'Saved difference between IDW result and ' + \
                  f'existing nudging layer to {output_path}'
        logger.info(message)

    #print('figure numbers: ', plt.get_fignums())
    logger.info('Nudging grid creation finished.')

    output_nc_file_name = \
        f'ssm1054_snodas_idw_test_{args.stop_date_YmdH}_{args.region_name}.nc'
    output_nc_file_path = os.path.join(output_dir, output_nc_file_name)
    long_name = 'A SWE nudging layer for NSA data assimilation'
    units = 'meters'
    create_module = os.path.basename(__file__)
    create_comment = 'Data created in Python using 3-D IDW interpolation'

    if args.file_location is None:
        color_table_dir = os.path.join(nsa_prefix,
                                       'gisrs',
                                       'color_tables')
        color_table_file = 'swe_nudge.ct'
        color_table_path = os.path.join(color_table_dir, color_table_file)
        if not os.path.exists(color_table_path):
            color_table_path = None
    else:
        color_table_path = None
    write_nc_nudging_raster(idw_result,
                            output_lat_axis,
                            output_lat_res,
                            output_lon_axis,
                            output_lon_res,
                            long_name,
                            units,
                            start_datetime,
                            stop_datetime,
                            create_module,
                            create_comment,
                            output_nc_file_path,
                            color_table_path=color_table_path)
    message = f'Created nudging file {output_nc_file_path}'
    logger.info(message)

    # TODO
    # if not args.no_layer:
    #     Create a GISRS layer from the output_nc_file_path.
    #     logger.info(f'Converting {output_nc_file_path} to a GISRS layer...')

    if args.gui:
        logger.info('Close plot windows to quit.')
        plt.ioff()
        plt.show()


if __name__ == '__main__':
    main()
