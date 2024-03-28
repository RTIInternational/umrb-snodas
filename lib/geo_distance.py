#!/usr/bin/env python3
'''
A library for calculating distances between earth coordinates. Its first and
primary function, one_to_many_km, can take different options, trading
performance for accuracy, and with some multiprocessing options. Since the
function is intended for spatial analysis applications (e.g. snodas_idw.py)
the chief objective is calculating the distance between a single location and
many others.
'''

import logging
import numpy as np
import math
from geopy import distance
from geographiclib.geodesic import Geodesic
from pyproj import Proj
# import itertools
from multiprocessing import Pool
# import time
from tqdm import tqdm

class Error(Exception):
    '''
    Base class for homemade exceptions.
    '''

class NotSupported(Error):
    '''
    Exception for unsupported function keywords.
    '''

class AzEqdMosaic:
    '''
    A mosaic of azimuthal equidistant map projections for estimating distances
    between a central location and points in its surrounding neighborhood. The
    purpose is to enable distance calculations more accurate than great circle
    estimates, but faster than full ellipsoid calculations.
    '''
    def __init__(self,
                 max_lat,
                 min_lat,
                 min_lon,
                 max_lon,
                 tile_deg=4.0):
        self.tile_deg = tile_deg
        # The stop argument for arange is extended slightly because arange
        # uses a half-open interval. 
        # self.tile_ctr_lat = np.arange(max_lat - 0.5 * tile_deg,
        #                               min_lat - 0.500001 * tile_deg,
        #                               -tile_deg)
        # self.tile_rows = len(self.tile_ctr_lat)
        # self.tile_ctr_lon = np.arange(min_lon + 0.5 * tile_deg,
        #                               max_lon + 0.500001 * tile_deg,
        #                               tile_deg)
        # self.tile_cols = len(self.tile_ctr_lon)

        # The minimum latitude and maximum longitude of the mosaic of
        # projections may differ from the input arguments.
        self.tile_rows = math.ceil((max_lat - min_lat) / tile_deg)
        self.max_lat = max_lat
        self.min_lat = self.max_lat - self.tile_rows * self.tile_deg
        self.tile_ctr_lat = np.linspace(self.max_lat - 0.5 * self.tile_deg,
                                        self.min_lat + 0.5 * self.tile_deg,
                                        self.tile_rows)
        self.tile_cols = math.ceil((max_lon - min_lon) / tile_deg)
        self.min_lon = min_lon
        self.max_lon = self.min_lon + self.tile_cols * self.tile_deg
        self.tile_ctr_lon = np.linspace(self.min_lon + 0.5 * self.tile_deg,
                                        self.max_lon - 0.5 * self.tile_deg,
                                        self.tile_cols)
        self.projections = [[Proj('+proj=aeqd ' +
                                  f'+lat_0={self.tile_ctr_lat[j]} + ' +
                                  f'lon_0={self.tile_ctr_lon[i]} ' +
                                  '+units=m')
                             for i in range(self.tile_cols)]
                            for j in range(self.tile_rows)]

    def get_proj(self, lat, lon):
        '''
        Select the projection nearest to a lat, lon location among a mosaic of
        azimuthal equidistant projections.
        '''
        j = coord_to_grid_ind(lat, self.tile_ctr_lat[0], 0, self.tile_deg,
                              invert=True)
        i = coord_to_grid_ind(lon, self.tile_ctr_lon[0], 0, self.tile_deg)
        # print(np.array(self.projections).shape)
        # print(f'lat {lat}, lon {lon}, j={j}, i={i}')
        return self.projections[j][i]


def coord_to_grid_ind(coord, benchmark_coord, benchmark_ind, resolution,
                      invert=False):
    '''
    General x-to-column, y-to-row function for numpy arrays. Use invert=True
    for row-to-y in north-down orientations.
    '''
    # Calculate the floating point index relative to the benchmark (reference
    # cell center) coordinate.
    if invert:
        ind = (benchmark_coord - coord) / resolution
    else:
        ind = (coord - benchmark_coord) / resolution
    return benchmark_ind + \
        np.floor(ind).astype(int) + (ind - np.floor(ind) + 0.5).astype(int)


def geographiclib_geodesic_m(latlon1, latlon2):
    '''
    Wrapper function for geographiclib.Geodesic.WGS84, takes (lat, lon) tuples
    for two points and returns the distance between them on the WGS84
    ellipsoid, in meters. This is the same functionality as geopy_geodesic_km,
    except for the units, and this one is faster.
    '''
    return Geodesic.WGS84.Inverse(latlon1[0], latlon1[1],
                                  latlon2[0], latlon2[1],
                                  Geodesic.DISTANCE)['s12']


def geopy_geodesic_km(latlon1, latlon2):
    '''
    Wrapper function for geopy.distance.geodesic; takes (lat, lon) tuples for
    two points and returns the distance between them on the WGS84 ellipsoid,
    in kilometers. This is the same functionality as geographiclib_geodesic_m,
    except for the units, and this one is slower.
    '''
    return distance.geodesic(latlon1, latlon2).km


def gisrs_great_circle_km(latlon1, latlon2):
    '''
    Wrapper function for geopy.distance.great_circle; takes (lat, lon) tuples
    for two points and returns the distance between them in kilometers.
    '''
    return distance.great_circle(latlon1, latlon2,
                                 radius=6367.444657).km


def geopy_great_circle_km(latlon1, latlon2):
    '''
    Wrapper function for geopy.distance.great_circle; takes (lat, lon) tuples
    for two points and returns the distance between them in kilometers.
    '''
    return distance.great_circle(latlon1, latlon2,
                                 radius=distance.EARTH_RADIUS).km


def one_to_many_km(target_lat_deg,
                   target_lon_deg,
                   neighborhood_lats_deg,
                   neighborhood_lons_deg,
                   distance_method='geopy_great_circle',
                   num_pool_processes=0,
                   progress=False,
                   return_errors=False,
                   aeqd_mosaic=None):
    '''
    Calculate the distance between (scalar( (target_lat_deg, target_lon_deg)
    and (arrays) neighborhood_lats_deg and neighborhood_lons_deg.
    Options for distance_method:
    geopy_great_circle
    gisrs_great_circle
    geopy_geodesic
    geographiclib_geodesic
    azimuthal_equidistant_mosaic
    '''
    logger = logging.getLogger()

    # In case somebody is crazy enough to call this with scalars... ;)
    if np.isscalar(neighborhood_lats_deg):
        neighborhood_lats_deg = [neighborhood_lats_deg]
    if np.isscalar(neighborhood_lons_deg):
        neighborhood_lons_deg = [neighborhood_lons_deg]
    num_points = len(neighborhood_lats_deg)

    if progress is True:
        tqdm_disable = False
    else:
        tqdm_disable = True

    if return_errors is True:

        # Calculate reference distances using the
        # geographiclib.Geodesic.WGS84.Inverse function. This is obviously
        # expensive.
        ref_dist_km = \
            np.array([Geodesic.WGS84.Inverse(target_lat_deg,
                                             target_lon_deg,
                                             neighborhood_lats_deg[i],
                                             neighborhood_lons_deg[i],
                                             Geodesic.DISTANCE)['s12']
                      for i in tqdm(range(num_points),
                                    disable=tqdm_disable)]) * 1.0e-3

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
            # aeqd = aeqd_mosaic

    if num_pool_processes == 0:

        if distance_method in ('geopy_great_circle', 'gisrs_great_circle'):

            if distance_method == 'gisrs_great_circle':
                radius = 6367.444657
            else:
                radius = distance.EARTH_RADIUS

            dist_km = \
                np.array([distance.great_circle((target_lat_deg,
                                                 target_lon_deg),
                                                (neighborhood_lats_deg[i],
                                                 neighborhood_lons_deg[i]),
                                                radius=radius).km
                          for i in tqdm(range(num_points),
                                        disable=tqdm_disable)])

        elif distance_method == 'geopy_geodesic':

            dist_km = \
                np.array([distance.geodesic((target_lat_deg,
                                             target_lon_deg),
                                            (neighborhood_lats_deg[i],
                                             neighborhood_lons_deg[i])).km
                          for i in tqdm(range(num_points),
                                        disable=tqdm_disable)])


        elif distance_method == 'geographiclib_geodesic':

            dist_km = \
                np.array([Geodesic.WGS84.Inverse(target_lat_deg,
                                                 target_lon_deg,
                                                 neighborhood_lats_deg[i],
                                                 neighborhood_lons_deg[i],
                                                 Geodesic.DISTANCE)['s12']
                          for i in tqdm(range(num_points),
                                        disable=tqdm_disable)]) * 1.0e-3

        elif distance_method == 'azimuthal_equidistant_mosaic':

            proj = aeqd_mosaic.get_proj(target_lat_deg, target_lon_deg)
            target_x, target_y = proj(target_lon_deg, target_lat_deg)
            neighborhood_x, neighborhood_y = proj(neighborhood_lons_deg,
                                                  neighborhood_lats_deg)
            dist_x = neighborhood_x - target_x
            dist_y = neighborhood_y - target_y
            dist_km = np.sqrt(dist_x * dist_x + dist_y * dist_y) * 1.0e-3

        else:

            msg = f'No support for distance_method="{distance_method}"'
            raise NotSupported(msg)

    else:

        chunksize = max([num_points // num_pool_processes // 4, 1])
        # print('chunksize={}'.format(chunksize))

        # target_coords = [(target_lat_deg, target_lon_deg)]
        neighborhood_coords = list(zip(neighborhood_lats_deg,
                                       neighborhood_lons_deg))
        # product = list(itertools.product(target_coords,
        #                                  neighborhood_coords))
        # product = [(ctr, nbr)
        #            for ctr in target_coords for nbr in neighborhood_coords]
        product = [((target_lat_deg, target_lon_deg), neighbor_coords)
                   for neighbor_coords in neighborhood_coords]

        if distance_method == 'gisrs_great_circle':

            with Pool(num_pool_processes) as pool:
                dist_km = pool.starmap(gisrs_great_circle_km,
                                       tqdm(product,
                                            total=num_points,
                                            disable=tqdm_disable),
                                       chunksize=chunksize)
            dist_km = np.array(dist_km)

        elif distance_method == 'geopy_great_circle':

            with Pool(num_pool_processes) as pool:
                dist_km = pool.starmap(geopy_great_circle_km,
                                       tqdm(product,
                                            total=num_points,
                                            disable=tqdm_disable),
                                       chunksize=chunksize)
            dist_km = np.array(dist_km)

        elif distance_method == 'geopy_geodesic':

            with Pool(num_pool_processes) as pool:
                dist_km = pool.starmap(geopy_geodesic_km,
                                       tqdm(product,
                                            total=num_points,
                                            disable=tqdm_disable),
                                       chunksize=chunksize)
            dist_km = np.array(dist_km)

        elif distance_method == 'geographiclib_geodesic':

            with Pool(num_pool_processes) as pool:
                dist_km = pool.starmap(geographiclib_geodesic_m,
                                       tqdm(product,
                                            total=num_points,
                                            disable=tqdm_disable),
                                       chunksize=chunksize)
            dist_km = np.array(dist_km) * 1.0e-3

        else:

            msg = 'No support for ' + \
                  f'distance_method="{distance_method}" ' + \
                  'with num_pool_processes > 0'
            raise NotSupported(msg)

    return dist_km


def one_to_one_km(target_lat_deg,
                  target_lon_deg,
                  neighbor_lat_deg,
                  neighbor_lon_deg,
                  distance_method='geopy_great_circle',
                  return_errors=False,
                  aeqd_mosaic=None):
    '''
    Calculate the distance between two points on the Earth:
    (target_lat_deg, target_lon_deg) and (neighbor_lat_deg, neighbor_lon_deg).
    Options for distance_method:
    geopy_great_circle
    gisrs_great_circle
    geopy_geodesic
    geographiclib_geodesic
    azimuthal_equidistant_mosaic
    '''
    logger = logging.getLogger()

    if return_errors is True:

        # Calculate reference distances using the
        # geographiclib.Geodesic.WGS84.Inverse function.
        ref_dist_km = \
            Geodesic.WGS84.Inverse(target_lat_deg,
                                   target_lon_deg,
                                   neighbor_lat_deg,
                                   neighbor_lon_deg,
                                   Geodesic.DISTANCE)['s12'] * 1.0e-3

    if (aeqd_mosaic is not None) and \
       (distance_method != 'azimuthal_equidistant_mosaic'):
        msg = 'Unnecessary setting of keyword ' + \
              '"aeqd_mosaic" with ' + \
              f'distance_method set to "{distance_method}"'
        logger.warning(msg)
    if (aeqd_mosaic is None) and \
       (distance_method == 'azimuthal_equidistant_mosaic'):
        msg = 'Missing required instance of ' +  \
              'class AzEqdMosaic'
        return None

    if distance_method in ('geopy_great_circle', 'gisrs_great_circle'):

        if distance_method == 'gisrs_great_circle':
            radius = 6367.444657
        else:
            radius = distance.EARTH_RADIUS

        dist_km = \
            distance.great_circle((target_lat_deg, target_lon_deg),
                                  (neighbor_lat_deg, neighbor_lon_deg),
                                  radius=radius).km

    elif distance_method == 'geopy_geodesic':

        dist_km = \
            distance.geodesic((target_lat_deg, target_lon_deg),
                              (neighbor_lat_deg, neighbor_lon_deg)).km

    elif distance_method == 'geographiclib_geodesic':

        dist_km = \
            Geodesic.WGS84.Inverse(target_lat_deg,
                                   target_lon_deg,
                                   neighbor_lat_deg,
                                   neighbor_lon_deg,
                                   Geodesic.DISTANCE)['s12'] * 1.0e-3

    elif distance_method == 'azimuthal_equidistant_mosaic':

        # print(f'target_lat_deg {target_lat_deg}')
        # print(f'target_lon_deg {target_lon_deg}')
        # print(f'neighbor_lat_deg {neighbor_lat_deg}')
        # print(f'neighbor_lon_deg {neighbor_lon_deg}')
        proj = aeqd_mosaic.get_proj(target_lat_deg, target_lon_deg)
        # print('called get_proj')
        # print(proj)
        target_x, target_y = proj(target_lon_deg, target_lat_deg)
        neighborhood_x, neighborhood_y = proj(neighbor_lon_deg,
                                              neighbor_lat_deg)
        dist_x = neighborhood_x - target_x
        dist_y = neighborhood_y - target_y
        dist_km = np.sqrt(dist_x * dist_x + dist_y * dist_y) * 1.0e-3

    else:

        msg = f'No support for distance_method="{distance_method}"'
        raise NotSupported(msg)

    return dist_km


