'''
Functions for managin geographic raster datasets (including GDAL and GRIB).
'''
import sys
# import os
# import errno
# import logging
# import time

import numpy as np
from osgeo import gdal,osr,gdalconst
import cartopy.crs as ccrs
# import matplotlib as mpl
# import matplotlib.colors as mplcol
# from matplotlib.colors import LinearSegmentedColormap
# import matplotlib.pyplot as plt
# from cartopy.feature import NaturalEarthFeature as cfNEF
#import pygrib

# class Error(Exception):
#     '''
#     Base class for homemade exceptions.
#     '''

# class GridInconsistency(Error):
#     '''
#     Exception for inconsistent grid/array geometries.
#     '''


class GeoRasterDS:
    '''
    Class for geographic rasters described as GDAL datasets.
    '''
    def __init__(self,
                 gdal_ds,  # GDAL dataset
                 crs,     # cartopy CRS
                 x_ll_ctr, # x min (center) of raster in CRS (i.e., native)
                 nx,       # x grid size (number of columns)
                 dx,       # x grid spacing
                 y_ll_ctr, # y min (center) of raster in CRS (i.e., native)
                 ny,       # y grid size (number of rows)
                 dy):      # y grid spacing
        self.gdal_ds = gdal_ds
        self.crs = crs
        self.x_ll_ctr = x_ll_ctr
        self.nx = nx
        self.dx = dx
        self.y_ll_ctr = y_ll_ctr
        self.ny = ny
        self.dy = dy


def regrid_ds_to_lon_lat(input_data,
                         min_lon,
                         max_lon,
                         lon_res,
                         num_cols,
                         min_lat,
                         max_lat,
                         lat_res,
                         num_rows,
                         ndv,
                         output_dataset_name,
                         sampling='bilinear'):
    '''
    Regrid a GDAL raster dataset in project coodinates to a lon/lat grid.
    '''

    mem_driver = gdal.GetDriverByName('MEM')
    data_type = input_data.gdal_ds.GetRasterBand(1).DataType
    data_type_name = gdal.GetDataTypeName(data_type)
    lon_lat_ds = mem_driver.Create(output_dataset_name,
                                   xsize=num_cols,
                                   ysize=num_rows,
                                   bands=input_data.gdal_ds.RasterCount,
                                   eType=eval('gdal.GDT_' + data_type_name))

    # Define the projection.
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    lon_lat_ds.SetProjection(srs.ExportToWkt())

    # Define the CRS for the lon/lat data.
    lon_lat_crs = ccrs.PlateCarree()

    # Define the GeoTransform.
    geo_transform = (min_lon, lon_res, 0.0,
                     max_lat, 0.0, -lat_res)
    lon_lat_ds.SetGeoTransform(geo_transform)

    # Pre-fill the raster bands with no-data values.
    for band in range(1, input_data.gdal_ds.RasterCount + 1):
        lon_lat_ds.GetRasterBand(band).Fill(ndv)

    if sampling == 'neighbor':
        sampling_const = gdalconst.GRA_NearestNeighbour
    else:
        sampling_const = gdalconst.GRA_Bilinear

    # Reproject.
    gdal.ReprojectImage(input_data.gdal_ds,
                        lon_lat_ds,
                        input_data.gdal_ds.GetProjection(),
                        lon_lat_ds.GetProjection(),
                        sampling_const)

    ll_ctr_lon = min_lon + 0.5 * lon_res
    ll_ctr_lat = min_lat + 0.5 * lat_res

    return GeoRasterDS(lon_lat_ds,
                       lon_lat_crs,
                       ll_ctr_lon,
                       num_cols,
                       lon_res,
                       ll_ctr_lat,
                       num_rows,
                       lat_res)


def gdalify_lonlat_grid(dataset_name,
                        ll_center_lon_deg,
                        ll_center_lat_deg,
                        lon_resolution,
                        lat_resolution,
                        no_data_value,
                        *args,
                        north_to_south=False,
                        description=None):

    '''
    Convert a latitude/longitude grid into a GDAL dataset,
    '''

    if len(args) < 1:
        print('At least one input grid is required.', file=sys.stderr)
        return None

    num_rows, num_cols = args[0].shape
    data_type = args[0].dtype.name
    for i in range(1, len(args)):
        if args[i].dtype.name != data_type:
            print('ERROR: all raster bands must be the same data type.',
                  file=sys.stderr)
            return None
    gdal_data_type_name = \
        gdal.GetDataTypeName(gdal.GetDataTypeByName(data_type))
    mem_driver = gdal.GetDriverByName('MEM')
    gdal_ds = mem_driver.Create(dataset_name,
                                xsize=num_cols,
                                ysize=num_rows,
                                bands=len(args),
                                eType=eval('gdal.GDT_' + gdal_data_type_name))

    gdal_ds.SetProjection('EPSG:4326')
    # # Generate a Proj4 string from projparams.
    # proj4_str = ''
    # for key in projparams:
    #     val = projparams[key]
    #     proj4_str += '+{}={} '.format(key, val)

    # # Define the projection.
    # srs = osr.SpatialReference()
    # srs.ImportFromProj4(proj4_str)
    # gdal_ds.SetProjection(srs.ExportToWkt())

    ## Define the GeoTransform of the dataset. ##

    # # Define the globe for projected coordinate systems.
    # globe = ccrs.Globe(semimajor_axis=semimajor_axis_meters,
    #                    semiminor_axis=semiminor_axis_meters)

    # Define the coordinate reference system.
    crs = ccrs.PlateCarree()

    # # Calculate the lower left cell center x and y coordinates.
    # #    if grid_type != 'regular_ll':
    # crs_geo = ccrs.Geodetic(globe=globe)
    # x_ll_ctr, y_ll_ctr = crs.transform_point(ll_center_lon_deg,
    #                                          ll_center_lat_deg,
    #                                          crs_geo)

    # Calculate and set the GeoTransform.
    x_min = ll_center_lon_deg - 0.5 * lon_resolution
    y_max = ll_center_lat_deg - 0.5 * lat_resolution \
            + num_rows * lat_resolution
    geo_transform = (x_min, lon_resolution, 0.0, y_max, 0.0, -lat_resolution)
    gdal_ds.SetGeoTransform(geo_transform)

    for band in range(1, len(args) + 1):
        # Add the grid to the dataset.
        if north_to_south:
            # Dataset is already oriented south-up/north-down.
            gdal_ds.GetRasterBand(band).WriteArray(args[band-1])
        else:
            # Dataset must be flipped to south-up/north-down.
            gdal_ds.GetRasterBand(band).WriteArray(np.flipud(args[band-1]))
        gdal_ds.GetRasterBand(band).SetNoDataValue(no_data_value)
        if description is not None:
            gdal_ds.GetRasterBand(band).SetDescription(description[band-1])

    return GeoRasterDS(gdal_ds,
                       crs,
                       ll_center_lon_deg,
                       num_cols,
                       lon_resolution,
                       ll_center_lat_deg,
                       num_rows,
                       lat_resolution)


def gdalify_lonlat_grib_record(dataset_name,
                               projparams,
                               semimajor_axis_meters,
                               semiminor_axis_meters,
                               ll_center_lon_deg,
                               ll_center_lat_deg,
                               x_res,
                               y_res,
                               no_data_value,
                               *args,
                               description=None):

    '''
    Convert GRIB output on a longitude/latitude grid into a GDAL dataset,
    possibly containing multiple "bands" (one for each raster given under
    *args).
    '''

    # NOTE: This is simply a version of gdalify_grib_record that supports an
    # ellipsoidal earth.
    #
    # Positional arguments should match keys and instance variables from
    # GRIB "messages" (I like to call them "records") generated by the
    # pygrib module. In some cases names have been modified:
    #
    #   semimajor_axis_meters is pygrib "scaledValueOfEarthMajorAxis"
    #   semiminor_axis_meters is pygrid "scaledValueOfEarthMinorAxis"
    #   ll_center_lon_deg is pygrib "longitudeOfFirstGridPointInDegrees"
    #   ll_center_lat_deg is pygrib "latitudeOfFirstGridPointInDegrees"
    #   x_res is pygrib "DxInMetres" or "iDirectionIncrement"
    #   y_res is pygrib "DyInMetres" or "jDirectionIncrement'
    #
    # The variable length list *args should provide all the GRIB rasters
    # that will be stored as "bands" in the GDAL dataset.

    ### Set up the dataset. ###

    ## Create a GDAL dataset. ##
    if len(args) < 1:
        print('At least one GRIB raster is required.', file=sys.stderr)
        return None

    num_rows, num_cols = args[0].shape
    data_type = args[0].dtype.name
    for i in range(1, len(args)):
        if args[i].dtype.name != data_type:
            print('ERROR: all raster bands must be the same data type.',
                  file=sys.stderr)
            return None
    gdal_data_type_name = \
        gdal.GetDataTypeName(gdal.GetDataTypeByName(data_type))
    mem_driver = gdal.GetDriverByName('MEM')
    gdal_ds = mem_driver.Create(dataset_name,
                                xsize=num_cols,
                                ysize=num_rows,
                                bands=len(args),
                                eType=eval('gdal.GDT_' + gdal_data_type_name))

    # Generate a Proj4 string from projparams.
    proj4_str = ''
    for key in projparams:
        val = projparams[key]
        proj4_str += f'+{key}={val} '

    # Define the projection.
    srs = osr.SpatialReference()
    srs.ImportFromProj4(proj4_str)
    gdal_ds.SetProjection(srs.ExportToWkt())

    ## Define the GeoTransform of the dataset. ##

    # Define the globe for projected coordinate systems.
    globe = ccrs.Globe(semimajor_axis=semimajor_axis_meters,
                       semiminor_axis=semiminor_axis_meters)

    # Define the coordinate reference system.
    crs = ccrs.PlateCarree()

    # Calculate the lower left cell center x and y coordinates.
    #    if grid_type != 'regular_ll':
    crs_geo = ccrs.Geodetic(globe=globe)
    x_ll_ctr, y_ll_ctr = crs.transform_point(ll_center_lon_deg,
                                             ll_center_lat_deg,
                                             crs_geo)

    # Calculate and set the GeoTransform.
    x_min = x_ll_ctr - 0.5 * x_res
    y_max = y_ll_ctr - 0.5 * y_res + num_rows * y_res
    geo_transform = (x_min, x_res, 0.0, y_max, 0.0, -y_res)
    gdal_ds.SetGeoTransform(geo_transform)

    for band in range(1, len(args) + 1):
        # Add the grid to the dataset.
        gdal_ds.GetRasterBand(band).WriteArray(np.flipud(args[band-1]))
        gdal_ds.GetRasterBand(band).SetNoDataValue(no_data_value)
        if description is not None:
            gdal_ds.GetRasterBand(band).SetDescription(description[band-1])

    return GeoRasterDS(gdal_ds,
                       crs,
                       x_ll_ctr,
                       num_cols,
                       x_res,
                       y_ll_ctr,
                       num_rows,
                       y_res)


def gdalify_grib_record(dataset_name,
                        projparams,
                        radius_meters,
                        grid_type,
                        ll_center_lon_deg,
                        ll_center_lat_deg,
                        x_res,
                        y_res,
                        no_data_value,
                        *args,
                        description=None):

    '''
    Convert GRIB output into a GDAL dataset, possibly containing multiple
    "bands" (one for each raster given under *args).
    '''

    # NOTE: Only spherical earth models are supported.
    #
    # Positional arguments should match keys and instance variables from
    # GRIB "messages" (I like to call them "records") generated by the
    # pygrib module. In some cases names have been modified:
    #
    #   radius_meters is pygrib "radius"
    #   ll_center_lon_deg is pygrib "longitudeOfFirstGridPointInDegrees"
    #   ll_center_lat_deg is pygrib "latitudeOfFirstGridPointInDegrees"
    #   x_res is pygrib "DxInMetres" or "iDirectionIncrement"
    #   y_res is pygrib "DyInMetres" or "jDirectionIncrement'
    #
    # The variable length list *args should provide all the GRIB rasters
    # that will be stored as "bands" in the GDAL dataset.

    ### Set up the dataset. ###

    ## Create a GDAL dataset. ##
    if len(args) < 1:
        print('At least one GRIB raster is required.', file=sys.stderr)
        return None

    num_rows, num_cols = args[0].shape
    data_type = args[0].dtype.name
    for i in range(1, len(args)):
        if args[i].dtype.name != data_type:
            print('ERROR: all raster bands must be the same data type.',
                  file=sys.stderr)
            return None
    gdal_data_type_name = \
        gdal.GetDataTypeName(gdal.GetDataTypeByName(data_type))
    mem_driver = gdal.GetDriverByName('MEM')
    gdal_ds = mem_driver.Create(dataset_name,
                                xsize=num_cols,
                                ysize=num_rows,
                                bands=len(args),
                                eType=eval('gdal.GDT_' + gdal_data_type_name))

    # Generate a Proj4 string from projparams.
    proj4_str = ''
    for key in projparams:
        val = projparams[key]
        proj4_str += f'+{key}={val} '

    # Define the projection.
    srs = osr.SpatialReference()
    srs.ImportFromProj4(proj4_str)
    gdal_ds.SetProjection(srs.ExportToWkt())
    # print(proj4_str)
    # print(srs.ExportToWkt())

    ## Define the GeoTransform of the dataset. ##

    # Define the globe for projected coordinate systems.
    globe = ccrs.Globe(semimajor_axis=radius_meters,
                       semiminor_axis=radius_meters,
                       ellipse=None)

    # Define the coordinate reference system.
    if grid_type == 'lambert':
        crs = ccrs.LambertConformal(globe=globe,
                                    central_longitude=\
                                    projparams['lon_0'],
                                    central_latitude=\
                                    projparams['lat_0'],
                                    standard_parallels=\
                                    (projparams['lat_1'],
                                     projparams['lat_2']))
    elif grid_type == 'polar_stereographic':
        crs = ccrs.Stereographic(globe=globe,
                                 central_longitude=\
                                 projparams['lon_0'],
                                 central_latitude=\
                                 projparams['lat_0'],
                                 true_scale_latitude=\
                                 projparams['lat_ts'])
    elif grid_type == 'regular_ll':
        crs = ccrs.PlateCarree()
    else:
        print(f'Unsupported grid_type "{grid_type}"')
        return None

    # Calculate the lower left cell center x and y coordinates.
    #    if grid_type != 'regular_ll':
    crs_geo = ccrs.Geodetic(globe=globe)
    x_ll_ctr, y_ll_ctr = crs.transform_point(ll_center_lon_deg,
                                             ll_center_lat_deg,
                                             crs_geo)

    # Calculate and set the GeoTransform.
    x_min = x_ll_ctr - 0.5 * x_res
    y_max = y_ll_ctr - 0.5 * y_res + num_rows * y_res
    geo_transform = (x_min, x_res, 0.0, y_max, 0.0, -y_res)
    gdal_ds.SetGeoTransform(geo_transform)

    for band in range(1, len(args) + 1):
        # Add the grid to the dataset.
        gdal_ds.GetRasterBand(band).WriteArray(np.flipud(args[band-1]))
        gdal_ds.GetRasterBand(band).SetNoDataValue(no_data_value)
        if description is not None:
            gdal_ds.GetRasterBand(band).SetDescription(description[band-1])

    return GeoRasterDS(gdal_ds,
                       crs,
                       x_ll_ctr,
                       num_cols,
                       x_res,
                       y_ll_ctr,
                       num_rows,
                       y_res)
