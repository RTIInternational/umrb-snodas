'''
Functions for displaying geographic raster data.
'''
# import sys
# import os
# import errno
import logging
# import time

import numpy as np
from osgeo import gdal,osr
import cartopy.crs as ccrs
import matplotlib as mpl
# import matplotlib.colors as mplcol
# from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from cartopy.feature import NaturalEarthFeature as cfNEF
from cartopy.feature import STATES
# import pygrib


class Error(Exception):
    '''
    Base class for homemade exceptions.
    '''


class GridInconsistency(Error):
    '''
    Exception for inconsistent grid/array geometries.
    '''


def geo_grid_map(crs,
                 figure_x_size,
                 figure_y_size,
                 num,
                 bbox,
                 x_ctr_data,
                 y_ctr_data,
                 dataset,
                 raster_band_index,
                 ndv,
                 title,
                 color_ramp,
                 color_ramp_label,
                 dpi=None,
                 cbar_fraction=0.1,
                 cbar_pad=0.05,
                 cbar_shrink=0.7,
                 gridspec_kw=None):
    '''
    Generate a geographic map of single band from a GDAL raster dataset.

    crs: The cartopy.crs object describing the coordinate system.
    figure_x_size: The figure x size in inches.
    figure_y_size: The figure y size in inches.
    num: The num argument that will be provided to matplotlib.pyplot.subplots
         (and thereby provided to matplotlib.pyplot.figure) identifying the
         figure.
    bbox: The bounding box of the raster, in the coordinate system of the
          data.
    x_ctr_data: The list of x axis locations for the data (i.e. the center
                coordinates of each column).
    y_ctr_data: The list of y axis locations for the data (i.e. the center
                coordinates of each row). This should apply to the data in a
                north-up orientation, even though the grids in the GDAL
                dataset are generally oriented north-down.
    dataset: The GDAL raster dataset.
    raster_band_index: The raster band to show, indexed from 1.
    ndv: The no-data value.
    title: A string for the plot title.
    color_ramp: a dictionary describing the color ramp to use for the data:
                {'colormap': matplotlib.colors colormap object
                 'col_levels': colorbar axis levels bounding each color
                 'tick_levels': colorbar tick levels
                 'tick_labels': colorbar tick labels
                 'norm': colormap index from a call to
                         matplotlib.colors.BoundaryNorm
                 'extend': 'min', 'max', 'both', or 'neither'}
    color_ramp_label: Typically the units of the values on the colorbar
                      axis.
    '''
    fig, geo_axis = plt.subplots(subplot_kw=dict(projection=crs),
                                 figsize=(figure_x_size, figure_y_size),
                                 num=num,
                                 clear=True,
                                 dpi=dpi)

    geo_axis.set_extent(bbox, crs=crs)

    # If uncommented, tight_layout overrides rcParams elements (namely left,
    # bottom, right and top margin values) for a figure.
    #fig.tight_layout()

    # Extract the requested raster from the dataset.
    grid = dataset.GetRasterBand(raster_band_index).ReadAsArray()

    # Convert the grid to a masked array.
    grid_masked = np.ma.masked_equal(grid, ndv)

    # Draw the grid.
    # color_mesh = geo_axis.contourf(x_ctr_data, y_ctr_data,
    #                  np.flipud(grid_masked),
    #                  color_ramp['col_levels'],
    #                  cmap=color_ramp['colormap'],
    #                  norm=color_ramp['norm'],
    #                  extend=color_ramp['extend'],
    #                  transform=crs)
    # This needs shading='nearest', 'auto', or 'gouraud'.

    color_mesh = geo_axis.pcolormesh(x_ctr_data, y_ctr_data,
                                     np.flipud(grid_masked),
                                     cmap=color_ramp['colormap'],
                                     norm=color_ramp['norm'],
                                     transform=crs,
                                     shading='auto')
    print(color_mesh.__dict__)

    geo_axis.set_title(title)

    cbar = None
    if color_ramp['tick_levels'] is not None and \
       color_ramp['tick_labels'] is not None:
        cbar = fig.colorbar(color_mesh,
                            orientation='vertical',
                            fraction=cbar_fraction,
                            shrink=cbar_shrink,
                            pad=cbar_pad,
                            extend=color_ramp['extend'])
        cbar.set_ticks(color_ramp['tick_levels'])
        cbar.set_ticklabels(color_ramp['tick_labels'])
        cbar.ax.yaxis.set_tick_params(length=4)
        cbar.set_label(color_ramp_label,
                       fontsize=mpl.rcParams['axes.labelsize'],
                       color=mpl.rcParams['ytick.color'])

    # Draw U.S. states and national boundaries.
    # *** NEITHER OF THESE WORK WITH CARTOPY 0.13 / PYTHON 3.6 ***
    # ax.add_feature(cfNEF(category='cultural',
    #                      name='admin_1_states_provinces_lakes',
    #                      scale='50m',
    #                      edgecolor='gray',
    #                      facecolor='none',
    #                      linewidth=0.4))
    # ax.add_feature(cfNEF(category='cultural',
    #                      name='admin_0_countries_lakes',
    #                      scale='50m',
    #                      edgecolor='black',
    #                      facecolor='none',
    #                      linewidth=0.4))

    return(fig, geo_axis, cbar)


def mpl_map(north_down_grid,
            lat_axis,
            lat_resolution,
            lon_axis,
            lon_resolution,
            dataset_name,
            color_ramp,
            colorbar_units,
            title,
            map_lines_color='red',
            facecolor='#4e4e4e',
            fig_num=1):
    '''
    Generate a map of a lon/lat grid using GDAL, matplotlib, and cartopy.
    This is a high-level way of accessing geo_grid_map without having to do as
    much heavy lifting as that function requires.
    '''

    logger = logging.getLogger()

    num_rows, num_cols = north_down_grid.shape
    if len(lat_axis) != num_rows:
        raise GridInconsistency
    if len(lon_axis) != num_cols:
        raise GridInconsistency

    # The grid should ideally be a masked array. Here all masked values are
    # set to np.nan. This way, geo_grid_map can be handed a no-data value.
    if np.ma.isMaskedArray(north_down_grid):
        masked_ind = np.where(np.ma.getmaskarray(north_down_grid) == True)
        north_down_grid[masked_ind] = np.nan
        north_down_grid = np.ma.masked_invalid(north_down_grid)

    # Create a GDAL dataset.
    mem_driver = gdal.GetDriverByName('MEM')
    np_dtype_name = north_down_grid.dtype.name
    np_data_type_num = gdal.GetDataTypeByName(np_dtype_name)
    gdal_data_type_name = gdal.GetDataTypeName(np_data_type_num)
    # TODO: replace eval('gdal.GDT_' + gdal_data_type_name) with
    #               getattr(gdal, f'GDT_{gdal_data_type_name}')
    idw_ds = mem_driver.Create(dataset_name,
                               xsize=num_cols,
                               ysize=num_rows,
                               bands=1,
                               eType=eval('gdal.GDT_' + gdal_data_type_name))

    # Define the "projection" (in quotes because this is geographic data).
    bbox = [lon_axis[0] - 0.5 * lon_resolution,
            lon_axis[-1] + 0.5 * lon_resolution,
            lat_axis[-1] - 0.5 * lat_resolution,
            lat_axis[0] + 0.5 * lat_resolution]
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    idw_ds.SetProjection(srs.ExportToWkt())
    idw_ds.SetGeoTransform((bbox[0],
                            lon_resolution,
                            0.0,
                            bbox[3],
                            0.0,
                            -lat_resolution))

    # Write the grid to the dataset.
    idw_ds.GetRasterBand(1).WriteArray(np.flipud(north_down_grid))

    # Define variables needed for plotting.
    crs = ccrs.PlateCarree()

    # Plot style/customization.
    plt.style.use('seaborn-notebook')
    # Default margins for subplots.

    debug_graphics = True

    # Set up the figure.

    left_margin = 0.05
    bottom_margin = 0.05
    right_margin = 0.95
    top_margin = 0.95

    mpl.rcParams['figure.subplot.left'] = left_margin
    mpl.rcParams['figure.subplot.bottom'] = bottom_margin
    mpl.rcParams['figure.subplot.right'] = right_margin
    mpl.rcParams['figure.subplot.top'] = top_margin

    # To estimate the figure size, note that the title and info text will lie
    # outside the bounds of the figure, in the margins.

    # Start from the premise of controlling the size of the MAP portion of the
    # graphic, rather than the size of the full image.
    aspect = float(num_cols) / float(num_rows)

    if aspect > 1.0:
        # wide map
        x_map_size = 8.0
        y_map_size = x_map_size / aspect
    else:
        y_map_size = 8.0
        x_map_size = y_map_size * aspect

    cbar_pad_inches = 0.25
    cbar_pad = cbar_pad_inches / x_map_size
    message = f'cbar_pad = {cbar_pad}'
    logger.info(message)
    cbar_width_inches = 1.0
    cbar_fraction = cbar_width_inches / x_map_size
    message = f'cbar_fraction = {cbar_fraction}'
    logger.info(message)
    title_height_guess_inches = 0.33
    info_height_guess_inches = 0.33

    # Using "fig" here to include everything inside the figure.subplot
    # margins.
    x_fig_size = x_map_size + cbar_pad_inches + cbar_width_inches
    y_fig_size = y_map_size \
        + title_height_guess_inches + info_height_guess_inches

    x_size = x_fig_size / (right_margin - left_margin)
    y_size = y_fig_size / (top_margin - bottom_margin)

    if debug_graphics:
        print(f'x_size is {x_size}, y_size is {y_size}')

    fig, geo_axes, cbar = \
        geo_grid_map(crs,
                     x_size,
                     y_size,
                     fig_num,
                     bbox,
                     lon_axis,
                     lat_axis,
                     idw_ds,
                     1,
                     np.nan,
                     title,
                     color_ramp,
                     colorbar_units,
                     cbar_pad=cbar_pad,
                     cbar_fraction=cbar_fraction,
                     cbar_shrink=0.7)
    geo_axes.set_facecolor(facecolor)

    # See
    # https://scitools.org.uk/cartopy/docs/v0.19/matplotlib/feature_interface.html
    geo_axes.coastlines(resolution='10m', linewidth=0.4)

    # Change the scale below to "50m" to get fewer details at the expense of
    # resolution for state boundaries, etc.
    geo_axes.add_feature(cfNEF(category='cultural',
                               name='admin_1_states_provinces_lakes',
                               scale='10m',
                               edgecolor=map_lines_color,
                               facecolor='none',
                               linewidth=0.4))

    return fig, geo_axes, cbar


def mpl_map_2023_old(north_down_grid,
                     lat_axis,
                     lat_resolution,
                     lon_axis,
                     lon_resolution,
                     dataset_name,
                     color_ramp,
                     colorbar_units,
                     title,
                     map_lines_color='red',
                     facecolor='#4e4e4e',
                     target_fig_dim_in=10.0,
                     fig_margin_l_in=0.25,
                     fig_margin_b_in=0.25,
                     fig_margin_r_in=0.25,
                     fig_margin_t_in=0.25,
                     cbar_scale=1.0,
                     cbar_gap_in=0.25,
                     map_title_h_in=0.5,
                     fig_dpi=108,
                     fig_num=1,
                     debug_graphics=False):
    '''
    Generate a map of a lon/lat grid using GDAL, matplotlib, and cartopy.
    This is an attempt to establish more control over the layout of the map
    and colorbar than the original mpl_map and geo_grid_map provide.
    '''

    logger = logging.getLogger()

    num_rows, num_cols = north_down_grid.shape
    if len(lat_axis) != num_rows:
        raise GridInconsistency('The latitude axis is inconsistent with ' +
                                'the data array rows.')
    if len(lon_axis) != num_cols:
        raise GridInconsistency('The longitude axis is inconsistent with ' +
                                'the data array columns.')

    # The grid should ideally be a masked array. Here all masked values are
    # set to np.nan. This way, geo_grid_map can be handed a no-data value.
    if np.ma.isMaskedArray(north_down_grid):
        # Use a copy to protect the contents of the original array.
        north_down_grid = np.ma.copy(north_down_grid)
        masked_ind = np.where(np.ma.getmaskarray(north_down_grid) == True)
        north_down_grid[masked_ind] = np.nan
        north_down_grid = np.ma.masked_invalid(north_down_grid)

    # Create a GDAL dataset.
    mem_driver = gdal.GetDriverByName('MEM')
    np_dtype_name = north_down_grid.dtype.name
    np_data_type_num = gdal.GetDataTypeByName(np_dtype_name)
    gdal_data_type_name = gdal.GetDataTypeName(np_data_type_num)
    idw_ds = mem_driver.Create(dataset_name,
                               xsize=num_cols,
                               ysize=num_rows,
                               bands=1,
                               eType=eval('gdal.GDT_' + gdal_data_type_name))

    # Define the "projection" (in quotes because this is geographic data).
    bbox = [lon_axis[0] - 0.5 * lon_resolution,
            lon_axis[-1] + 0.5 * lon_resolution,
            lat_axis[-1] - 0.5 * lat_resolution,
            lat_axis[0] + 0.5 * lat_resolution]
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    idw_ds.SetProjection(srs.ExportToWkt())
    idw_ds.SetGeoTransform((bbox[0],
                            lon_resolution,
                            0.0,
                            bbox[3],
                            0.0,
                            -lat_resolution))

    # Write the grid to the dataset.
    idw_ds.GetRasterBand(1).WriteArray(np.flipud(north_down_grid))

    # Define variables needed for plotting.
    crs = ccrs.PlateCarree()

    # Plot style/customization.
    # plt.style.use('seaborn-notebook')
    # Default margins for subplots.

    # Set up the figure.

    # left_margin = 0.05
    # bottom_margin = 0.05
    # right_margin = 0.95
    # top_margin = 0.95

    # mpl.rcParams['figure.subplot.left'] = left_margin
    # mpl.rcParams['figure.subplot.bottom'] = bottom_margin
    # mpl.rcParams['figure.subplot.right'] = right_margin
    # mpl.rcParams['figure.subplot.top'] = top_margin

    map_aspect = float(num_cols) / float(num_rows)
    if debug_graphics:
        msg = f'Map aspect is {map_aspect}'
        logger.info(msg)

    # Figure the colorbar is about 1:5 in aspect.
    target_cbar_w_in = 0.1 * cbar_scale * target_fig_dim_in
    target_cbar_h_in = 0.5 * cbar_scale * target_fig_dim_in

    # Calculate a guess at the figure dimensions. Since we cannot predict how
    # the colorbar will render, this guess is unlikely to match the result
    # perfectly, so the figure will be adjusted after the colorbar has been
    # rendered.
    map_scale_in = 0.0
    fig_w_in = 0.0
    fig_h_in = 0.0
    while (fig_w_in < target_fig_dim_in) and (fig_h_in < target_fig_dim_in):
        map_scale_in += 0.001
        map_w_in = map_scale_in * map_aspect
        map_h_in = map_w_in / map_aspect
        fig_w_in = fig_margin_l_in + map_w_in + cbar_gap_in \
            + target_cbar_w_in + fig_margin_r_in
        content_h_in = max([map_h_in + map_title_h_in, target_cbar_h_in])
        fig_h_in = fig_margin_b_in + content_h_in + fig_margin_t_in

    if debug_graphics:
        msg = 'Requested figure dimensions are ' + \
              f'{fig_w_in:.3f} x {fig_h_in:.3f} inches.'
        logger.info(msg)
        msg = 'Requested map dimensions are ' + \
              f'{map_w_in:.3f} x {map_h_in:.3f} inches.'
        logger.info(msg)

    # Locate the map and legend in normalized (figure) coordinates.
    map_fig_bbox = [fig_margin_l_in / fig_w_in,
                    fig_margin_b_in / fig_h_in,
                    (fig_w_in - fig_margin_r_in) / fig_w_in,
                    (fig_h_in - fig_margin_t_in) / fig_h_in]

    # if aspect > 1.0:
    #     # wide map
    #     x_map_size = 8.0
    #     y_map_size = x_map_size / aspect
    # else:
    #     y_map_size = 8.0
    #     x_map_size = y_map_size * aspect

    # cbar_pad_inches = 0.25
    # cbar_pad = cbar_pad_inches / x_map_size
    # message = 'cbar_pad = {}'.format(cbar_pad)
    # logger.info(message)
    # cbar_width_inches = 1.0
    # cbar_fraction = cbar_width_inches / x_map_size
    # message = 'cbar_fraction = {}'.format(cbar_fraction)
    # logger.info(message)
    # title_height_guess_inches = 0.33
    # info_height_guess_inches = 0.33

    cbar_pad = cbar_gap_in / map_w_in
    cbar_fraction = target_cbar_w_in / map_w_in * cbar_scale
    cbar_shrink = 1.0 * cbar_scale

    # Using "fig" here to include everything inside the figure.subplot
    # margins.
    # x_fig_size = x_map_size + cbar_pad_inches + cbar_width_inches
    # y_fig_size = y_map_size \
    #     + title_height_guess_inches + info_height_guess_inches

    # x_size = x_fig_size / (right_margin - left_margin)
    # y_size = y_fig_size / (top_margin - bottom_margin)

    # if debug_graphics:
    #     print('x_size is {}, y_size is {}'.format(x_size, y_size))

    #wspace = cbar_gap_in / (0.5 * (map_w_in + target_cbar_w_in))
    #wspace = cbar_gap_in / fig_w_in
    fig, geo_axes, cbar = \
        geo_grid_map(crs,
                     fig_w_in,
                     fig_h_in,
                     fig_num,
                     bbox,
                     lon_axis,
                     lat_axis,
                     idw_ds,
                     1,
                     np.nan,
                     title,
                     color_ramp,
                     colorbar_units,
                     dpi=fig_dpi,
                     cbar_pad=cbar_pad,
                     cbar_fraction=cbar_fraction,
                     cbar_shrink=cbar_shrink,
                     gridspec_kw={'left': map_fig_bbox[0],
                                  'bottom': map_fig_bbox[1],
                                  'right': map_fig_bbox[2],
                                  'top': map_fig_bbox[3],
                                  'width_ratios': [map_w_in,
                                                   target_cbar_w_in]
                                  },
                     )
    geo_axes.set_facecolor(facecolor)

    # See
    # https://scitools.org.uk/cartopy/docs/v0.19/matplotlib/feature_interface.html
    geo_axes.coastlines(resolution='10m', linewidth=0.4)

    # Change the scale below to "50m" to get fewer details at the expense of
    # resolution for state boundaries, etc.
    geo_axes.add_feature(cfNEF(category='cultural',
                               name='admin_1_states_provinces_lakes',
                               scale='10m',
                               edgecolor=map_lines_color,
                               facecolor='none',
                               linewidth=0.4))

    # The axes outlines produced by ax_draw_outline include labels and
    # titles.
    # if debug_graphics:
    #     ax_draw_outline(fig, geo_axes, edgecolor='green')
    #     ax_draw_outline(fig, cbar.ax, edgecolor='red')

    # Determine the geo_axes position. The dimensions produced here DO NOT
    # include labels and titles (unlike ax_draw_outline).
    _, _, geo_ax_w, geo_ax_h = geo_axes.get_position().bounds
    # geo_ax_aspect = geo_ax_w / geo_ax_h
    # if debug_graphics:
    #     fig.add_artist(plt.Rectangle((geo_ax_x0, geo_ax_y0),
    #                                  geo_ax_w, geo_ax_h,
    #                                  linewidth=4,
    #                                  color='blue',
    #                                  alpha=0.1))

    # Determine the cbar.ax position. The dimensions produced here DO NOT
    # include labels and titles (unlike ax_draw_outline).
    _, _, cbar_ax_w, cbar_ax_h = cbar.ax.get_position().bounds
    # cbar_ax_aspect = cbar_ax_w / cbar_ax_h
    # if debug_graphics:
    #     fig.add_artist(plt.Rectangle((cbar_ax_x0, cbar_ax_y0),
    #                                  cbar_ax_w, cbar_ax_h,
    #                                  linewidth=4,
    #                                  color='orange',
    #                                  alpha=0.1))

    # Determine the figure dimensions as drawn.
    fig_width_in, fig_height_in = fig.get_size_inches()
    if debug_graphics:
        logger.info(f'Actual figure dimensions are {fig_width_in:.3f} x ' +
                    f'{fig_height_in:.3f} inches.')

    # Calculate the actual map dimensions (not including its title).
    map_w_in = geo_ax_w * fig_width_in
    map_h_in = geo_ax_h * fig_height_in
    if debug_graphics:
        logger.info(f'Actual map dimensions are {geo_ax_w * fig_width_in} x ' +
                    f'{geo_ax_h * fig_height_in}')

    # Get the dimensions of the map + title using the same methods as in
    # ax_draw_outline.
    geo_ax_bbox = geo_axes.get_tightbbox(fig.canvas.get_renderer())
    x0, y0, width, height = \
        geo_ax_bbox.transformed(fig.transFigure.inverted()).bounds
    if debug_graphics:
        msg = 'Map + title axis outline rectangle: ' + \
              f'{x0}, {y0}, {width}, {height}'
        logger.info(msg)
    map_plus_title_h_in = height * fig_height_in
    if debug_graphics:
        msg = f'Map + title height in inches: {map_plus_title_h_in}'
        logger.info(msg)

    map_title_h_in = map_plus_title_h_in - map_h_in
    if debug_graphics:
        msg = f'Title height: {map_title_h_in:.3f} inches'
        # logger.info(msg)
    # Get the dimensions of the colorbar and its labels.
    cbar_ax_bbox = cbar.ax.get_tightbbox(fig.canvas.get_renderer())
    x0, y0, width, height = \
        cbar_ax_bbox.transformed(fig.transFigure.inverted()).bounds
    cbar_w_in = width * fig_width_in
    cbar_h_in = height * fig_height_in
    if debug_graphics:
        msg = f'Target colorbar dimensions: {target_cbar_w_in} x ' + \
              f'{target_cbar_h_in} inches'
        logger.info(msg)
        msg = f'Actual colorbar dimensions: {cbar_w_in} x {cbar_h_in} inches'
        logger.info(msg)

    # Calculate the width and height of the current figure based on each
    # element. These will probably not match the actual figure dimensions, but
    # are meant to establish how much of the allocated space is actually being
    # used by the current figure elements.
    fig_w_used_in = fig_margin_l_in + map_w_in + cbar_gap_in \
        + cbar_w_in + fig_margin_r_in
    fig_h_used_in = fig_margin_b_in + map_plus_title_h_in + fig_margin_t_in

    fig_dim_in = max([fig_w_used_in, fig_h_used_in])
    if fig_dim_in != target_fig_dim_in and debug_graphics:
        msg = f'Existing elements only use {fig_dim_in:.3f} of ' + \
              f'requested {target_fig_dim_in} inches ' + \
              'of figure dimension'
        logger.info(msg)

    # Iteratively resize the map until the target_fig_dim_in is reached.
    # Directly expanding/contracting the geo_axes does not quite get it.
    new_map_w_in = map_w_in
    new_map_h_in = map_h_in
    new_fig_w_in = fig_w_in
    new_fig_h_in = fig_h_in
    new_fig_dim_in = fig_dim_in
    while new_fig_dim_in != target_fig_dim_in:

        # Tweak the figure size to match target_fig_dim_in
        fig_size_ratio = target_fig_dim_in / max([new_fig_w_in, new_fig_h_in])

        # Figure out how much space is available for the map (adjusted by
        # fig_size_ratio) with the existing colorbar.
        map_w_potential_in = new_fig_w_in * fig_size_ratio \
            - fig_margin_l_in - fig_margin_r_in - cbar_gap_in - cbar_w_in
        map_h_potential_in = new_fig_h_in * fig_size_ratio \
            - fig_margin_b_in - fig_margin_t_in - map_title_h_in

        map_ratio = min([map_w_potential_in / new_map_w_in,
                         map_h_potential_in / new_map_h_in])

        # Adjust the map_ratio to get to the target_fig_dim_in.
        if debug_graphics:
            msg = f'map_ratio: {map_ratio}'
            logger.info(msg)
        new_map_w_in = new_map_w_in * map_ratio
        new_map_h_in = new_map_h_in * map_ratio

        new_fig_w_in = fig_margin_l_in + new_map_w_in \
            + cbar_gap_in + cbar_w_in + fig_margin_r_in
        new_fig_h_in = fig_margin_b_in + new_map_h_in \
            + map_title_h_in + fig_margin_t_in
        if cbar_h_in > new_fig_h_in:
            # The figure height is not enough to accommodate the
            # colorbar. Either the colorbar needs to get smaller or the map
            # needs to get bigger! Either way, a break is needed to avoid
            # infinite loops.
            # cbar_ratio = new_fig_h_in / cbar_h_in
            # cbar_h_in = cbar_h_in * cbar_ratio
            # cbar_w_in = cbar_w_in * cbar_ratio
            map_ratio = \
                (cbar_h_in - fig_margin_b_in - fig_margin_t_in) / new_map_h_in
            new_map_h_in = new_map_h_in * map_ratio
            new_map_w_in = new_map_w_in * map_ratio
            new_fig_w_in = fig_margin_l_in + new_map_w_in \
                + cbar_gap_in + cbar_w_in + fig_margin_r_in
            new_fig_h_in = fig_margin_b_in + new_map_h_in \
                + map_title_h_in + fig_margin_t_in
            if debug_graphics:
                msg = 'Colorbar is too tall; ' + \
                      f'expanding map by {map_ratio}, ' + \
                      'will accept resulting figure dimensions.'
                logger.info(msg)
            break

        new_fig_dim_in = max([new_fig_w_in, new_fig_h_in])

    if debug_graphics:
        msg = f'New map dimensions are {new_map_w_in} x {new_map_h_in}'
        logger.info(msg)
        msg = 'New figure dimensions are ' + \
              f'{new_fig_w_in:.3f} x {new_fig_h_in:.3f} inches'
        logger.info(msg)

    # Resize the figure before adjusting axis positions.
    fig.set_size_inches(new_fig_w_in, new_fig_h_in)

    # Reposition the colorbar.
    cbar_ax_pos = [(new_fig_w_in - fig_margin_r_in - cbar_w_in) / new_fig_w_in,
                   0.5 * (new_fig_h_in - cbar_h_in) / new_fig_h_in,
                   cbar_w_in / new_fig_w_in,
                   cbar_h_in / new_fig_h_in]
    cbar.ax.set_position(cbar_ax_pos)
    # print(f'final colorbar position: {cbar_ax_pos}')

    # Reposition the geo_axes.
    new_map_plus_title_h_in = new_map_h_in + map_title_h_in
    geo_ax_pos = [fig_margin_l_in / new_fig_w_in,
                  0.5 * (new_fig_h_in - new_map_plus_title_h_in) / new_fig_h_in,
                  new_map_w_in / new_fig_w_in,
                  new_map_h_in / new_fig_h_in]
    geo_axes.set_position(geo_ax_pos)
    # print(f'final geoaxes position: {geo_ax_pos}')

    return fig, geo_axes, cbar


# def format_ticks_and_cursor(x, pos):
#     if x is np.ma.masked:
#         x_str = str(x)
#     else:
#         x_str = f'{x:.2f}'
#         while x_str[-1] == '0':
#             x_str = x_str[:-1]
#         if x_str[-1] == '.':
#             x_str = x_str[:-1]
#         # print(f'{x}' + ' = ' + x_str)
#     return(x_str)


def mpl_map_2023(north_down_grid,
                 lat_axis,
                 lat_resolution,
                 lon_axis,
                 lon_resolution,
                 dataset_name,
                 color_ramp,
                 colorbar_units,
                 title,
                 map_lines_color='red',
                 map_lines_width=0.4,
                 facecolor='#4e4e4e',
                 target_fig_dim_in=10.0,
                 fig_margin_l_in=0.25,
                 fig_margin_b_in=0.25,
                 fig_margin_r_in=0.25,
                 fig_margin_t_in=0.25,
                 cbar_scale=1.0,
                 cbar_gap_in=0.25,
                 map_title_h_in=0.5,
                 fig_dpi=108,
                 fig_num=1,
                 debug_graphics=False):
                 # auto_format_z=False):
    '''
    Generate a map of a lon/lat grid using GDAL, matplotlib, and cartopy.
    This is an attempt to create a version of mpl_map_2023 that works on
    the cloud and ssh+X11 arrangements where the GUI created by
    mpl_map_2023_old has unresolved performance and workability problems,
    apparently caused by the call to pcolormesh in geo_grid_map. This version
    uses imshow instead.
    '''

    logger = logging.getLogger()

    num_rows, num_cols = north_down_grid.shape
    if len(lat_axis) != num_rows:
        raise GridInconsistency
    if len(lon_axis) != num_cols:
        raise GridInconsistency

    if not np.ma.isMaskedArray(north_down_grid):
        logger.error('Input grid must be a NumPy MaskedArray')
        return None, None, None
    # TODO: possibly convert masked values to np.nan the way it is done by
    # mpl_map_2023.

    # Define the "projection" (in quotes because this is geographic data).
    # TODO: possibly switch to a mode where the data have a projection/CRS
    # and the display can have a different one.
    bbox = [lon_axis[0] - 0.5 * lon_resolution,
            lon_axis[-1] + 0.5 * lon_resolution,
            lat_axis[-1] - 0.5 * lat_resolution,
            lat_axis[0] + 0.5 * lat_resolution]
    crs = ccrs.PlateCarree()

    # Determine figure dimensions.
    map_aspect = float(num_cols) / float(num_rows)

    # The colorbar aspect will be 1:5 by default.
    target_cbar_w_in = 0.1 * target_fig_dim_in
    target_cbar_h_in = 0.5 * target_fig_dim_in * cbar_scale

    # Calculate a guess at the figure dimensions. Since we cannot predict
    # exactly how the title and colorbar will render, this guess is unlikely
    # to match the result perfectly, so the figure will be adjusted after the
    # colorbar has been rendered.
    map_scale_in = 0.0
    fig_w_in = 0.0
    fig_h_in = 0.0
    while (fig_w_in < target_fig_dim_in) and (fig_h_in < target_fig_dim_in):
        map_scale_in += 0.001
        map_w_in_expected = map_scale_in * map_aspect
        map_h_in_expected = map_w_in_expected / map_aspect
        fig_w_in = fig_margin_l_in + map_w_in_expected + cbar_gap_in \
            + target_cbar_w_in + fig_margin_r_in
        # Do not allow the colorbar to be taller than the map.
        target_cbar_h_in_new = min([map_h_in_expected, target_cbar_h_in])
        axes_h_in = max([map_h_in_expected, target_cbar_h_in_new])
        content_h_in = axes_h_in + map_title_h_in
        # content_h_in = max([map_h_in_expected + map_title_h_in, target_cbar_h_in])
        fig_h_in = fig_margin_b_in + content_h_in + fig_margin_t_in
    target_cbar_h_in = target_cbar_h_in_new

    # Define the bounding box for the figure in normal coordinates. Note that
    # space for the title is OUTSIDE this box.
    fig_bbox = [fig_margin_l_in / fig_w_in,
                fig_margin_b_in / fig_h_in,
                (fig_w_in - fig_margin_r_in) / fig_w_in,
                (fig_margin_b_in + axes_h_in) / fig_h_in]

    # Create the figure, with no axes defined yet for the colorbar.
    fig, geo_axes = plt.subplots(figsize=(fig_w_in, fig_h_in),
                                 num=fig_num,
                                 dpi=fig_dpi,
                                 subplot_kw={'projection':crs},
                                 gridspec_kw={'left': fig_bbox[0],
                                              'bottom': fig_bbox[1],
                                              'right': fig_bbox[2],
                                              'top': fig_bbox[3]})

    geo_axes.set_facecolor(facecolor)

    # pcolormesh is slower and uses more memory than imshow, which has the
    # added advantage that it can display grid values under the mouse cursor.
    # img = geo_axes.pcolormesh(lon_axis, lat_axis,
    #                           np.flipud(north_down_grid),
    #                           transform=crs,
    #                           cmap=color_ramp['colormap'],
    #                           norm=color_ramp['norm'],
    #                           shading='auto')
    img = geo_axes.imshow(north_down_grid,
                          extent=bbox,
                          transform=crs,
                          cmap=color_ramp['colormap'],
                          norm=color_ramp['norm'])

    # A locally-defined format_cursor_data applied to the imshow artist is the
    # best method currently for formatting grid values displayed at the
    # pointer location in interactive graphics.
    def format_cursor_data(img_val):
        if img_val is np.ma.masked:
            img_val_str = str(img_val)
        else:
            img_val_str = f'{img_val:.5f}'
            while img_val_str[-1] == '0':
                img_val_str = img_val_str[:-1]
            if img_val_str[-1] == '.':
                img_val_str = img_val_str[:-1]
        return(img_val_str)
    img.format_cursor_data = format_cursor_data

    geo_axes.set_title(title)

    # def format_coord(lon, lat):
    #     return(f'lon={lon}, lat={lat}')
    # geo_axes.format_coord = format_coord

    # Get the dimensions of the map and its title.
    geo_ax_bbox = geo_axes.get_tightbbox(fig.canvas.get_renderer())
    _, _, map_plus_title_w_norm, map_plus_title_h_norm = \
        geo_ax_bbox.transformed(fig.transFigure.inverted()).bounds
    map_plus_title_h_in = map_plus_title_h_norm * fig_h_in
    map_title_h_in_actual = map_plus_title_h_in - \
                                geo_axes.get_position().bounds[3] * fig_h_in
    # print(f'map_title_h_in: {map_title_h_in}')
    # print(f'actual: {map_title_h_in_actual}')

    # Reconfigure the figure for the actual title dimension.
    fig_w_in_orig = fig_w_in
    fig_h_in_orig = fig_h_in
    map_scale_in = 0.0
    fig_w_in = 0.0
    fig_h_in = 0.0
    while (fig_w_in < target_fig_dim_in) and (fig_h_in < target_fig_dim_in):
        map_scale_in += 0.001
        map_w_in_expected = map_scale_in * map_aspect
        map_h_in_expected = map_w_in_expected / map_aspect
        fig_w_in = fig_margin_l_in + map_w_in_expected + cbar_gap_in \
            + target_cbar_w_in + fig_margin_r_in
        # Do not allow the colorbar to be taller than the map.
        target_cbar_h_in_new = min([map_h_in_expected, target_cbar_h_in])
        axes_h_in = max([map_h_in_expected, target_cbar_h_in])
        content_h_in = axes_h_in + map_title_h_in_actual
        fig_h_in = fig_margin_b_in + content_h_in + fig_margin_t_in
    target_cbar_h_in = target_cbar_h_in_new

    # print(f'resizing figure from {fig_w_in_orig} x {fig_h_in_orig} to ' +
    #       f'{fig_w_in} x {fig_h_in}')
    # plt.pause(5)
    fig.set_size_inches(fig_w_in, fig_h_in)
    # print('done')
    # plt.pause(5)

    # See
    # https://scitools.org.uk/cartopy/docs/v0.19/matplotlib/feature_interface.html
    geo_axes.coastlines(resolution='10m', linewidth=map_lines_width)

    # Change the scale below to "50m" to get fewer details at the expense of
    # resolution for state boundaries, etc.
    # geo_axes.add_feature(cfNEF(category='cultural',
    #                            name='admin_1_states_provinces_lakes',
    #                            scale='10m',
    #                            edgecolor=map_lines_color,
    #                            facecolor='none',
    #                            linewidth=map_lines_width))
    # Draw U.S. states and Canadian provinces, but not Mexican states.
    geo_axes.add_feature(STATES,
                         edgecolor=map_lines_color,
                         facecolor='none',
                         linewidth=map_lines_width)

    # Determine the colorbar fraction and pad relative to the figure
    # dimensions, which were calculated to provide for this space.
    cbar_fraction = \
        target_cbar_w_in / (fig_w_in - fig_margin_l_in - fig_margin_r_in)
    cbar_pad = \
        cbar_gap_in / (fig_w_in - fig_margin_l_in - fig_margin_r_in)

    # Create the colorbar, which steals an axis from the space allocated to
    # the original axes of the figure.
    # if auto_format_z:
    #     # Use a formatter, which may benefit display of the image value at
    #     # the cursor location in interactive plotting. This will depend on
    #     # your matplotlib version. It works with 3.4.3, it does not work with
    #     # 3.5.1 (gives a "BoundaryNorm is not invertible error"), and it is
    #     # ignored with 3.7.2, which displays the image value but I have not
    #     # figured out how it is formatted. 
    #     formatter = mpl.ticker.FuncFormatter(format_ticks_and_cursor)
    #     logger.info('!!! Using auto formatter !!!')
    # else:
    #     formatter = None
    if color_ramp['colormap'].N > 20:
        extendfrac=None
    else:
        extendfrac='auto'
    cbar = fig.colorbar(img,
                        fraction=cbar_fraction,
                        pad=cbar_pad,
                        # cax=cbar_ax,
                        # use_gridspec=True,
                        # pad=cbar_gap_in / target_cbar_w_in,
                        extend=color_ramp['extend'],
                        extendfrac=extendfrac)
                        # format=formatter)
    cbar.set_ticks(color_ramp['tick_levels'])
    # if not auto_format_z:
    cbar.set_ticklabels(color_ramp['tick_labels'])
    cbar_ax = cbar.ax
    cbar.ax.yaxis.set_tick_params(length=4)
    cbar.set_label(colorbar_units,
                   fontsize=mpl.rcParams['axes.labelsize'],
                   color=mpl.rcParams['ytick.color'])

    # Determine the figure dimensions as drawn. This is not needed, only here
    # to verify figure was drawn as requested.
    # fig_w_in, fig_h_in = fig.get_size_inches()
    # print(f'Expected figure size: {fig_w_in}, {fig_h_in}')
    # print(f'Actual figure size: {fig.get_size_inches()}')

    # Adjust positions of axes to match expectations. Remember that the
    # arguments for Axes.set_position are [left, bottom, width, height]
    # in normalized coordinates.
    geo_ax_pos = [fig_margin_l_in / fig_w_in,
                  # fig_margin_b_in / fig_h_in,
                  (fig_margin_b_in + \
                   0.5 * (axes_h_in - map_h_in_expected)) / fig_h_in,
                  map_w_in_expected / fig_w_in,
                  map_h_in_expected / fig_h_in]

    cbar_ax_pos = \
        [(fig_w_in - fig_margin_r_in - target_cbar_w_in) / fig_w_in,
         (fig_margin_b_in + 0.5 * (axes_h_in - target_cbar_h_in)) / fig_h_in,
         target_cbar_w_in / fig_w_in,
         target_cbar_h_in / fig_h_in]

    # print('\nDesired figure size, geo_ax_pos, and cbar_ax_pos:')
    # print(f'figure size: {fig_w_in} x {fig_h_in}')
    # print(f'geo_axes position: {geo_ax_pos}')
    # print(f'colorbar position: {cbar_ax_pos}')
    # print(f'geo_axes width: {map_w_in_expected}')
    # print(f'colorbar width: {target_cbar_w_in}')
    # plt.pause(15)

    # print('\nbefore/after geo_axes.set_position:')
    # print(f'figure size: {fig.get_size_inches()}')
    # print(f'geo_axes position: {geo_axes.get_position()}')
    # print(f'geo_axes width {geo_axes.get_position().bounds[2] * fig_w_in}')
    # print(f'colorbar position: {cbar.ax.get_position()}')
    # print(f'colorbar width: {cbar.ax.get_position().bounds[2] * fig_w_in}')
    geo_axes.set_position(geo_ax_pos)
    # print(f'figure size: {fig.get_size_inches()}')
    # print(f'geo_axes position: {geo_axes.get_position()}')
    # print(f'geo_axes width {geo_axes.get_position().bounds[2] * fig_w_in}')
    # print(f'colorbar position: {cbar.ax.get_position()}')
    # print(f'colorbar width: {cbar.ax.get_position().bounds[2] * fig_w_in}')
    # plt.pause(15)

    # print('\nbefore/after cbar.ax.set_position:')
    # print(f'figure size: {fig.get_size_inches()}')
    # print(f'geo_axes position: {geo_axes.get_position()}')
    # print(f'geo_axes width {geo_axes.get_position().bounds[2] * fig_w_in}')
    # print(f'colorbar position: {cbar.ax.get_position()}')
    # print(f'colorbar width: {cbar.ax.get_position().bounds[2] * fig_w_in}')
    cbar.ax.set_position(cbar_ax_pos)
    # print(f'figure size: {fig.get_size_inches()}')
    # print(f'geo_axes position: {geo_axes.get_position()}')
    # print(f'geo_axes width {geo_axes.get_position().bounds[2] * fig_w_in}')
    # print(f'colorbar position: {cbar.ax.get_position()}')
    # print(f'colorbar width: {cbar.ax.get_position().bounds[2] * fig_w_in}')
    # plt.pause(15)

    #fig.tight_layout()

    # # The only uncertainty at this point is whether or not the title text
    # # fits as anticipated by the map_title_h_in.

    # # Get the dimensions of the map and its title.
    # geo_ax_bbox = geo_axes.get_tightbbox(fig.canvas.get_renderer())
    # _, _, map_plus_title_w_norm, map_plus_title_h_norm = \
    #     geo_ax_bbox.transformed(fig.transFigure.inverted()).bounds
    # map_plus_title_h_in = map_plus_title_h_norm * fig_h_in
    # print(f'map_title_h_in: {map_title_h_in_actual}')
    # map_title_h_in_actual = map_plus_title_h_in - \
    #                             geo_axes.get_position().bounds[3] * fig_h_in
    # print(f'actual: {map_title_h_in_actual}')

    # # Reconfigure the figure for the actual title dimension.
    # fig_w_in_orig = fig_w_in
    # fig_h_in_orig = fig_h_in
    # map_scale_in = 0.0
    # fig_w_in = 0.0
    # fig_h_in = 0.0
    # while (fig_w_in < target_fig_dim_in) and (fig_h_in < target_fig_dim_in):
    #     map_scale_in += 0.001
    #     map_w_in_expected = map_scale_in * map_aspect
    #     map_h_in_expected = map_w_in_expected / map_aspect
    #     fig_w_in = fig_margin_l_in + map_w_in_expected + cbar_gap_in \
    #         + target_cbar_w_in + fig_margin_r_in
    #     axes_h_in = max([map_h_in_expected, target_cbar_h_in])
    #     content_h_in = axes_h_in + map_title_h_in_actual
    #     # content_h_in = max([map_h_in_expected + map_title_h_in, target_cbar_h_in])
    #     fig_h_in = fig_margin_b_in + content_h_in + fig_margin_t_in

    # print('preparing to resize figure')
    # plt.pause(10)

    # print('resizing figure from ' +
    #       f'{fig_w_in_orig} x {fig_h_in_orig} to ' +
    #       f'{fig_w_in} x {fig_h_in}')
    # fig.set_size_inches(fig_w_in, fig_h_in)
    # plt.pause(10)

    # print(f'original geo_axes position: {geo_ax_pos}')
    # geo_ax_pos = [fig_margin_l_in / fig_w_in,
    #               fig_margin_b_in / fig_h_in,
    #               map_w_in_expected / fig_w_in,
    #               map_h_in_expected / fig_h_in]
    # print(f'new geo_axes position: {geo_ax_pos}')
    # geo_axes.set_position(geo_ax_pos)
    # plt.pause(10)

    # print(f'original colorbar position: {cbar_ax_pos}')
    # print(f'a.k.a. {cbar.ax.get_position()}')
    # cbar_ax_pos = \
    #     [(fig_w_in - fig_margin_r_in - target_cbar_w_in) / fig_w_in,
    #      (fig_margin_b_in + 0.5 * (axes_h_in - target_cbar_h_in)) / fig_h_in,
    #      target_cbar_w_in / fig_w_in,
    #      target_cbar_h_in / fig_h_in]
    # print(f'new colorbar position: {cbar_ax_pos}')
    # cbar.ax.set_position(cbar_ax_pos)
    # print(f'a.k.a. {cbar.ax.get_position()}')
    # plt.pause(10)
    # print(f'geo_axes width {geo_axes.get_position().bounds[2] * fig_w_in}')
    # pass
    # # Determine the initial geo_axes position, not including labels and
    # # titles, in normalized coordinates.
    # geo_ax_xmin_norm, geo_ax_ymin_norm, map_w_norm, map_h_norm = \
    #     geo_axes.get_position().bounds

    # # Get the initial map dimensions in inches.
    # map_w_in_initial = map_w_norm * fig_w_in
    # map_h_in_initial = map_h_norm * fig_h_in

    # print(f'Expected map size: {map_w_in_expected}, {map_h_in_expected}')
    # print(f'Actual map size: {map_w_in_initial}, {map_h_in_initial}')

    # # Get the title dimensions.
    # geo_ax_bbox = geo_axes.get_tightbbox(fig.canvas.get_renderer())
    # _, _, map_plus_title_width_norm, map_plus_title_height_norm = \
    #     geo_ax_bbox.transformed(fig.transFigure.inverted()).bounds
    # map_plus_title_h_in_initial = map_plus_title_height_norm * fig_h_in
    # map_title_h_in = map_plus_title_h_in_initial - map_h_in_initial

    # # Get the initial colorbar dimensions.
    # cbar_ax_bbox = cbar.ax.get_tightbbox(fig.canvas.get_renderer())
    # _, _, cbar_w_norm_initial, cbar_h_norm_initial = \
    #     cbar_ax_bbox.transformed(fig.transFigure.inverted()).bounds
    # cbar_w_in_initial = cbar_w_norm_initial * fig_w_in
    # cbar_h_in_initial = cbar_h_norm_initial * fig_h_in

    # print(f'Colorbar width {cbar_w_in_initial}, height {cbar_h_in_initial}')
    # print(f'geo_axes width {map_w_in_initial}, height {map_h_in_initial}')
    # print(f'Title height {map_title_h_in}')

    # # Rules for the colorbar:
    # # 1. Height: at least 50% the figure height
    # # 2. Height: not larger than the map
    # # 3. Width: at most 10% of the figure

    # # Determine what the dimensions of the figure would be if the current map
    # # and colorbar were combined with the requested margins/gap.
    # fig_w_used_in = fig_margin_l_in + map_w_in_initial + cbar_gap_in \
    #     + cbar_w_in_initial + fig_margin_r_in

    # # Since the margins/gap have not been enforced, there is no way the
    # # figure will match the arrangement assumed when fig_w_in and fig_h_in
    # # were calculated.
    # # Iteratively resize the map.
    # new_map_w_in = map_w_norm * fig_w_in
    # new_map_h_in = map_h_norm * fig_h_in
    # new_fig_w_in = fig_w_in
    # new_fig_h_in = fig_h_in
    # # while new_fig_di

    return fig, geo_axes, cbar
