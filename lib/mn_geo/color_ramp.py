'''
Color ramps for geographic display
'''
# import sys
# import os
# import errno
import logging
# import time
import math

import numpy as np
# from osgeo import gdal,osr,gdalconst
# import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.colors as mplcol
from matplotlib.colors import LinearSegmentedColormap
import colorsys
# import matplotlib.pyplot as plt
# from cartopy.feature import NaturalEarthFeature as cfNEF
# import pygrib

# TODO: update last 10 functions, starting with
#       dbz_monthly_precip_mm_colors_dm


def snowfall_difference_from_mean_inches():
    """
    Define a color ramp for the difference between a snowfall accumulation
    and a normal or period-of-record mean, assuming units of inches.
    """
    red = [191, 204, 234, 249, 255, 210, 178,  62,  23,  62, 191]
    grn = [152,  51, 129, 203, 255, 210, 255, 218, 120,   0, 114]
    blu = [114,   0,  23,  62, 178, 210, 255, 249, 234, 204, 191]
    alp = [1.0] * len(red)

    red = np.array(red, dtype=np.float64) / 255.0
    grn = np.array(grn, dtype=np.float64) / 255.0
    blu = np.array(blu, dtype=np.float64) / 255.0
    alp = np.array(alp, dtype=np.float64)

    col_levels = [-1.0e8, -36.0, -24.0, -12.0, -6.0, -1.0,
                  1.0, 6.0, 12.0, 24.0, 36.0, 1.0e8]

    # Steal the first and last colors for set_under and set_over.
    extend = 'both'

    # Steal the first color for set_under.
    rgba_bottom = (red[0], grn[0], blu[0], alp[0])
    red = red[1:]
    grn = grn[1:]
    blu = blu[1:]
    alp = alp[1:]
    col_levels = col_levels[1:]

    # Steal the last color for set_over.
    rgba_top = (red[-1], grn[-1], blu[-1], alp[-1])
    red = red[:-1]
    grn = grn[:-1]
    blu = blu[:-1]
    alp = alp[:-1]
    col_levels = col_levels[:-1]

    tick_labels = [format_float(lev) for lev in col_levels]

    rgba = np.transpose(np.array([red, grn, blu, alp]))
    colormap = mplcol.ListedColormap(rgba, name='difference_from_mean')
    colormap.set_under(rgba_bottom)
    colormap.set_over(rgba_top)
    norm = mplcol.BoundaryNorm(col_levels, len(col_levels) - 1)

    color_ramp = {'colormap': colormap,
                  'col_levels': col_levels,
                  'tick_levels': col_levels,
                  'tick_labels': tick_labels,
                  'norm': norm,
                  'extend': extend}

    return color_ramp


def percent_of_mean_dry_to_wet():
    """
    Define a color ramp for percent of normal (precipitation, snowfall, etc.).
    Originally intended for snowfall percentage of period-of-record mean.
    """
    red = [191, 204, 234, 249, 255, 210, 178,  62,  23,  62, 191]
    grn = [152,  51, 129, 203, 255, 210, 255, 218, 120,   0, 114]
    blu = [114,   0,  23,  62, 178, 210, 255, 249, 234, 204, 191]
    alp = [1.0] * len(red)

    red = np.array(red, dtype=np.float64) / 255.0
    grn = np.array(grn, dtype=np.float64) / 255.0
    blu = np.array(blu, dtype=np.float64) / 255.0
    alp = np.array(alp, dtype=np.float64)

    col_levels = [0.0, 10.0, 25.0, 50.0, 75.0, 90.0,
                  110.0, 125.0, 150.0, 200.0, 300.0, 1.0e8]

    # Steal the last color for set_over.
    extend = 'max'
    rgba_top = (red[-1], grn[-1], blu[-1], alp[-1])
    red = red[:-1]
    grn = grn[:-1]
    blu = blu[:-1]
    alp = alp[:-1]
    col_levels = col_levels[:-1]

    tick_labels = [format_float(lev) for lev in col_levels]

    rgba = np.transpose(np.array([red, grn, blu, alp]))
    colormap = mplcol.ListedColormap(rgba, name='percent_of_mean')
    colormap.set_over(rgba_top)
    norm = mplcol.BoundaryNorm(col_levels, len(col_levels) - 1)

    color_ramp = {'colormap': colormap,
                  'col_levels': col_levels,
                  'tick_levels': col_levels,
                  'tick_labels': tick_labels,
                  'norm': norm,
                  'extend': extend}

    return color_ramp


def seasonal_snowfall():
    '''
    Define a colormap for snowfall that matches that used for the National
    Snowfall Analysis seasonal accumulations. Recommended units are inches.
    '''
    red = [255,
           189, 107, 49, 8, 8,
           255, 255, 255, 219, 158, 105,
           204, 159, 124, 86, 46,
           64]
    grn = [255,
           215, 174, 130, 81, 38,
           255, 196, 135, 20, 0, 0,
           204, 140, 82, 28, 0,
           223]
    blu = [255,
           231, 214, 189, 156, 148,
           150, 0, 0, 0, 0, 0,
           255, 216, 165, 114, 51,
           255]
    alp = [1.0] * len(red)

    red = np.array(red, dtype=np.float64) / 255.0
    grn = np.array(grn, dtype=np.float64) / 255.0
    blu = np.array(blu, dtype=np.float64) / 255.0
    alp = np.array(alp, dtype=np.float64)

    col_levels = [0.0, 1.0e-8,
                  0.1, 1.0, 2.0, 6.0, 12.0,
                  24.0, 36.0, 48.0, 72.0, 96.0, 120.0,
                  180.0, 240.0, 360.0, 480.0, 600.0, 5000.0]

    tick_labels = [' ',
                   'trace', '0.1 in.', '1 in.', '2 in.', '6 in.', '1 ft',
                   '2 ft', '3 ft', '4 ft', '6 ft', '8 ft', '10 ft',
                   '15 ft', '20 ft', '30 ft', '40 ft', '50 ft',
                   '> 50 ft']

    # Steal the last color for set_over.
    extend = 'max'
    rgba_top = (red[-1], grn[-1], blu[-1], alp[-1])
    red = red[:-1]
    grn = grn[:-1]
    blu = blu[:-1]
    alp = alp[:-1]
    col_levels = col_levels[:-1]
    tick_labels = tick_labels[:-1]

    rgba = np.transpose(np.array([red, grn, blu, alp]))
    colormap = mplcol.ListedColormap(rgba, name='seasonal_snowfall')
    colormap.set_over(rgba_top)
    norm = mplcol.BoundaryNorm(col_levels, len(col_levels) - 1)

    color_ramp = {'colormap': colormap,
                  'col_levels': col_levels,
                  'tick_levels': col_levels,
                  'tick_labels': tick_labels,
                  'norm': norm,
                  'extend': extend}

    return color_ramp


def format_float(val, max_decimals=5, error_threshold=0.0005):
    """
    Format a floating point value compactly while limiting the degradation
    of precision. The choice of a relative error of 0.05% is arbitrary.
    """
    num_dec = 0
    if val == 0:
        return '0'
    while num_dec <= max_decimals:
        test_val = float(f'{val:.{num_dec}f}')
        error = (test_val - val) / val
        # if abs(error) < 10**-max_decimals:
        if abs(error) < error_threshold or num_dec == max_decimals:
            break
        num_dec += 1
    return f'{val:.{num_dec}f}'


def discrete_linear_color_ramp(cmap,
                               cmap_name,
                               min_value,
                               max_value,
                               num_colors,
                               extend=None,
                               sat_begin=1.0,
                               sat_end=1.0,
                               val_begin=1.0,
                               val_end=1.0):
    """
    Using a provided colormap, create a discrete linear color ramp covering
    a given range from min_value to max_value across num_colors RGB values
    taken from the cmap.
    """

    # Sample the colormap.
    cmap_ramp = np.linspace(0, 1, num_colors)
    rgba = cmap(cmap_ramp)

    # Apply linear variations in saturation and/or brightness.
    # sat_begin = 0.9
    # sat_end = 0.1
    # val_begin = 0.2
    # val_end = 0.8
    if (sat_begin != sat_end) or (val_begin != val_end):
        sat_multiplier = np.linspace(sat_begin, sat_end, num_colors)
        val_multiplier = np.linspace(val_begin, val_end, num_colors)
        rgb = rgba[:,:3]
        for i, [r, g, b] in enumerate(rgb):
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            s = s * sat_multiplier[i]
            s = max([s, 0.0])
            s = min([s, 1.0])
            v = v * val_multiplier[i]
            v = max([v, 0.0])
            v = min([v, 1.0])
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            rgba[i, :3] = [r, g, b]
        rgb = None 

    col_levels = np.linspace(min_value, max_value, num_colors + 1)
    tick_labels = [format_float(lev) for lev in col_levels]

    red = rgba[:, 0]
    grn = rgba[:, 1]
    blu = rgba[:, 2]
    alp = rgba[:, 3]

    if extend == 'both' or extend == 'min':
        # Steal the first color for set_under.
        rgba_bottom = (red[0], grn[0], blu[0], alp[0])
        red = red[1:]
        grn = grn[1:]
        blu = blu[1:]
        alp = alp[1:]
        col_levels = col_levels[1:]
        tick_labels = tick_labels[1:]
        extend = 'both'

    if extend == 'both' or extend == 'max':
        # Steal the last color for set_over.
        rgba_top = (red[-1], grn[-1], blu[-1], alp[-1])
        red = red[:-1]
        grn = grn[:-1]
        blu = blu[:-1]
        alp = alp[:-1]
        col_levels = col_levels[:-1]
        tick_labels = tick_labels[:-1]

    rgba = np.transpose(np.array([red, grn, blu, alp]))

    colormap = mplcol.ListedColormap(rgba, name=cmap_name)

    if extend == 'both' or extend == 'min':
        colormap.set_under(rgba_bottom)
    if extend == 'both' or extend == 'max':
        colormap.set_over(rgba_top)

    tick_levels = col_levels
    norm = mplcol.BoundaryNorm(col_levels, len(col_levels) - 1)

    color_ramp = {'colormap': colormap,
                  'col_levels': col_levels,
                  'tick_levels': tick_levels,
                  'tick_labels': tick_labels,
                  'norm': norm,
                  'extend': extend}

    return color_ramp


def auto_generate_discrete_stepwise_scale(z_min,
                                          z_max,
                                          min_steps,
                                          max_steps,
                                          allow_out_of_bounds=False):
    """
    Attempt to find a discrete stepwise scale that uses from min_steps to
    max_steps, has a user-friendly equal-sized step, and comes close to
    accommodating the data range. Useful for auto-generating discrete
    linear color ramps.
    """
    z_range = z_max - z_min
    num_steps = [i for i in range(min_steps, max_steps + 1)]

    best_sufficient_padding = None
    best_insufficient_padding = None

    if allow_out_of_bounds:
        num_out_of_bounds = 2
    else:
        num_out_of_bounds = 0

    for i, nc in enumerate(num_steps):
        orig_step = z_range / nc
        order = np.ceil(np.log10(orig_step))
        scale = 10**order
        potential_steps = \
            scale * np.array([0.1, 0.2, 0.25, 0.5, 1.0, 2.0])
        for step in potential_steps:
            ramp_low_ind = math.floor(z_min / step)
            ramp_high_ind = math.ceil(z_max / step)
            num_steps_needed = ramp_high_ind - ramp_low_ind
            if num_steps_needed > nc + num_out_of_bounds:
                continue
            num_steps_used = num_steps_needed

            if num_steps_used < nc:
                # To use all nc steps, expand the ramp (even though the
                # range of the data does not require it). Start with
                # whichever end is closest to its corresponding extreme.
                top_padding = ramp_high_ind * step - z_max
                bottom_padding = z_min - ramp_low_ind * step
                if top_padding < 0 or bottom_padding < 0:
                    print('programming mistake')
                    sys.exit(1)
                if top_padding < bottom_padding:
                    end = 1 # start at the top
                else:
                    end = -1 # start at the bottom
            while num_steps_used < nc:
                # Alternately expand the range of the potential ramp until
                # we hit nc steps.
                if end < 0:
                    ramp_low_ind -= 1
                else:
                    ramp_high_ind += 1
                end *= -1
                num_steps_used = ramp_high_ind - ramp_low_ind

            if num_steps_used > nc:
                # Will need to compress the map. Start with whichever
                # end is furthest from its corresponding extreme.
                top_padding = ramp_high_ind * step - z_max
                bottom_padding = z_min - ramp_low_ind * step
                if top_padding < 0 or bottom_padding < 0:
                    print('programming mistake')
                    sys.exit(1)
                if top_padding > bottom_padding:
                    end = 1 # start at the top
                else:
                    end = -1 # start at the bottom
            while num_steps_used > nc:
                # Alternately compress the range of the potential ramp.
                if end < 0:
                    ramp_low_ind += 1
                else:
                    ramp_high_ind -= 1
                end *= -1
                num_steps_used = ramp_high_ind - ramp_low_ind

            ramp_min = ramp_low_ind * step
            ramp_max = ramp_high_ind * step
            top_padding = ramp_max - z_max
            bottom_padding = z_min - ramp_min
            total_padding = top_padding + bottom_padding

            if total_padding >= 0:
                if best_sufficient_padding is None or \
                    total_padding < best_sufficient_padding:
                    best_sufficient_padding = total_padding
                    best_sufficient_num_cols = nc
                    best_sufficient_step = step
                    best_sufficient_ramp_min = ramp_min
                    best_sufficient_ramp_max = ramp_max
            if total_padding < 0:
                if best_insufficient_padding is None or \
                    total_padding > best_insufficient_padding:
                    best_insufficient_padding = total_padding
                    best_insufficient_num_cols = nc
                    best_insufficient_step = step
                    best_insufficient_ramp_min = ramp_min
                    best_insufficient_ramp_max = ramp_max

    # Find the combination of num_steps and step size that gives the
    # least padding.
    if best_insufficient_padding is not None and \
        abs(best_insufficient_padding) < best_sufficient_padding:
        num_steps = best_insufficient_num_cols
        step = best_insufficient_step
        ramp_min = best_insufficient_ramp_min
        ramp_max = best_insufficient_ramp_max
    else:
        num_steps = best_sufficient_num_cols
        step = best_sufficient_step
        ramp_min = best_sufficient_ramp_min
        ramp_max = best_sufficient_ramp_max

    return ramp_min, ramp_max, num_steps, step


def snow():
    '''
    Define a color ramp to display snow water equivalent (SWE) or snow depth.
    Recommended units are mm for SWE, cm for snow depth. Formerly
    greg_geo.snow_colormap.
    '''
    red = [204, 163, 122, 83, 78, 86, 149, 207,
           229, 245, 255, 255]
    grn = [249, 214, 157, 89, 48, 20, 36, 69,
           106, 147, 187, 226]
    blu = [255, 245, 229, 207, 182, 153, 182, 199,
           188, 185, 194, 221]
    alp = [1.0] * len(red)

    red = np.array(red, dtype=np.float64) / 255.0
    grn = np.array(grn, dtype=np.float64) / 255.0
    blu = np.array(blu, dtype=np.float64) / 255.0
    alp = np.array(alp, dtype=np.float64)
    rgba = np.transpose(np.array([red, grn, blu, alp]))

    colormap = mplcol.ListedColormap(rgba, name='snow')

    colormap.set_under('xkcd:light grey')
    colormap.set_over('xkcd:ocean blue')

    # Set color levels for mm of SWE or cm of depth.
    col_levels = [0.0, 1.0, 5.0, 10.0, 25.0, 50.0,
                  100.0, 150.0, 250.0, 500.0, 750.0,
                  1000.0, 2000.0]
    tick_levels = col_levels

    # Label color levels.
    tick_labels = ['0', '1', '5', '10', '25', '50', '100',
                   '150', '250', '500', '750', '1000', '2000']

    norm = mplcol.BoundaryNorm(col_levels, len(col_levels) - 1)

    color_ramp = {'colormap': colormap,
                  'col_levels': col_levels,
                  'tick_levels': tick_levels,
                  'tick_labels': tick_labels,
                  'norm': norm,
                  'extend': 'both'}

    return color_ramp


def stage4_qpe():
    '''
    Define a colormap for the colors used by NCEP/EMC for precipitation
    images at ftp://ftp.ncep.noaa.gov/pub/data/nccf/com/pcpanl/prod
    Formerly greg_geo.stage4_colors
    '''
    red = [255, 0, 0, 127, 238, 255, 255, 255,
           238, 205, 139, 145, 137, 16, 30]
    grn = [228, 139, 205, 255, 238, 215, 165, 127,
           64, 0, 0, 44, 104, 78, 144]
    blu = [220, 0, 0, 0, 0, 0, 79, 0,
           0, 0, 0, 238, 205, 139, 255]
    alp = [1.0] * len(red)

    red = np.array(red, dtype=np.float64) / 255.0
    grn = np.array(grn, dtype=np.float64) / 255.0
    blu = np.array(blu, dtype=np.float64) / 255.0
    alp = np.array(alp, dtype=np.float64)
    rgba= np.transpose(np.array([red, grn, blu, alp]))

    colormap = mplcol.ListedColormap(rgba, name='stage_IV')

    colormap.set_over('xkcd:robin egg blue')

    # Set levels for a discrete colormap. The lowest value needs to be a
    # tiny negative. If it is set to zero, and the grid has any significant
    # number of below-bounds values (e.g., large negative no-data values),
    # zero values will be displayed as out-of-bounds by the contourf function.
    col_levels = [0.0, 0.1, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0,
                  35.0, 50.0, 75.0, 100.0, 125.0, 150.0, 175.0, 1.0e6]
    tick_levels = col_levels

    tick_labels = ['0', '0.1', '2', '5', '10', '15', '20', '25',
                   '35', '50', '75', '100', '125', '150', '175',
                   '>175']

    norm = mplcol.BoundaryNorm(col_levels, 15)

    color_ramp = {'colormap': colormap,
                  'col_levels': col_levels,
                  'tick_levels': tick_levels,
                  'tick_labels': tick_labels,
                  'norm': norm,
                  'extend': 'max'}

    return color_ramp


def snowfall():
    '''
    Define a colormap for snowfall that matches that used for the National
    Snowfall Analysis. Recommended units are inches. Formerly
    greg_geo.snowfall_colors.
    '''
    red = [228, 189, 107, 49,
           8, 8,
           255, 255, 255, 219, 158, 105, 54, 204,
           159, 124, 86]
    grn = [238, 215, 174, 130,
           81, 38,
           255, 196, 135, 20, 0, 0, 0, 204,
           140, 82, 28]
    blu = [245, 231, 214, 189,
           156, 148,
           150, 0, 0, 0, 0, 0, 0, 255,
           216, 165, 114]
    alp = [1.0] * len(red)

    red = np.array(red, dtype=np.float64) / 255.0
    grn = np.array(grn, dtype=np.float64) / 255.0
    blu = np.array(blu, dtype=np.float64) / 255.0
    alp = np.array(alp, dtype=np.float64)
    rgba = np.transpose(np.array([red, grn, blu, alp]))

    colormap = mplcol.ListedColormap(rgba, name='snowfall')

    colormap.set_under(color='#ffffffff')
    colormap.set_over(color='#2e0033ff') # 46, 0, 51

    # Set levels for a discrete colormap.
    col_levels = [0.0, 0.1, 1.0, 2.0, 3.0,
                  4.0, 6.0,
                  8.0, 12.0, 18.0, 24.0, 30.0, 36.0, 48.0, 60.0,
                  72.0, 96.0, 120.0]
    tick_levels = col_levels

    tick_labels = ['0', '0.1', '1', '2', '3', '4', '6', '8',
                   '12', '18', '24', '30', '36', '48', '60',
                   '72', '96', '120']

    norm = mplcol.BoundaryNorm(col_levels, len(col_levels) - 1)

    # TODO: change 'both' to 'max'? Snowfall is bounded. Something like a
    #       "faceolor" or similar should be used.
    color_ramp = {'colormap': colormap,
                  'col_levels': col_levels,
                  'tick_levels': tick_levels,
                  'tick_labels': tick_labels,
                  'norm': norm,
                  'extend': 'both'}

    return color_ramp


def dbz_hourly_precip_mm():
    '''
    Define a simple color ramp that is similar to radar reflectivity images
    for hourly precipitation accumulations. Strongly recommended units for the
    displayed data are millimeters, since a mm/in conversion is used for tick
    labels. Formerly greg_geo.dbz_hourly_precip_mm_colors
    '''
    red = [255, 34, 25, 17, 255, 224, 253, 251, 202, 176, 252]
    grn = [228, 255, 194, 130, 255, 181, 125, 0, 0, 0, 0]
    blu = [220, 8, 3, 0, 11, 10, 9, 6, 5, 2, 255]
    alp = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    red = np.array(red, dtype=np.float64) / 255.0
    grn = np.array(grn, dtype=np.float64) / 255.0
    blu = np.array(blu, dtype=np.float64) / 255.0
    rgba = np.transpose(np.array([red,grn,blu,alp]))

    colormap = mplcol.ListedColormap(rgba, name='dBZ')

    colormap.set_over(color='#853abeff') # 133, 58, 190

    # OLD COMMENT
    # Set levels for a discrete colormap. The lowest value needs to be a
    # tiny negative. If it is set to zero, and the grid has any significant
    # number of below-bounds values (e.g., large negative no-data values),
    # zero values will be displayed as out-of-bounds by the contourf function.
    # col_levels = [-1.0e-6, 0.1, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0,
    #               35.0, 50.0, 75.0, 100.0, 125.0, 150.0, 175.0, 1.0e6]
    # col_levels = [0.0, 1.0e-8, 0.25, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0,
    #               200.0, 400.0]
    col_levels_in = [0.0, 1.0e-8, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0,
                     4.0, 8.0, 16.0]
    tick_labels_in = ['', '0', 'trace', '0.05', '0.1', '0.25', '0.5', '1',
                      '2', '4', '8', '16']

    col_levels_mm = [i * 25.4 for i in col_levels_in]
    tick_levels_mm = col_levels_mm

    norm = mplcol.BoundaryNorm(col_levels_mm, len(col_levels_mm) - 1)

    color_ramp = {'colormap': colormap,
                  'col_levels': col_levels_mm,
                  'tick_levels': tick_levels_mm,
                  'tick_labels': tick_labels_in,
                  'norm': norm,
                  'extend': 'max'}

    return color_ramp


def dbz_hourly_precip_mm_cont(color_ramp_scale=0):
    '''
    Define a simple color ramp that is similar to radar reflectivity images
    for hourly precipitation accumulations, but use a large number of levels
    to produce a ramp that appears more continuous. Strongly recommended units
    for the displayed data are millimeters, since a mm/in conversion is used
    for tick labels. Formerly greg_geo.dbz_hourly_precip_mm_cont_colors.
    '''
    red = [255, 34, 25, 17, 255, 224, 253, 251, 202, 176, 252]
    grn = [228, 255, 194, 130, 255, 181, 125, 0, 0, 0, 0]
    blu = [220, 8, 3, 0, 11, 10, 9, 6, 5, 2, 255]

    red = np.array(red, dtype=np.float64) / 255.0
    grn = np.array(grn, dtype=np.float64) / 255.0
    blu = np.array(blu, dtype=np.float64) / 255.0

    # Since I am not smart enough to figure out how to combine colormap
    # creation and boundary normalization, I will subdivide all but the bottom
    # two colors.

    # This is a little weird, because we take in the values as millimeters,
    # but the tick_labels are equivalent values in inches. These are closer to
    # dBZ amounts as shown at https://www.weather.gov/jetstream/refl
    # These levels work for aggregate precip over 1 hour to 3 days.

    if color_ramp_scale == 1: # up to 8 feet
        col_levels_in = [0.0, 1.0e-8, 1.0, 6.0, 12.0, 18.0, 24.0, 36.0, 48.0,
                         60.0, 72.0, 96.0]
        tick_labels_in = ['', '0', '1', '6', '12', '18', '24', '36', '48',
                          '60', '72', '96']
    else:
        col_levels_in = [0.0, 1.0e-8, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0,
                         4.0, 8.0, 16.0]
        tick_labels_in = ['', '0', 'trace', '0.05', '0.1', '0.25', '0.5', '1',
                          '2', '4', '8', '16']

    tick_levels_mm = np.array(col_levels_in) * 25.4

    # Create new levels and colors manually for smoother transitions between
    # levels above num_cols_same. Subdivide all the but the first
    # num_cols_same steps in col_levels_in and red, grn, blu values.
    num_cols_orig = len(red)

    num_cols_same = 2 # for zero and trace amounts
    num_cols_sub = 10
    num_cols = num_cols_orig * num_cols_sub

    new_levs = np.empty(num_cols + 1)
    new_red = np.empty(num_cols)
    new_grn = np.empty(num_cols)
    new_blu = np.empty(num_cols)
    for i in range(num_cols_orig):
        if i < num_cols_same:
            new_red[i * num_cols_sub:(i+1) * num_cols_sub] = red[i]
            new_grn[i * num_cols_sub:(i+1) * num_cols_sub] = grn[i]
            new_blu[i * num_cols_sub:(i+1) * num_cols_sub] = blu[i]
        else:
            new_red[i * num_cols_sub:(i+1) * num_cols_sub] = \
                [red[i-1] + (j+1) * (red[i] - red[i-1]) / num_cols_sub
                 for j in range(num_cols_sub)]
            new_grn[i * num_cols_sub:(i+1) * num_cols_sub] = \
                [grn[i-1] + (j+1) * (grn[i] - grn[i-1]) / num_cols_sub
                 for j in range(num_cols_sub)]
            new_blu[i * num_cols_sub:(i+1) * num_cols_sub] = \
                [blu[i-1] + (j+1) * (blu[i] - blu[i-1]) / num_cols_sub
                 for j in range(num_cols_sub)]
        new_levs[i * num_cols_sub:(i+1) * num_cols_sub] = \
            [col_levels_in[i] + j *
             (col_levels_in[i+1] - col_levels_in[i]) / num_cols_sub
             for j in range(num_cols_sub)]
    new_levs[num_cols] = col_levels_in[num_cols_orig]

    col_levels_mm = np.array(new_levs) * 25.4

    rgb = np.transpose(np.array([new_red, new_grn, new_blu]))

    colormap = LinearSegmentedColormap.from_list('dBZ', rgb, num_cols)
    #colormap = mplcol.ListedColormap(rgb, name='dBZ')
    colormap.set_over(color='#853abeff') # 133, 58, 190

    norm = mplcol.BoundaryNorm(col_levels_mm, num_cols)

    color_ramp = {'colormap': colormap,
                  'col_levels': col_levels_mm,
                  'tick_levels': tick_levels_mm,
                  'tick_labels': tick_labels_in,
                  'norm': norm,
                  'extend': 'max'}

    return color_ramp


def dbz_monthly_precip_mm():
    '''
    Define a simple color ramp that is similar to radar reflectivity images
    for hourly precipitation accumulations in millimeters, adjusted to show
    monthly totals. Formerly greg_geo.dbz_monthly_precip_mm_colors
    '''
    red = [255, 34, 25, 17, 255, 224, 253, 251, 202, 176, 252]
    grn = [228, 255, 194, 130, 255, 181, 125, 0, 0, 0, 0]
    blu = [220, 8, 3, 0, 11, 10, 9, 6, 5, 2, 255]
    alp = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    red = np.array(red, dtype=np.float64) / 255.0
    grn = np.array(grn, dtype=np.float64) / 255.0
    blu = np.array(blu, dtype=np.float64) / 255.0
    rgba = np.transpose(np.array([red, grn, blu, alp]))

    colormap = mplcol.ListedColormap(rgba, name='dBZ')

    colormap.set_over(color='#853abeff') # 133, 58, 190

    # col_levels = [0.0, 1.0e-8, 0.25, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0,
    #               200.0, 400.0]
    col_levels = [0.0, 1.0e-8, 2.5, 10.0,
                  25.0, 50.0, 100.0, 250.0, 500.0,
                  1000.0, 2000.0, 4000.0]
    tick_levels = col_levels

    tick_labels = ['', '0', '2.5', '10', '25', '50', '100', '250', '500',
                   '1000', '2000', '4000']

    norm = mplcol.BoundaryNorm(col_levels, len(col_levels) - 1)

    color_ramp = {'colormap': colormap,
                  'col_levels': col_levels,
                  'tick_levels': tick_levels,
                  'tick_labels': tick_labels,
                  'norm': norm,
                  'extend': 'max'}

    return color_ramp


def dbz_monthly_precip_mm_colors_dm():
    '''
    Define a simple color ramp that is similar to radar reflectivity images
    for hourly precipitation accumulations in millimeters, adjusted to show
    monthly totals.
    By user request, this version differs from dbz_monthly_precip_mm_colors
    only in its "zero" color, which is changed from [255, 228, 220] (#ffe4dc)
    to [179, 160, 154] (#b3a09a)
    '''
    red = [179, 34, 25, 17, 255, 224, 253, 251, 202, 176, 252]
    grn = [160, 255, 194, 130, 255, 181, 125, 0, 0, 0, 0]
    blu = [154, 8, 3, 0, 11, 10, 9, 6, 5, 2, 255]
    alp = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    red = np.array(red, dtype=np.float64) / 255.0
    grn = np.array(grn, dtype=np.float64) / 255.0
    blu = np.array(blu, dtype=np.float64) / 255.0
    rgb = np.transpose(np.array([red,grn,blu,alp]))

    colormap = mplcol.ListedColormap(rgb, name='dBZ')
    colormap.set_over(color='#853abeff') # 133, 58, 190

    col_levels = [0.0, 1.0e-8, 0.25, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0,
                  200.0, 400.0]
    col_levels = [0.0, 1.0e-8, 2.5, 10.0,
                  25.0, 50.0, 100.0, 250.0, 500.0,
                  1000.0, 2000.0, 4000.0]
    tick_levels = col_levels
    tick_labels = ['', '0', '2.5', '10', '25', '50', '100', '250', '500',
                   '1000', '2000', '4000']
    norm = mplcol.BoundaryNorm(col_levels, len(col_levels) - 1)

    color_ramp = {'colormap': colormap,
                      'col_levels': col_levels,
                      'tick_levels': tick_levels,
                      'tick_labels': tick_labels,
                      'norm': norm,
                      'extend': 'max'}

    return color_ramp


def smooth_rgb(red, grn, blu, col_levels, num_cols_sub):
    '''
    Smoothly subdivide RGB values and associated levels to expand a color
    ramp by an integer factor num_cols_sub. Code was formerly local to
    greg_geo.precip_diff_mm_cont_colors.
    '''
    # Interpolate RGB values.
    num_colors_orig = len(red)
    num_colors_new = num_colors_orig * num_cols_sub
    flt_orig_col_ind = np.linspace(0, num_colors_orig - 1, num_colors_new)
    ind1 = np.floor(flt_orig_col_ind).astype('int')
    wt1 = np.floor(flt_orig_col_ind) + 1 - flt_orig_col_ind
    ind2 = np.ceil(flt_orig_col_ind).astype('int')
    wt2 = flt_orig_col_ind - np.floor(flt_orig_col_ind)
    new_red = np.take(red, ind1) * wt1 + np.take(red, ind2) * wt2
    new_grn = np.take(grn, ind1) * wt1 + np.take(grn, ind2) * wt2
    new_blu = np.take(blu, ind1) * wt1 + np.take(blu, ind2) * wt2

    # Interpolate col_levels.
    num_levs_orig = len(col_levels)
    num_levs_new = (num_levs_orig - 1) * num_cols_sub + 1
    flt_orig_lev_ind = np.linspace(0, num_levs_orig - 1, num_levs_new)
    ind1 = np.floor(flt_orig_lev_ind).astype('int')
    wt1 = np.floor(flt_orig_lev_ind) + 1 - flt_orig_lev_ind
    ind2 = np.ceil(flt_orig_lev_ind).astype('int')
    wt2 = flt_orig_lev_ind - np.floor(flt_orig_lev_ind)
    new_levs = np.take(col_levels, ind1) * wt1 + \
        np.take(col_levels, ind2) * wt2

    return new_red, new_grn, new_blu, new_levs


def precip_diff_mm_cont_colors(color_ramp_scale=0, num_cols_sub=12):
    '''
    Define a color ramp to show precipitation differences, with positive
    values portrayed as "wet" (blue), and negative values portrayed as "dry"
    (yellow/orange).
    '''
    red = [215, 230, 245, 253, 254, 220, 221, 187, 145, 94, 44]
    grn = [25, 84, 144, 190, 222, 220, 239, 224, 198, 160, 123]
    blu = [28, 55, 83, 115, 153, 220, 207, 224, 222, 202, 182]
    # red = [230, 245, 253, 254, 220, 221, 187, 145, 94]
    # grn = [84, 144, 190, 222, 220, 239, 224, 198, 160]
    # blu = [55, 83, 115, 153, 220, 207, 224, 222, 202]

    # These colors combine 7-class YlOrRd and 7-class YlGnBu from
    # colorbrewer2.org, skipping the first color from each, putting [220, 220,
    # 220] in the middle, and using the last color from each in set_under and
    # set_over below.
    # red = [227, 252, 253, 254, 254, 220, 199, 127, 65, 29, 34]
    # grn = [26, 78, 141, 178, 217, 220, 233, 205, 182, 145, 94]
    # blu = [28, 42, 60, 76, 118, 220, 180, 187, 196, 192, 168]

    # These colors combine 7-class OrRd and 7-class GnBu from
    # colorbrewer2.org, skipping the first color from each, putting [220, 220,
    # 220] in the middle, and using the last color from each in set_under and
    # set_over below.
    # red = [215, 239, 252, 253, 253, 220, 204, 168, 123, 78, 43]
    # grn = [48, 101, 141, 187, 212, 220, 235, 221, 204, 179, 140]
    # blu = [31, 72, 89, 132, 158, 220, 197, 181, 196, 211, 190]

    red = np.array(red, dtype=np.float64) / 255.0
    grn = np.array(grn, dtype=np.float64) / 255.0
    blu = np.array(blu, dtype=np.float64) / 255.0

    # Since I am not smart enough to figure out how to combine colormap
    # creation and boundary normalization, I will subdivide all but the bottom
    # two colors.

    # This is a little weird, because we take in the values as millimeters,
    # but the tick_labels are equivalent values in inches. These are closer to
    # dBZ amounts as shown at https://www.weather.gov/jetstream/refl
    # These levels work for aggregate precip over 1 hour to 3 days.

    if color_ramp_scale == 1:
        col_levels_mm = [-1000.0, -500.0, -250.0, -100.0, -50.0, -25.0,
                         25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
        tick_labels_mm = ['-1000', '-500', '-250', '-100', '-50', '-25',
                          '25', '50', '100', '250', '500', '1000']
    else:
        col_levels_mm = [-100, -50, -25, -10, -5.0, -1.0,
                         1.0, 5.0, 10.0, 25.0, 50.0, 100.0]
        tick_labels_mm = ['-100', '-50', '-25', '-10', '-5', '-1',
                          '1', '5', '10', '25', '50', '100']
    # num_levels = len(col_levels_mm)
    tick_levels_mm = np.array(col_levels_mm)

    # Subdivide colors and levels manually for a continuous-looking ramp.
    # Use num_cols_sub = 1 for no subdivision.
    red, grn, blu, col_levels_mm = \
        smooth_rgb(red, grn, blu, col_levels_mm, num_cols_sub)
    num_colors = len(red)

    rgb = np.transpose(np.array([red, grn, blu]))
    colormap = LinearSegmentedColormap.from_list('delta_precip',
                                                 rgb, num_colors)
    # colormap.set_under('#d7191c') # 215, 25, 58
    # colormap.set_over('#2c7bb6') # 44, 123, 182
    colormap.set_under('#a50026') # 165, 0, 38
    colormap.set_over('#313695') # 49, 54, 149

    # Extremes of 7-class YlOrRd and 7-class YlGnBu
    # colormap.set_under('#8c2d04') # 140, 45, 4
    # colormap.set_over('#0c2c84') # 12, 44, 132

    # Extremes of 7-class OrRd and 7-class GnBu
    # colormap.set_under('#990000') # 153, 0, 0
    # colormap.set_over('#08589e') # 8, 88, 158

    norm = mplcol.BoundaryNorm(col_levels_mm, num_colors)
    color_ramp = {'colormap': colormap,
                      'col_levels': col_levels_mm,
                      'tick_levels': tick_levels_mm,
                      'tick_labels': tick_labels_mm,
                      'norm': norm,
                      'extend': 'both'}

    return color_ramp


def mrms_precip_mm_colors(scale_inches=24,
                          set_under='xkcd:light grey',
                          set_under_alpha=1.0):
    '''
    Define a simple color ramp that is similar to that used for precipitation
    in the MRMS Operational Product Viewer.
    https://mrms.nssl.noaa.gov/qvs/product_viewer

    The "scale_inches" keyword matches the options for "QPE Scale" in the
    Operational Product Viewer. Defaults are the following:

    duration  scale
    --------  -----
       3 hr     8
       6 hr    16
      12 hr    16
      24 hr    24
      48 hr    32
      72 hr    40

    Other scales available include 3, 5, 12, 50, 60, 75, and 90 inch.
    Currently this function only supportys 3, 5, 8, 12, 16, and 24. Too lazy
    to slog through the other six today.
    '''

    logger = logging.getLogger()

    # The MRMS color ramps all use the same 24 RGB values. The last value is
    # the set_over color.
    red = [0, 0, 0, 0, 0, 0, 0, 0, 255, 240, 231, 200,
           255, 255, 230, 180, 255, 217, 164, 120, 255, 192, 192, 255]
    grn = [236, 200, 160, 60, 255, 220, 190, 141, 255, 210, 180, 120,
           160, 60, 0, 0, 0, 0, 0, 0, 255, 192, 255, 255]
    blu = [236, 240, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0,
           160, 60, 0, 0, 255, 217, 164, 120, 255, 255, 255, 192]
    alp = [1.0] * len(red)
    red = np.array(red, dtype=np.float64) / 255.0
    grn = np.array(grn, dtype=np.float64) / 255.0
    blu = np.array(blu, dtype=np.float64) / 255.0
    rgb = np.transpose(np.array([red[:-1],grn[:-1],blu[:-1],alp[:-1]]))

    mrms_colormap = mplcol.ListedColormap(rgb, name='MRMS')
    mrms_colormap.set_under(color=set_under, alpha=set_under_alpha)
    mrms_colormap.set_over(color='#ffffc0ff') # 255, 255, 192

    # Initial values of col_levels_in represent the lower bound of the range
    # represented by each RGB value.
    if scale_inches == 3:
        col_levels_in = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
                         0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.20,
                         1.40, 1.60, 1.80, 2.00, 2.25, 2.50, 2.75, 3.00]
    elif scale_inches == 5:
        col_levels_in = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40,
                         0.50, 0.60, 0.80, 1.00, 1.25, 1.50, 1.75, 2.00,
                         2.25, 2.50, 2.75, 3.00, 3.50, 4.00, 4.50, 5.00]
    elif scale_inches == 8:
        col_levels_in = [0.01, 0.05, 0.10, 0.15, 0.20, 0.40, 0.60, 0.80,
                         1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00, 3.50,
                         4.00, 4.50, 5.00, 5.50, 6.00, 6.50, 7.00, 8.00]
    elif scale_inches == 12:
        col_levels_in = [0.01, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00,
                         1.25, 1.50, 1.75, 2.00, 2.50, 3.00, 3.50, 4.00,
                         5.00, 6.00, 7.00, 8.00, 9.00, 10.0, 11.0, 12.0]
    elif scale_inches == 16:
        col_levels_in = [0.01, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00,
                         1.25, 1.50, 2.00, 2.50, 3.00, 3.50, 4.00, 5.00,
                         6.00, 7.00, 8.00, 9.00, 10.0, 12.0, 14.0, 16.0]
    elif scale_inches == 24:
        col_levels_in = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50,
                         2.00, 2.50, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00,
                         9.00, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 24.0]
    else:
        msg = f'scale_inches value of {scale_inches} is not supported.'
        logger.error(msg)

    col_levels_mm = [f * 25.4 for f in col_levels_in]

    # Add a value for the upper bound of the last value.
    # col_levels_mm.append(1.0e8)

    tick_labels = [f'{f}' for f in col_levels_in]
    tick_labels.append('')

    norm = mplcol.BoundaryNorm(col_levels_mm, len(col_levels_mm) - 1)
    mrms_color_ramp = {'colormap': mrms_colormap,
                       'col_levels': col_levels_mm,
                       'tick_levels': col_levels_mm,
                       'tick_labels': tick_labels,
                       'norm': norm,
                       'extend': 'max'}

    return mrms_color_ramp


def mrms_precip_in_colors(scale_inches=24,
                          set_under='xkcd:light grey',
                          set_under_alpha=1.0):
    '''
    Define a simple color ramp that is similar to that used for precipitation
    in the MRMS Operational Product Viewer.
    https://mrms.nssl.noaa.gov/qvs/product_viewer

    The "scale_inches" keyword matches the options for "QPE Scale" in the
    Operational Product Viewer. Defaults are the following:

    duration  scale
    --------  -----
       3 hr     8
       6 hr    16
      12 hr    16
      24 hr    24
      48 hr    32
      72 hr    40

    Other scales available include 3, 5, 12, 50, 60, 75, and 90 inch.
    Currently this function only supportys 3, 5, 8, 12, 16, and 24. Too lazy
    to slog through the other six today.
    '''

    logger = logging.getLogger()

    # The MRMS color ramps all use the same 24 RGB values.
    red = [0, 0, 0, 0, 0, 0, 0, 0, 255, 240, 231, 200,
           255, 255, 230, 180, 255, 217, 164, 120, 255, 192, 192, 255]
    grn = [236, 200, 160, 60, 255, 220, 190, 141, 255, 210, 180, 120,
           160, 60, 0, 0, 0, 0, 0, 0, 255, 192, 255, 255]
    blu = [236, 240, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0,
           160, 60, 0, 0, 255, 217, 164, 120, 255, 255, 255, 192]
    alp = [1.0] * len(red)

    red = np.array(red, dtype=np.float64) / 255.0
    grn = np.array(grn, dtype=np.float64) / 255.0
    blu = np.array(blu, dtype=np.float64) / 255.0
    alp = np.array(alp)

    # Steal the last color for set_over.
    rgba_top = (red[-1], grn[-1], blu[-1], alp[-1])
    red = red[:-1]
    grn = grn[:-1]
    blu = blu[:-1]
    alp = alp[:-1]

    rgb = np.transpose(np.array([red, grn, blu, alp]))

    mrms_colormap = mplcol.ListedColormap(rgb, name='MRMS')
    mrms_colormap.set_under(color=set_under, alpha=set_under_alpha)
    mrms_colormap.set_over(rgba_top)

    # Initial values of col_levels_in represent the lower bound of the range
    # represented by each RGB value.
    if scale_inches == 3:
        col_levels_in = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
                         0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.20,
                         1.40, 1.60, 1.80, 2.00, 2.25, 2.50, 2.75, 3.00]
    elif scale_inches == 5:
        col_levels_in = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40,
                         0.50, 0.60, 0.80, 1.00, 1.25, 1.50, 1.75, 2.00,
                         2.25, 2.50, 2.75, 3.00, 3.50, 4.00, 4.50, 5.00]
    elif scale_inches == 8:
        col_levels_in = [0.01, 0.05, 0.10, 0.15, 0.20, 0.40, 0.60, 0.80,
                         1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00, 3.50,
                         4.00, 4.50, 5.00, 5.50, 6.00, 6.50, 7.00, 8.00]
    elif scale_inches == 12:
        col_levels_in = [0.01, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00,
                         1.25, 1.50, 1.75, 2.00, 2.50, 3.00, 3.50, 4.00,
                         5.00, 6.00, 7.00, 8.00, 9.00, 10.0, 11.0, 12.0]
    elif scale_inches == 16:
        col_levels_in = [0.01, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00,
                         1.25, 1.50, 2.00, 2.50, 3.00, 3.50, 4.00, 5.00,
                         6.00, 7.00, 8.00, 9.00, 10.0, 12.0, 14.0, 16.0]
    elif scale_inches == 24:
        col_levels_in = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50,
                         2.00, 2.50, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00,
                         9.00, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 24.0]
    else:
        msg = f'scale_inches value of {scale_inches} is not supported.'
        logger.error(msg)

    tick_labels = [f'{f}' for f in col_levels_in]

    # Add a value for the upper bound of the last value.
    #6/16 col_levels_in.append(1.0e8)
    #6/16 tick_labels.append('')

    norm = mplcol.BoundaryNorm(col_levels_in, len(col_levels_in) - 1)
    mrms_color_ramp = {'colormap': mrms_colormap,
                       'col_levels': col_levels_in,
                       'tick_levels': col_levels_in,
                       'tick_labels': tick_labels,
                       'norm': norm,
                       'extend': 'max'}

    return mrms_color_ramp


def byr_hourly_precip_mm_colors():
    '''
    Define a color ramp for precipitation based on the colorbrewer 2.0 9-class
    RdYlBu color ramp, but with blue first so I call it "BuYlRd". Like
    dbz_hourly_precip_mm_colors_hack(), this color ramp tops out at 8 inches.
    '''

    # First four blue from 8-class RdYlBu, from light to dark, then four
    # yellow/red from 7-class RdYlBu.
    red = [255, 224, 171, 116, 69, 255, 254, 252, 215]
    grn = [228, 243, 217, 173, 117, 255, 224, 141, 48]
    blu = [220, 248, 233, 209, 180, 191, 144, 89, 39]

    alp = [1.0] * len(red)

    red = np.array(red, dtype=np.float64) / 255.0
    grn = np.array(grn, dtype=np.float64) / 255.0
    blu = np.array(blu, dtype=np.float64) / 255.0
    rgb = np.transpose(np.array([red,grn,blu,alp]))

    byr_colormap = mplcol.ListedColormap(rgb, name='BuYlRd')
    byr_colormap.set_over(color='#853abeff') # 133, 58, 190
    byr_colormap.set_over(color='#ff9fffff')

    col_levels_in = [0.0, 1.0e-8, 0.01, 0.1, 0.25, 0.5, 1.0, 2.0,
                     4.0, 8.0]
    tick_labels_in = ['', '0', 'trace', '0.1', '0.25', '0.5', '1',
                      '2', '4', '8']

    col_levels_mm = [i * 25.4 for i in col_levels_in]
    tick_levels_mm = col_levels_mm
    norm = mplcol.BoundaryNorm(col_levels_mm, len(col_levels_mm) - 1)
    byr_color_ramp = {'colormap': byr_colormap,
                      'col_levels': col_levels_mm,
                      'tick_levels': tick_levels_mm,
                      'tick_labels': tick_labels_in,
                      'norm': norm,
                      'extend': 'max'}

    return byr_color_ramp


def delta_swe_m():
    '''
    Color ramp for observed - modeled SWE, taken directly from swe_nudge.ct
    in the GISRS system. Recommended units are meters, with displayed tick labels in millimeters. Formerly greg_geo.delta_swe_m_color_ramp.
    '''
    red = [152, 207, 246, 255, 255, 255, 255,
           230,
           190, 126, 23, 0, 0, 75, 100]
    grn = [0, 0, 113, 169, 216, 255, 255,
           230,
           255, 255, 216, 169, 113, 0, 0]
    blu = [100, 75, 0, 0, 23, 126, 190,
           255,
           255, 255, 255, 255, 246, 246, 152]
    alp = [1.0] * len(red)

    red = np.array(red, dtype=np.float64) / 255.0
    grn = np.array(grn, dtype=np.float64) / 255.0
    blu = np.array(blu, dtype=np.float64) / 255.0
    alp = np.array(alp, dtype=np.float64)

    col_levels = [-10.0, -0.5, -0.25, -0.1, -0.05, -0.01, -0.005, -0.001,
                  0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 10.0]
    tick_labels = ['', '-500', '-250', '-100', '-50', '-10', '-5', '-1',
                   '1', '5', '10', '50', '100', '250', '500', '']

    # Steal the first and last colors for set_under and set_over.
    rgba_bottom = (red[0], grn[0], blu[0], alp[0])
    rgba_top = (red[-1], grn[-1], blu[-1], alp[-1])
    red = red[1:-1]
    grn = grn[1:-1]
    blu = blu[1:-1]
    alp = alp[1:-1]

    rgba = np.transpose(np.array([red, grn, blu, alp]))

    col_levels = col_levels[1:-1]
    tick_labels = tick_labels[1:-1]

    colormap = mplcol.ListedColormap(rgba, name='delta_swe')

    colormap.set_under(rgba_bottom)
    colormap.set_over(rgba_top)

    tick_levels = col_levels
    norm = mplcol.BoundaryNorm(col_levels, len(col_levels) - 1)

    color_ramp = {'colormap': colormap,
                  'col_levels': col_levels,
                  'tick_levels': tick_levels,
                  'tick_labels': tick_labels,
                  'norm': norm,
                  'extend': 'both'}

    return color_ramp


def delta_swe_mm():
    '''
    Color ramp for observed - modeled SWE, taken directly from swe_nudge.ct
    in the GISRS system. Recommended units are millimeters, with displayed
    tick labels in millimeters. This is meant to be more sane than
    delta_swe_m, which makes things very confusing by expecting values in
    meters and tick labels in millimeters.
    '''
    red = [152, 207, 246, 255, 255, 255, 255,
           230,
           190, 126, 23, 0, 0, 75, 100]
    grn = [0, 0, 113, 169, 216, 255, 255,
           230,
           255, 255, 216, 169, 113, 0, 0]
    blu = [100, 75, 0, 0, 23, 126, 190,
           255,
           255, 255, 255, 255, 246, 246, 152]
    alp = [1.0] * len(red)

    red = np.array(red, dtype=np.float64) / 255.0
    grn = np.array(grn, dtype=np.float64) / 255.0
    blu = np.array(blu, dtype=np.float64) / 255.0
    alp = np.array(alp, dtype=np.float64)

    col_levels = [-1.0e4, -500.0, -250.0, -100.0, -50.0, -10.0, 
                  -5.0, -1.0, 1.0, 5.0,
                  10.0, 50.0, 100.0, 250.0, 500.0, 10000.0]
    tick_labels = ['', '-500', '-250', '-100', '-50', '-10', '-5', '-1',
                   '1', '5', '10', '50', '100', '250', '500', '']

    # Steal the first and last colors for set_under and set_over.
    rgba_bottom = (red[0], grn[0], blu[0], alp[0])
    rgba_top = (red[-1], grn[-1], blu[-1], alp[-1])
    red = red[1:-1]
    grn = grn[1:-1]
    blu = blu[1:-1]
    alp = alp[1:-1]

    rgba = np.transpose(np.array([red, grn, blu, alp]))

    col_levels = col_levels[1:-1]
    tick_labels = tick_labels[1:-1]

    colormap = mplcol.ListedColormap(rgba, name='delta_swe')

    colormap.set_under(rgba_bottom)
    colormap.set_over(rgba_top)

    tick_levels = col_levels
    norm = mplcol.BoundaryNorm(col_levels, len(col_levels) - 1)

    color_ramp = {'colormap': colormap,
                  'col_levels': col_levels,
                  'tick_levels': tick_levels,
                  'tick_labels': tick_labels,
                  'norm': norm,
                  'extend': 'both'}

    return color_ramp


def elev_m(max_elev=6000.0):
    '''
    Color ramp for terrain, generally elevation in meters. Uses a single blue
    for zero, then jumps a bit into the greens.
    '''
    rgba = np.vstack((mpl.cm.terrain(20),
                      mpl.cm.terrain(np.linspace(0.25, 1.0, 255))))
    land_colormap = mplcol.ListedColormap(rgba, name='land terrain')
    col_levels = np.insert(np.linspace(0.0, max_elev, 255), 0, 0.0)
    col_levels[1] = 1.0e-6
    tick_levels = np.linspace(0.0, max_elev, 7)
    tick_labels = [f'{i:.0f}' for i in tick_levels]
    #tick_levels = [0.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0]
    #tick_labels = ['0', '1000', '2000', '3000', '4000', '5000', '6000+']
    norm = mplcol.BoundaryNorm(col_levels, 256)
    terrain_color_ramp = {'colormap': land_colormap,
                          'col_levels': col_levels,
                          'tick_levels': tick_levels,
                          'tick_labels': tick_labels,
                          'norm': norm,
                          'extend': 'max'}
    return terrain_color_ramp


def terrain_m_plasma_ramp():
    '''
    Discretized plasma-based ramp for terrain, generally elevation in meters.
    '''
    # rgba = np.vstack((mpl.cm.terrain(20),
    #                   mpl.cm.terrain(np.linspace(0.25, 1.0, 255))))
    # rgba = mpl.cm.rainbow(np.linspace(0, 1, 256))
    # #print(len(rgba))
    # land_colormap = mplcol.ListedColormap(rgba, name='land terrain')
    # # col_levels = np.insert(np.linspace(0.0, 6000.0, 255), 0, 0.0)
    # # col_levels[1] = 1.0e-6
    # col_levels = np.linspace(0.0, 2000.0, 256)
    # tick_levels = [0.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0]
    # tick_labels = ['0', '1000', '2000', '3000', '4000', '5000', '6000+']
    # tick_levels = [0.0, 250.0, 500.0, 750.0, 1000.0,
    #                1250.0, 1500.0, 1750.0, 2000.0]
    # tick_labels = ['0', '250', '500', '750', '1000',
    #                '1250', '1500', '1750', '2000']

    rgba = mpl.cm.plasma(np.linspace(0, 1, 256))
    land_colormap = mplcol.ListedColormap(rgba, name='land terrain')
    num_colors = 20
    min_terrain_val = 0.0
    max_terrain_val = 2000.0
    color_range = (max_terrain_val - min_terrain_val) / num_colors
    col_levels = np.linspace(min_terrain_val, max_terrain_val, num_colors)
    tick_levels = [i * color_range for i in range(0, num_colors+1)]
    tick_labels = [f'{i:.0f}' for i in tick_levels]
    norm = mplcol.BoundaryNorm(col_levels, 256)
    terrain_color_ramp = {'colormap': land_colormap,
                          'col_levels': col_levels,
                          'tick_levels': tick_levels,
                          'tick_labels': tick_labels,
                          'norm': norm,
                          'extend': 'max'}
    return terrain_color_ramp


def delta_water_in_nohrsc_color_ramp():
    '''
    Color ramp for observed - modeled water (precipitation snow, SWE,
    whatever) in inches, taken directly from swe_nudge.ct in the GISRS system.
    '''
    red = [152, 207, 246, 255, 255, 255, 255,
           230,
           190, 126, 23, 0, 0, 75, 100]
    grn = [0, 0, 113, 169, 216, 255, 255,
           230,
           255, 255, 216, 169, 113, 0, 0]
    blu = [100, 75, 0, 0, 23, 126, 190,
           255,
           255, 255, 255, 255, 246, 246, 152]
    alp = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
           1.0, 1.0]
    red = np.array(red, dtype=np.float64) / 255.0
    grn = np.array(grn, dtype=np.float64) / 255.0
    blu = np.array(blu, dtype=np.float64) / 255.0
    alp = np.array(alp)

    col_levels = [-24.0, -18.0, -12.0, -6.0, -2.0, -1.0, -0.5, -0.25,
                  0.25, 0.5, 1.0, 2.0, 6.0, 12.0, 18.0, 24.0]
    tick_labels = ['-24', '-18', '-12', '-6', '-2', '-1', '-0.5', '-0.25',
                   '0.25', '0.5', '1', '2', '6', '12', '18', '24']

    # # Initial values of col_levels_in represent the lower bound of the range
    # # represented by each RGB value.
    # if scale_inches == 3:
    #     col_levels_in = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
    #                      0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.20,
    #                      1.40, 1.60, 1.80, 2.00, 2.25, 2.50, 2.75, 3.00]
    # elif scale_inches == 5:
    #     col_levels_in = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40,
    #                      0.50, 0.60, 0.80, 1.00, 1.25, 1.50, 1.75, 2.00,
    #                      2.25, 2.50, 2.75, 3.00, 3.50, 4.00, 4.50, 5.00]
    # elif scale_inches == 8:
    #     col_levels_in = [0.01, 0.05, 0.10, 0.15, 0.20, 0.40, 0.60, 0.80,
    #                      1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00, 3.50,
    #                      4.00, 4.50, 5.00, 5.50, 6.00, 6.50, 7.00, 8.00]
    # elif scale_inches == 12:
    #     col_levels_in = [0.01, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00,
    #                      1.25, 1.50, 1.75, 2.00, 2.50, 3.00, 3.50, 4.00,
    #                      5.00, 6.00, 7.00, 8.00, 9.00, 10.0, 11.0, 12.0]
    # elif scale_inches == 16:
    #     col_levels_in = [0.01, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00,
    #                      1.25, 1.50, 2.00, 2.50, 3.00, 3.50, 4.00, 5.00,
    #                      6.00, 7.00, 8.00, 9.00, 10.0, 12.0, 14.0, 16.0]
    # elif scale_inches == 24:
    #     col_levels_in = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50,
    #                      2.00, 2.50, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00,
    #                      9.00, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 24.0]
    # else:
    #     logger.error(f'scale_inches value of {scale_inches} '  +
    #                  'is not supported.')

    # Steal the first and last colors for set_under and set_over.
    rgba_bottom = (red[0], grn[0], blu[0], alp[0])
    rgba_top = (red[-1], grn[-1], blu[-1], alp[-1])

    red = red[1:-1]
    grn = grn[1:-1]
    blu = blu[1:-1]
    alp = alp[1:-1]
    rgba = np.transpose(np.array([red, grn, blu, alp]))
    col_levels = col_levels[1:-1]
    tick_labels = tick_labels[1:-1]
    delta_water_colormap = mplcol.ListedColormap(rgba, name='delta_water')
    delta_water_colormap.set_under(rgba_bottom)
    delta_water_colormap.set_over(rgba_top)

    tick_levels = col_levels
    norm = mplcol.BoundaryNorm(col_levels, len(col_levels) - 1)

    delta_water_color_ramp = {'colormap': delta_water_colormap,
                              'col_levels': col_levels,
                              'tick_levels': tick_levels,
                              'tick_labels': tick_labels,
                              'norm': norm,
                              'extend': 'both'}

    return delta_water_color_ramp


def delta_water_in_cb11_color_ramp(scale_inches=24,
                                   middle_gray=191,
                                   use_alt_colors=False):
    '''
    Color ramp for observed - modeled water (precipitation snow, SWE,
    whatever) in inches, taken directly from 11-class RdYlBu at
    colorbrewer2.org, but with the middle color neutralized.
    '''

    logger = logging.getLogger()

    red = [165, 215, 244, 253, 254,
           middle_gray,
           224, 171, 116, 69, 49]
    grn = [0, 48, 109, 174, 224,
           middle_gray,
           243, 217, 173, 117, 54]
    blu = [38, 39, 67, 97, 144,
           middle_gray,
           248, 233, 209, 180, 149]

    if use_alt_colors:
        red = [191, 204, 234, 249, 255, 210, 178,  62,  23,  62, 191]
        grn = [152,  51, 129, 203, 255, 210, 255, 218, 120,   0, 114]
        blu = [114,   0,  23,  62, 178, 210, 255, 249, 234, 204, 191]

    alp = [1.0] * len(red)

    red = np.array(red, dtype=np.float64) / 255.0
    grn = np.array(grn, dtype=np.float64) / 255.0
    blu = np.array(blu, dtype=np.float64) / 255.0
    alp = np.array(alp)

    col_levels = [-24.0, -12.0, -6.0, -2.0, -1.0, -0.5,
                  0.5, 1.0, 2.0, 6.0, 12.0, 24.0]
    # tick_labels = ['-24', '-12', '-6', '-2', '-1', '-0.5',
    #                '0.5', '1', '2', '6', '12', '24']
    if scale_inches == 0.5:
        col_levels = [-0.5, -0.4, -0.3, -0.2, -0.1, -0.05,
                      0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        # col_levels = [-0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
        #               0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        # tick_labels = ['-0.6', '-0.5', '-0.4', '-0.3', '-0.2', '-0.1',
        #                '0.1', '0.2', '0.3', '0.4', '0.5', '0.6']
    elif scale_inches == 1:
        col_levels = [-1.25, -1.0, -0.75, -0.5, -0.25, -0.1,
                      0.1, 0.25, 0.5, 0.75, 1.0, 1.25]
        # col_levels = [-2.0, -1.0, -0.75, -0.5, -0.25, -0.1,
        #               0.1, 0.25, 0.5, 0.75, 1.0, 2.0]
        # tick_labels = ['-2', '-1', '-0.75', '-0.5', '-0.25', '-0.1',
        #                '0.1', '0.25', '0.5', '0.75', '1.0', '2.0']
    elif scale_inches == 2:
        col_levels = [-2.0, -1.0, -0.75, -0.5, -0.25, -0.1,
                      0.1, 0.25, 0.5, 0.75, 1.0, 2.0]
    elif scale_inches == 3:
        col_levels = [-3.0, -2.0, -1.0, -0.5, -0.25, -0.1,
                      0.1, 0.25, 0.5, 1.0, 2.0, 3.0]
    elif scale_inches == 4:
        col_levels = [-4.0, -3.0, -2.0, -1.0, -0.5, -0.25,
                      0.25, 0.5, 1.0, 2.0, 3.0, 4.0]
    elif scale_inches == 6:
        col_levels = [-6.0, -4.0, -2.0, -1.0, -0.5, -0.25,
                       0.25, 0.5, 1.0, 2.0, 4.0, 6.0]
        tick_labels = ['-6', '-4', '-2', '-1', '-0.5', '0.25',
                       '0.25', '0.5', '1', '2', '4', '6']
    elif scale_inches == 24:
        col_levels = [-24.0, -12.0, -6.0, -2.0, -1.0, -0.5,
                      0.5, 1.0, 2.0, 6.0, 12.0, 24.0]
        tick_labels = ['-24', '-12', '-6', '-2', '-1', '-0.5',
                       '0.5', '1', '2', '6', '12', '24']
    else:
        msg = f'scale_inches value of {scale_inches} is not supported.'
        logger.error(msg)

    if 'tick_labels' not in locals():
        tick_labels = [format_float(lev) for lev in col_levels]


    # if scale_inches == 3:
    #     col_levels_in = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
    #                      0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.20,
    #                      1.40, 1.60, 1.80, 2.00, 2.25, 2.50, 2.75, 3.00]
    # elif scale_inches == 5:
    #     col_levels_in = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40,
    #                      0.50, 0.60, 0.80, 1.00, 1.25, 1.50, 1.75, 2.00,
    #                      2.25, 2.50, 2.75, 3.00, 3.50, 4.00, 4.50, 5.00]
    # elif scale_inches == 8:
    #     col_levels_in = [0.01, 0.05, 0.10, 0.15, 0.20, 0.40, 0.60, 0.80,
    #                      1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00, 3.50,
    #                      4.00, 4.50, 5.00, 5.50, 6.00, 6.50, 7.00, 8.00]
    # elif scale_inches == 12:
    #     col_levels_in = [0.01, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00,
    #                      1.25, 1.50, 1.75, 2.00, 2.50, 3.00, 3.50, 4.00,
    #                      5.00, 6.00, 7.00, 8.00, 9.00, 10.0, 11.0, 12.0]
    # elif scale_inches == 16:
    #     col_levels_in = [0.01, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00,
    #                      1.25, 1.50, 2.00, 2.50, 3.00, 3.50, 4.00, 5.00,
    #                      6.00, 7.00, 8.00, 9.00, 10.0, 12.0, 14.0, 16.0]
    # elif scale_inches == 24:
    #     col_levels_in = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50,
    #                      2.00, 2.50, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00,
    #                      9.00, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 24.0]
    # else:
    #     logger.error(f'scale_inches value of {scale_inches} '  +
    #                  'is not supported.')

    # Steal the first and last colors for set_under and set_over.
    rgba_bottom = (red[0], grn[0], blu[0], alp[0])
    rgba_top = (red[-1], grn[-1], blu[-1], alp[-1])
    red = red[1:-1]
    grn = grn[1:-1]
    blu = blu[1:-1]
    alp = alp[1:-1]
    rgba = np.transpose(np.array([red, grn, blu, alp]))
    col_levels = col_levels[1:-1]
    tick_labels = tick_labels[1:-1]
    # print(col_levels)
    # print(tick_labels)
    delta_water_colormap = mplcol.ListedColormap(rgba, name='delta_water')
    delta_water_colormap.set_under(rgba_bottom)
    delta_water_colormap.set_over(rgba_top)

    tick_levels = col_levels
    norm = mplcol.BoundaryNorm(col_levels, len(col_levels) - 1)

    delta_water_color_ramp = {'colormap': delta_water_colormap,
                              'col_levels': col_levels,
                              'tick_levels': tick_levels,
                              'tick_labels': tick_labels,
                              'norm': norm,
                              'extend': 'both'}

    return delta_water_color_ramp


def elev_ncl_oceanlakelandsnow(max_elev=6000.0):
    '''
    NCL color ramp for terrain, generally elevation in meters.
    See https://www.ncl.ucar.edu/Document/Graphics/color_table_gallery.shtml
    and
    https://github.com/samwisehawkins/nclcmaps
    I got these RGB values by downloading OceanLakeLandSnow.rgb from the
    above-linked gallery.
    Then you can just do this:
    >>> import numpy as np
    >>> from pprint import pprint
    >>> rgb = np.loadtxt('OceanLakeLandSnow.rgb', comments=('#', 'ncolors'))
    >>> pprint(rgb)
    '''
    rgb = np.array([[ 24., 116., 205.],
                    [135., 206., 255.],
                    [ 38., 103.,  40.],
                    [ 42., 107.,  40.],
                    [ 46., 110.,  41.],
                    [ 50., 114.,  41.],
                    [ 54., 117.,  41.],
                    [ 58., 121.,  42.],
                    [ 62., 124.,  42.],
                    [ 66., 128.,  42.],
                    [ 70., 131.,  43.],
                    [ 74., 135.,  43.],
                    [ 78., 138.,  43.],
                    [ 82., 142.,  44.],
                    [ 86., 145.,  44.],
                    [ 90., 149.,  44.],
                    [ 94., 152.,  45.],
                    [ 98., 156.,  45.],
                    [102., 159.,  46.],
                    [106., 163.,  46.],
                    [110., 166.,  46.],
                    [114., 170.,  47.],
                    [118., 173.,  47.],
                    [122., 177.,  47.],
                    [126., 180.,  48.],
                    [130., 184.,  48.],
                    [134., 187.,  48.],
                    [138., 191.,  49.],
                    [142., 194.,  49.],
                    [146., 198.,  49.],
                    [154., 205.,  50.],
                    [155., 206.,  52.],
                    [156., 207.,  54.],
                    [158., 208.,  56.],
                    [159., 210.,  57.],
                    [160., 211.,  59.],
                    [161., 212.,  61.],
                    [162., 213.,  63.],
                    [163., 214.,  65.],
                    [165., 215.,  67.],
                    [166., 216.,  69.],
                    [167., 218.,  70.],
                    [168., 219.,  72.],
                    [169., 220.,  74.],
                    [170., 221.,  76.],
                    [172., 222.,  78.],
                    [173., 223.,  80.],
                    [174., 224.,  82.],
                    [175., 225.,  84.],
                    [176., 227.,  85.],
                    [177., 228.,  87.],
                    [179., 229.,  89.],
                    [180., 230.,  91.],
                    [181., 231.,  93.],
                    [182., 232.,  95.],
                    [183., 233.,  97.],
                    [184., 235.,  98.],
                    [186., 236., 100.],
                    [188., 238., 104.],
                    [190., 238., 107.],
                    [192., 238., 111.],
                    [194., 239., 114.],
                    [197., 239., 117.],
                    [199., 239., 121.],
                    [201., 239., 124.],
                    [203., 240., 127.],
                    [205., 240., 130.],
                    [207., 240., 134.],
                    [209., 240., 137.],
                    [212., 241., 140.],
                    [214., 241., 144.],
                    [216., 241., 147.],
                    [218., 241., 150.],
                    [220., 242., 154.],
                    [222., 242., 157.],
                    [224., 242., 160.],
                    [226., 242., 164.],
                    [229., 243., 167.],
                    [231., 243., 170.],
                    [233., 243., 174.],
                    [235., 243., 177.],
                    [237., 244., 180.],
                    [239., 244., 183.],
                    [241., 244., 187.],
                    [244., 244., 190.],
                    [246., 245., 193.],
                    [250., 245., 200.],
                    [250., 244., 199.],
                    [249., 244., 197.],
                    [249., 243., 196.],
                    [249., 243., 194.],
                    [248., 242., 193.],
                    [248., 242., 191.],
                    [248., 241., 190.],
                    [247., 241., 188.],
                    [247., 240., 187.],
                    [247., 239., 186.],
                    [246., 239., 184.],
                    [246., 238., 183.],
                    [246., 238., 181.],
                    [245., 237., 180.],
                    [245., 237., 178.],
                    [244., 236., 177.],
                    [244., 236., 175.],
                    [244., 235., 174.],
                    [243., 235., 172.],
                    [243., 234., 171.],
                    [243., 233., 170.],
                    [242., 233., 168.],
                    [242., 232., 167.],
                    [242., 232., 165.],
                    [241., 231., 164.],
                    [241., 231., 162.],
                    [241., 230., 161.],
                    [240., 229., 158.],
                    [239., 228., 155.],
                    [238., 226., 153.],
                    [237., 225., 150.],
                    [236., 224., 148.],
                    [235., 222., 145.],
                    [234., 221., 142.],
                    [233., 220., 140.],
                    [233., 219., 137.],
                    [232., 217., 135.],
                    [231., 216., 132.],
                    [230., 215., 130.],
                    [229., 213., 127.],
                    [228., 212., 124.],
                    [227., 211., 122.],
                    [226., 209., 119.],
                    [225., 208., 117.],
                    [224., 207., 114.],
                    [223., 205., 111.],
                    [222., 204., 109.],
                    [221., 203., 106.],
                    [220., 201., 104.],
                    [220., 200., 101.],
                    [219., 199.,  99.],
                    [218., 198.,  96.],
                    [217., 196.,  93.],
                    [216., 195.,  91.],
                    [213., 191.,  83.],
                    [213., 190.,  81.],
                    [212., 189.,  79.],
                    [212., 187.,  77.],
                    [212., 186.,  76.],
                    [212., 185.,  74.],
                    [211., 184.,  72.],
                    [211., 182.,  70.],
                    [211., 181.,  68.],
                    [211., 180.,  66.],
                    [210., 179.,  64.],
                    [210., 177.,  63.],
                    [210., 176.,  61.],
                    [209., 175.,  59.],
                    [209., 174.,  57.],
                    [209., 172.,  55.],
                    [209., 171.,  53.],
                    [208., 170.,  51.],
                    [208., 169.,  49.],
                    [208., 167.,  48.],
                    [207., 166.,  46.],
                    [207., 165.,  44.],
                    [207., 164.,  42.],
                    [207., 162.,  40.],
                    [206., 161.,  38.],
                    [206., 160.,  36.],
                    [206., 159.,  35.],
                    [206., 157.,  33.],
                    [205., 155.,  29.],
                    [203., 153.,  29.],
                    [200., 151.,  30.],
                    [198., 148.,  30.],
                    [196., 146.,  31.],
                    [194., 144.,  31.],
                    [191., 142.,  32.],
                    [189., 139.,  32.],
                    [187., 137.,  33.],
                    [185., 135.,  33.],
                    [182., 133.,  34.],
                    [180., 130.,  34.],
                    [178., 128.,  35.],
                    [175., 126.,  35.],
                    [173., 124.,  36.],
                    [171., 121.,  36.],
                    [169., 119.,  37.],
                    [166., 117.,  37.],
                    [164., 115.,  38.],
                    [162., 112.,  38.],
                    [159., 110.,  39.],
                    [157., 108.,  39.],
                    [155., 106.,  40.],
                    [153., 103.,  40.],
                    [150., 101.,  41.],
                    [148.,  99.,  41.],
                    [146.,  97.,  42.],
                    [144.,  94.,  42.],
                    [139.,  90.,  43.],
                    [137.,  89.,  42.],
                    [135.,  88.,  42.],
                    [133.,  86.,  41.],
                    [131.,  85.,  41.],
                    [129.,  84.,  40.],
                    [127.,  83.,  39.],
                    [125.,  81.,  39.],
                    [124.,  80.,  38.],
                    [122.,  79.,  38.],
                    [120.,  78.,  37.],
                    [118.,  76.,  37.],
                    [116.,  75.,  36.],
                    [114.,  74.,  35.],
                    [112.,  73.,  35.],
                    [110.,  71.,  34.],
                    [108.,  70.,  34.],
                    [106.,  69.,  33.],
                    [104.,  68.,  32.],
                    [102.,  66.,  32.],
                    [100.,  65.,  31.],
                    [ 98.,  64.,  31.],
                    [ 97.,  63.,  30.],
                    [ 95.,  61.,  30.],
                    [ 93.,  60.,  29.],
                    [ 91.,  59.,  28.],
                    [ 89.,  58.,  28.],
                    [ 87.,  56.,  27.],
                    [ 83.,  54.,  26.],
                    [ 89.,  61.,  34.],
                    [ 94.,  67.,  42.],
                    [100.,  74.,  50.],
                    [106.,  81.,  58.],
                    [111.,  87.,  65.],
                    [117.,  94.,  73.],
                    [123., 101.,  81.],
                    [129., 108.,  89.],
                    [134., 114.,  97.],
                    [140., 121., 105.],
                    [146., 128., 113.],
                    [151., 134., 121.],
                    [157., 141., 129.],
                    [163., 148., 137.],
                    [168., 154., 144.],
                    [174., 161., 152.],
                    [180., 168., 160.],
                    [185., 174., 168.],
                    [191., 181., 176.],
                    [197., 188., 184.],
                    [202., 194., 192.],
                    [208., 201., 200.],
                    [214., 208., 208.],
                    [220., 215., 216.],
                    [225., 221., 223.],
                    [231., 228., 231.],
                    [237., 235., 239.],
                    [248., 248., 255.]])

    colormap = mplcol.ListedColormap(rgb, name='OceanLakeLandSnow')

    col_levels = np.insert(np.linspace(0.0, max_elev, colormap.N - 1),
                           0, 0.0)
    col_levels[1] = 1.0e-6
    tick_levels = np.linspace(0.0, max_elev, 7)
    tick_labels = [f'{i:.0f}' for i in tick_levels]
    norm = mplcol.BoundaryNorm(col_levels, len(col_levels) - 1)

    color_ramp = {'colormap': colormap,
                  'col_levels': col_levels,
                  'tick_levels': tick_levels,
                  'tick_labels': tick_labels,
                  'norm': norm,
                  'extend': 'both'}

    return color_ramp


def water_fit_percent_error_cb11_color_ramp(middle_gray=191,
                                            use_alt_colors=False):
    """
    Color ramp for a percent errror in an estimate of water, intended to be
    roughly lineat on a logarithmic scale through the middle section. This
    ramp is meant to approach its extremities beyond -0.33 / +0.5, so it
    gives a good illustration of the multiplicative bias of an attempt to be
    correct, rather than something like a normal.
    """

    logger = logging.getLogger()

    red = [165, 215, 244, 253, 254,
           middle_gray,
           224, 171, 116, 69, 49]
    grn = [0, 48, 109, 174, 224,
           middle_gray,
           243, 217, 173, 117, 54]
    blu = [38, 39, 67, 97, 144,
           middle_gray,
           248, 233, 209, 180, 149]

    if use_alt_colors:
        red = [191, 204, 234, 249, 255, 210, 178,  62,  23,  62, 191]
        grn = [152,  51, 129, 203, 255, 210, 255, 218, 120,   0, 114]
        blu = [114,   0,  23,  62, 178, 210, 255, 249, 234, 204, 191]

    alp = [1.0] * len(red)

    red = np.array(red, dtype=np.float64) / 255.0
    grn = np.array(grn, dtype=np.float64) / 255.0
    blu = np.array(blu, dtype=np.float64) / 255.0
    alp = np.array(alp)

    col_levels = [1/3, 1/2, 3/4, 0.85, 0.9, 0.95,
                  1.05, 1.1, 1.15, 4/3, 2.0, 3.0]
    col_levels = [1/3, 1/2, 2/3, 3/4, 0.85, 0.95,
                  1.05, 1.15, 4/3, 3/2, 2.0, 3.0]
    col_levels = [0.33, 0.5, 0.67, 0.75, 0.85, 0.95,
                  1.05, 1.15, 1.33, 1.5, 2.0, 3.0]
    col_levels = [(i - 1) * 100 for i in col_levels]
    tick_labels = [format_float(lev) for lev in col_levels]

    # Steal the first and last colors for set_under and set_over.
    rgba_bottom = (red[0], grn[0], blu[0], alp[0])
    rgba_top = (red[-1], grn[-1], blu[-1], alp[-1])
    red = red[1:-1]
    grn = grn[1:-1]
    blu = blu[1:-1]
    alp = alp[1:-1]
    rgba = np.transpose(np.array([red, grn, blu, alp]))
    col_levels = col_levels[1:-1]
    tick_labels = tick_labels[1:-1]
    percent_water_colormap = mplcol.ListedColormap(rgba, name='percent_water')
    percent_water_colormap.set_under(rgba_bottom)
    percent_water_colormap.set_over(rgba_top)

    tick_levels = col_levels
    norm = mplcol.BoundaryNorm(col_levels, len(col_levels) - 1)

    percent_water_color_ramp = {'colormap': percent_water_colormap,
                                'col_levels': col_levels,
                                'tick_levels': tick_levels,
                                'tick_labels': tick_labels,
                                'norm': norm,
                                'extend': 'both'}

    return percent_water_color_ramp


def water_fit_percent_cb11_color_ramp(middle_gray=191,
                                      use_alt_colors=False):
    """
    Color ramp for a percent of water, intended to be roughly linear
    on a logarithmic scale through the middle section. This ramp is
    meant to approach its extremities beyond 0.67 / 1.5, so it gives a
    good illustration of the multiplicative bias of an attempt to be
    correct, rather than something like a normal.
    """

    logger = logging.getLogger()

    red = [165, 215, 244, 253, 254,
           middle_gray,
           224, 171, 116, 69, 49]
    grn = [0, 48, 109, 174, 224,
           middle_gray,
           243, 217, 173, 117, 54]
    blu = [38, 39, 67, 97, 144,
           middle_gray,
           248, 233, 209, 180, 149]

    if use_alt_colors:
        red = [191, 204, 234, 249, 255, 210, 178,  62,  23,  62, 191]
        grn = [152,  51, 129, 203, 255, 210, 255, 218, 120,   0, 114]
        blu = [114,   0,  23,  62, 178, 210, 255, 249, 234, 204, 191]

    alp = [1.0] * len(red)

    red = np.array(red, dtype=np.float64) / 255.0
    grn = np.array(grn, dtype=np.float64) / 255.0
    blu = np.array(blu, dtype=np.float64) / 255.0
    alp = np.array(alp)

    col_levels = [1/3, 1/2, 3/4, 0.85, 0.9, 0.95,
                  1.05, 1.1, 1.15, 4/3, 2.0, 3.0]
    col_levels = [1/3, 1/2, 2/3, 3/4, 0.85, 0.95,
                  1.05, 1.15, 4/3, 3/2, 2.0, 3.0]
    col_levels = [0.33, 0.5, 0.67, 0.75, 0.85, 0.95,
                  1.05, 1.15, 1.33, 1.5, 2.0, 3.0]
    col_levels = [i * 100 for i in col_levels]
    tick_labels = [format_float(lev) for lev in col_levels]

    # Steal the first and last colors for set_under and set_over.
    rgba_bottom = (red[0], grn[0], blu[0], alp[0])
    rgba_top = (red[-1], grn[-1], blu[-1], alp[-1])
    red = red[1:-1]
    grn = grn[1:-1]
    blu = blu[1:-1]
    alp = alp[1:-1]
    rgba = np.transpose(np.array([red, grn, blu, alp]))
    col_levels = col_levels[1:-1]
    tick_labels = tick_labels[1:-1]
    percent_water_colormap = mplcol.ListedColormap(rgba, name='percent_water')
    percent_water_colormap.set_under(rgba_bottom)
    percent_water_colormap.set_over(rgba_top)

    tick_levels = col_levels
    norm = mplcol.BoundaryNorm(col_levels, len(col_levels) - 1)

    percent_water_color_ramp = {'colormap': percent_water_colormap,
                                'col_levels': col_levels,
                                'tick_levels': tick_levels,
                                'tick_labels': tick_labels,
                                'norm': norm,
                                'extend': 'both'}

    return percent_water_color_ramp


