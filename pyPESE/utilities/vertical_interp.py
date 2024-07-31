#!/bin/python
'''
    FUNCTIONS TO VERTICALLY INTERPOLATE DATA

    Two kinds of interpolation:
    ---------------------------
    1)  Interpolation to pressure surfaces
        Can handle interpolation to below-ground pressure levels through
        the approach described by Trenberth et al 1993 ("Vertical Interpolation
        and Truncation of Model-Coordinate Data", NCAR Techical Note).
    
    2)  Interpolation to other surfaces
        Basically the inverse of the interpolation to pressure surfaces.
'''


import numpy as np
from numba import njit, float64 as nb_f64


flag_jit_cache = False









'''

'''











'''
    BASIC INTERPOLATION ONTO PRESSURE LEVELS
'''
@njit(  nb_f64[:,:,:]( nb_f64[:,:,:], nb_f64[:,:,:], nb_f64[:] ), cache=flag_jit_cache)
def basic_interpolate_to_pressure_levs( pres3d, data_arr3d, plvls1d):

    # Ascertain dimensions
    nx, ny, nz0 = pres3d.shape
    nz1 = len( plvls1d )

    # Init output array
    out_arr3d = np.zeros( (nx, ny, nz1), dtype='f8' )

    # Loop over horizontal dimensions (assumed to be the first 2 dimensions)
    for ix in range(nx):
        for iy in range(ny):
            out_arr3d[ix,iy,:] = np.interp( 
                np.log(plvls1d)*-1, np.log(pres3d[ix,iy,:])*-1, data_arr3d[ix,iy,:] 
            )
            # Minus sign added to ensure raw data monotonicity.

    return out_arr3d







'''
    Function to interpolate geopotential to specified pressure levels

    Uses linear interpolation for above-ground situations.
    Uses Trenberth et al (1993) Eq 15 for sub-surface interpolation
'''
@njit(
    nb_f64[:,:,:](
        nb_f64[:,:,:], nb_f64[:,:,:], nb_f64[:,:], nb_f64[:,:], nb_f64[:,:], nb_f64[:]
    ), cache=flag_jit_cache
)
def interp_geopotential_to_plvls( geopot3d, pres3d, psurf2d, tsurf2d, terrain2d, plvls1d ):

    # Extract dimensions
    nx, ny, nz = geopot3d.shape
    npres = len(plvls1d)

    # Useful constants
    GRAVITY_ACCEL = 9.80665 
    DRY_AIR_GAS_CONSTANT = 287.04 # J/K/kg
    
    # 0.286 * GRAVITY_ACCEL /DRY_AIR_GAS_CONSTANT #

    # Init array to hold interpolated geopotential fields
    out_geopot3d = np.empty( (nx, ny, npres), dtype='f8')
    
    # Log-pressure coordinates
    log_plvls1d = np.log( plvls1d ) * -1
    log_pres3d = np.log( pres3d )   * -1


    # loop over all locations
    for ix in range( nx ):
        for iy in range( ny ):

            # Special temperatures defined in Trenbeth Eq (13)
            T_o = tsurf2d[ix,iy] + 0.0065 * terrain2d[ix,iy]
            T_pl = min( T_o, 298 )

            # Determining lapse rate to use
            dry_lapse_rate = 0.0065 * DRY_AIR_GAS_CONSTANT / GRAVITY_ACCEL
        
            # Determine which levels are above ground, which levels are below ground
            flag_abv_ground = ( plvls1d  < psurf2d[ix,iy] )
            flag_blw_ground = ( plvls1d >= psurf2d[ix,iy] )

            # Interpolation for above-ground levels
            tmp = np.interp( 
                log_plvls1d[flag_abv_ground], log_pres3d[ix,iy,:], geopot3d[ix,iy,:]
            )
            out_geopot3d[ix,iy,flag_abv_ground] = tmp

            # Extrapolation for below-ground levels
            sub_plvls1d = plvls1d[flag_blw_ground]

            factor = dry_lapse_rate*np.log( sub_plvls1d/psurf2d[ix,iy] )
            out_geopot3d[ix,iy,flag_blw_ground] = (
                terrain2d[ix,iy] * GRAVITY_ACCEL
                -
                DRY_AIR_GAS_CONSTANT * tsurf2d[ix,iy] * np.log( sub_plvls1d/psurf2d[ix,iy] )
                    * ( 1 + 0.5*factor + (factor**2)/6. )
            )
        # --- End of loop over latitude-dimension
    # --- End of loop over longitude dimension

    return out_geopot3d















'''
    BASIC INTERPOLATION FROM PRESSURE LEVELS TO ETA LEVELS
'''
@njit(  nb_f64[:,:,:]( nb_f64[:], nb_f64[:,:,:], nb_f64[:,:,:] ), cache=flag_jit_cache)
def basic_interpolate_to_eta_levs( plvls1d, data_plvls, pres3d):

    # Ascertain dimensions
    nx, ny, nz0 = pres3d.shape

    # Init output array
    out_arr3d = np.zeros( (nx, ny, nz0), dtype='f8' )

    # Loop over horizontal dimensions (assumed to be the first 2 dimensions)
    for ix in range(nx):
        for iy in range(ny):
            out_arr3d[ix,iy,:] = np.interp( 
                np.log(pres3d[ix,iy,:])*-1, 
                np.log(plvls1d)*-1, data_plvls[ix,iy,:] 
            )
            # Minus sign added to ensure raw data monotonicity.

    return out_arr3d