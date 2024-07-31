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
    Function for fast interpolation onto pressure levels
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





