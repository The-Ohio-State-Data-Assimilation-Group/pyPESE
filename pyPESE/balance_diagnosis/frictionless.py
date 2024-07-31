#!/bin/python3
'''
    PYTHON FUNCTIONS TO DIAGNOSE GEOPOTENTIAL FIELD CORRESPONDING TO A FRICTIONLESS
    INCOMPRESSIBLE BALANCED FLOW FIELD

    Written by: Man-Yau (Joseph) Chan

    DESCRIPTION:



    NOTES:
    -------
    1)  All derivative values on the corresponding boundaries of the NumPy arrays are likely crappy due to 
        the use of finite element schemes. 
            * d/dx values on the eastmost and westmost boundaries are crappy.
            * d/dy values on the northmost and southmost boundaries are crappy.
            * d2/dxdy values on the northmost, southmost, eastmost, westmost boundaries are crappy.


'''

import numpy as np
from numba import njit
from numba import float64
from numba.types import Tuple as nbtuple
from math import pi as PI
from time import time

# Load all functions relating to taking spatial derivatives & spherical padding
from pyPESE.utilities.global_latlon_grid import *

# Load functions relating to vertical interpolation
from pyPESE.utilities.vertical_interp import interp_geopotential_to_plvls, basic_interpolate_to_pressure_levs, basic_interpolate_to_eta_levs


t0 = time()

# flag for caching
jit_cache_flag = False
















'''
    FUNCTION TO COMPUTE SPATIAL GEOPOTENTIAL PERTURBATIONS THAT ARE CONSISTENT
    WITH FRICTIONLESS BALANCED 3D FLOW
    
    Note: Geopotential perturbations has a spatial mean of zero! Need to add 
    a constant value to it.

    Inputs:
    -------
    1) uwind3d (lon, lat, level)
            3D NumPy array of zonal wind velocities on PRESSURE SURFACES.
    2) vwind3d (lon, lat, level)
            3D NumPy array of meridional wind velocities on PRESSURE SURFACES.
    3) wwind3d (lon, lat, level)
            3D NumPy array of omega wind velocities on PRESSURE SURFACES.
    4) lon1d (lon)
            1D NumPy array of longitude values (in degrees).
    5) lat1d (lat)
            1D NumPy array of latitude values (in degrees). 
'''
def compute_geopotential_from_frictionless_3d_flow( uwind3d, vwind3d, wwind3d, lon1d, lat1d, plvls1d ):

    # Useful constants
    DEG_2_RAD = PI/180
    EARTH_ANGULAR_SPEED = 7.2921159e-5
    GRAVITY_ACCEL = 9.80665 

    # Obtain info about dimensions
    nlon, nlat, nlvl = uwind3d.shape

    # Generate coriolis parameter
    latmesh, lonmesh = np.meshgrid( lat1d, lon1d)
    coriolis_param3d = np.empty( uwind3d.shape, dtype='f8' )
    for k in range( nlvl ):
        coriolis_param3d[:,:,k] =  2 * EARTH_ANGULAR_SPEED * np.sin( DEG_2_RAD * latmesh )
    # --- End of loop over pressure levels

    # Pad stuff...
    plon1d, plat1d, pu3d = pad_field_due_to_spherical_symmetry( uwind3d, lon1d, lat1d)
    plon1d, plat1d, pv3d = pad_field_due_to_spherical_symmetry( vwind3d, lon1d, lat1d)

    # Generate cube of pressures
    pres3d = np.empty( (nlon, nlat, nlvl), dtype='f8' )
    for kk in range(nlvl):
        pres3d[:,:,kk] = plvls1d


    # Compute advection and coriolis terms for eastward momentum
    xterms = (
        # Advection terms
        uwind3d * compute_df_dx_on_eta_surface(pu3d, plon1d, plat1d)[1:-1,1:-1]
        + vwind3d * compute_df_dy_on_eta_surface(pu3d, plon1d, plat1d)[1:-1,1:-1]
        + wwind3d * compute_df_dP( pu3d, pres3d )
        # Coriolis term
        - coriolis_param3d * vwind3d
    )


    # Compute advection and coriolis terms for northward momentum
    yterms = (
        # Advection terms
        uwind3d * compute_df_dx_on_eta_surface(pv3d, plon1d, plat1d)[1:-1,1:-1]
        + vwind3d * compute_df_dy_on_eta_surface(pv3d, plon1d, plat1d)[1:-1,1:-1]
        + wwind3d * compute_df_dP( pv3d, pres3d )
        # Coriolis term
        + coriolis_param3d * uwind3d
    )

    # Compute convergence of forces
    plon1d, plat1d, pxterms = pad_field_due_to_spherical_symmetry( xterms, lon1d, lat1d)
    plon1d, plat1d, pyterms = pad_field_due_to_spherical_symmetry( yterms, lon1d, lat1d)
    conv_terms = (
        compute_df_dx_on_eta_surface( pxterms, plon1d, plat1d )[1:-1,1:-1,:]
        + compute_df_dy_on_eta_surface( pyterms, plon1d, plat1d )[1:-1,1:-1,:]
    ) * (-1)
    
    # Solve Poisson equation to obtain balanced geopotential's spatial perturbations
    zero_mean_geopot3d = spherical_invert_poisson_equation( conv_terms )
    zero_mean_geopot3d -= np.mean( np.mean( zero_mean_geopot3d, axis=0), axis=1 )

    # Return mean-zero geopotential
    return zero_mean_geopot3d 













'''
    FUNCTION TO COMPUTE BALANCED GEOPOTENTIAL ON ETA LEVELS

    Inputs:
    -------
    1) uwind3d (lon, lat, level)
            3D NumPy array of zonal wind velocities on ETA LEVELS.
    2) vwind3d (lon, lat, level)
            3D NumPy array of meridional wind velocities on ETA LEVELS.
    3) pres3d (lon, lat, level)
            3D NumPy array of pressure values on ETA LEVELS in UNITS OF PASCALS
    4) hgt3d (lon, lat, level)
            3D NumPy array of geopotential heights on ETA LEVELS.
    5) psurf2d (lon, lat)
            2D NumPy array of surface pressure in pascals
    6) tsurf2d (lon, lat)
            2D NumPy array of surface temperature in K
    7) terrain2d (lon, lat)
            2D NumPy array of terrain heights in meters
    8) lon1d (lon)
            1D NumPy array of longitude values (in degrees).
    9) lat1d (lat)
            1D NumPy array of latitude values (in degrees). 
'''
def compute_balanced_geopotential_from_frictionles_hydrostatic_flow_on_eta_lvls( uwind3d, vwind3d, pres3d, hgt3d, psurf2d, tsurf2d, terrain2d, lon1d, lat1d ):

    # Detect dimensions
    nlon, nlat, neta = uwind3d.shape
    nplvl = 201

    # Useful constants
    GRAVITY_ACCEL = 9.80665 
    
    # Pressure layers to do calculations on 
    plvls1d = np.linspace( pres3d.min(), 1.01e5, nplvl)[::-1]

    # Interpolate U & V to desired pressure levels
    uwind_plvl = basic_interpolate_to_pressure_levs( pres3d, uwind3d, plvls1d)
    vwind_plvl = basic_interpolate_to_pressure_levs( pres3d, vwind3d, plvls1d)

    # Interpolate geopotential to pressure levels
    geoopot_plvl = interp_geopotential_to_plvls( 
        hgt3d*GRAVITY_ACCEL, pres3d, psurf2d, tsurf2d, terrain2d, plvls1d
    )

    # Compute geopotnetial with mean zero
    mean_zero_geopot = compute_geopotential_from_frictionless_3d_flow( uwind_plvl, vwind_plvl, lon1d, lat1d, plvls1d )

    # Adjust the layerwise mean of the geooptential
    geopot_plvl = mean_zero_geopot + np.mean( np.mean( geoopot_plvl, axis=0), axis=1)

    # Interpolate geopotential from plvls back to eta lvls
    geopot3d = basic_interpolate_to_eta_levs( plvls1d, geopot_plvl, pres3d )


    return geopot3d







































'''
    SANITY CHECKS
'''
if __name__ == '__main__':

    print( 
        '\nTime spent on eager compilation of functions: %d seconds\n' 
        % (time()-t0)
    )

    SANITY_CHECK_pad_field_due_to_spherical_symmetry()

    SANITY_CHECK_geostrophic_flow_diagnosis()


    print('meow')