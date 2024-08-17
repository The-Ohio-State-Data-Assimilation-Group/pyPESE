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
from scipy.ndimage import gaussian_filter

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
# @njit( float64[:,:,:]( float64[:,:,:], float64[:,:,:], float64[:,:,:], float64[:], float64[:], float64[:] ) )
def compute_geopotential_from_frictionless_3d_flow( uwind3d, vwind3d, wwind3d, lon1d, lat1d, plvls1d ):

    # Useful constants
    DEG_2_RAD = PI/180
    EARTH_ANGULAR_SPEED = 7.2921159e-5
    EARTH_RADIUS = 6378*1000
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
        pres3d[:,:,kk] = plvls1d[kk]

    # terms from the x-direction primitive equation
    xterms = (
        # Advection terms
        uwind3d * compute_df_dx_on_eta_surface(pu3d, plon1d, plat1d)[1:-1,1:-1]
        + vwind3d * compute_df_dy_on_eta_surface(pu3d, plon1d, plat1d)[1:-1,1:-1]
        + wwind3d * compute_df_dP( uwind3d, pres3d )
        # Coriolis term
        - coriolis_param3d * vwind3d
    )
    xterms *= -1


    # terms from the y-direction primitive equation
    yterms = (
        # Advection terms
        uwind3d * compute_df_dx_on_eta_surface(pv3d, plon1d, plat1d)[1:-1,1:-1]
        + vwind3d * compute_df_dy_on_eta_surface(pv3d, plon1d, plat1d)[1:-1,1:-1]
        + wwind3d * compute_df_dP( vwind3d, pres3d )
        # Coriolis term
        + coriolis_param3d * uwind3d
    )
    yterms *= -1


    # Generate divergence of the primitive equations
    plon1d, plat1d, pxterms = pad_field_due_to_spherical_symmetry( xterms, lon1d, lat1d )
    plon1d, plat1d, pyterms = pad_field_due_to_spherical_symmetry( yterms, lon1d, lat1d )
    div_terms = (
        compute_df_dx_on_eta_surface( pxterms, plon1d, plat1d )[1:-1,1:-1] 
        + compute_df_dy_on_eta_surface( pyterms, plon1d, plat1d )[1:-1,1:-1] 
    )

    # div_terms[:,50:130,:] -= 0.3e-9


    # Invert Poisson equation for geopotential
    geopot3d = spherical_invert_poisson_equation( div_terms )
    geopot3d -= np.mean( np.mean( geopot3d, axis=0 ), axis=0)

    # Return mean-zero geopotential
    return geopot3d

























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
# @njit( float64[:,:,:]( float64[:,:,:], float64[:,:,:], float64[:,:,:], float64[:], float64[:], float64[:] ) )
def OLD_compute_geopotential_from_frictionless_3d_flow( uwind3d, vwind3d, wwind3d, lon1d, lat1d, plvls1d ):

    # Useful constants
    DEG_2_RAD = PI/180
    EARTH_ANGULAR_SPEED = 7.2921159e-5
    EARTH_RADIUS = 6378*1000
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
        pres3d[:,:,kk] = plvls1d[kk]

    # terms from the x-direction primitive equation
    xterms = (
        # # Advection terms
        # uwind3d * compute_df_dx_on_eta_surface(pu3d, plon1d, plat1d)[1:-1,1:-1]
        # + vwind3d * compute_df_dy_on_eta_surface(pu3d, plon1d, plat1d)[1:-1,1:-1]
        # + wwind3d * compute_df_dP( uwind3d, pres3d )
        # Coriolis term
        - coriolis_param3d * vwind3d
    )
    xterms *= -1



    # Integrate away the x-derivative
    geopot3d_x  = np.zeros( (nlon, nlat, nlvl), dtype='f8' )
    for j in range( nlat ):
        lon_interval = (lon1d[1]-lon1d[0]) * DEG_2_RAD * EARTH_RADIUS * np.cos( lat1d[j] * DEG_2_RAD )
        for k in range(nlvl):
            geopot3d_x[:,j,k] = invert_first_order_derivative( xterms[:,j,k], lon_interval )
        # --- End of loop over lvls
    # --- End of loop over latitudes
    

    # The integrals are only accurate up to a constant. Killing those constants off.
    geopot3d_x -= np.mean( geopot3d_x, axis=0 )


    # Combining the two integrals
    zero_mean_geopot3d = geopot3d_x  # 0.5 * (geopot3d_x + geopot3d_y)


    # # Kill off small-scale signals (mimicking numerical diffusion in NWP models)
    # for k in range(nlvl):
    #     zero_mean_geopot3d[:,:,k] = gaussian_filter( zero_mean_geopot3d[:,:,k], sigma=1,  mode=('wrap','reflect') )
    #     # Note: Tried sigma=1, 2, and 3. Sigma=2 is the smallest trialed sigma value that produces visually
    #     # ok geopotential heights.


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
# @njit( float64[:,:,:]( float64[:,:,:], float64[:,:,:], float64[:,:,:], float64[:,:,:], float64[:,:,:], float64[:,:], float64[:,:], float64[:,:], float64[:], float64[:] ) )
def compute_balanced_geopotential_heights_from_frictionles_hydrostatic_flow_on_eta_lvls( uwind3d, vwind3d, wwind3d, pres3d, psurf2d, tsurf2d, terrain2d, lon1d, lat1d ):

    # Detect dimensions
    nlon, nlat, neta = uwind3d.shape
    nplvl = 201

    # Useful constants
    GRAVITY_ACCEL = 9.81

    # Pressure layers to do calculations on 
    plvls1d = np.linspace( pres3d.min(), 1.01e5, nplvl)[::-1]

    # Interpolate U & V to desired pressure levels
    uwind_plvl = basic_interpolate_to_pressure_levs( pres3d, uwind3d, plvls1d)
    vwind_plvl = basic_interpolate_to_pressure_levs( pres3d, vwind3d, plvls1d)
    wwind_plvl = basic_interpolate_to_pressure_levs( pres3d, wwind3d, plvls1d)

    # Compute geopotnetial with mean zero
    mean_zero_geopot = compute_geopotential_from_frictionless_3d_flow( 
        uwind_plvl, vwind_plvl, wwind_plvl, lon1d, lat1d, plvls1d 
    )

    # Filter away zonal-averages
    mean_zero_geopot -= np.mean( mean_zero_geopot, axis=0 )

    # Interpolate geopotential from plvls back to eta lvls
    mean_zero_geopot = basic_interpolate_to_eta_levs( plvls1d, mean_zero_geopot, pres3d )

    return mean_zero_geopot/GRAVITY_ACCEL