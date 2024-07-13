#!/bin/python3
'''
    PYTHON FUNCTIONS TO DIAGNOSE GEOSTROPHIC FLOW FROM THERMODYNAMIC FIELDS
    Written by: Man-Yau (Joseph) Chan

    DESCRIPTION:
    ------------
    Geostrophic flow can be diagnosed from the horizontal derivatives of geopotential on pressure surfaces. This library of functions execute that diagnostic.
    
    Note that the geostrophic equation is written in (x,y,P) coordinates. However, many weather models use terrain-following vertical coordinates instead of isobaric vertical coordinates. The approach of Kasahara (1974) is used to estimate geostrophic flow on terrain-following coordinates.
    
    This diagnostic process assumes global Gaussian grids.

    The terrain-following vertical coordinate is called "N" (for "eta").


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

t0 = time()

# flag for caching
jit_cache_flag = False


































'''
    FUNCTIONS TO TAKE HORIZONTAL DERIVATIVES ON ETA LEVELS
                                                ----------
'''




'''
    Function to compute (df/dx)_N ("x-direction partial derivative of f on eta surface")

    Inputs:
    -------
    1) field3d (lon, lat, level)
        3D NumPy array to take partial derivative on.
    2) lon1d (lon)
            1D NumPy array of longitude values (in degrees).
    3) lat1d (lat)
            1D NumPy array of latitude values (in degrees).

    Returns a 3D NumPy array (lon, lat, level) containing those derivative values
'''
@njit( float64[:,:,:]( float64[:,:,:], float64[:], float64[:] ), cache=jit_cache_flag )
def compute_df_dx_on_eta_surface( field3d, lon1d, lat1d ):

    # Useful constants
    DEG_2_RAD = PI/180
    METERS_PER_LAT_DEG = 6371*1000.*2*PI/360.

    # Longitude interval
    dlon = lon1d[1:] - lon1d[:-1]

    # Initialize array to hold derivatives
    output = np.empty( field3d.shape, dtype='f8')

    # Compute distances per degree of longitude (~111,000 m/deg on equator, 0 m/deg at poles)
    dist_per_deg_lon = METERS_PER_LAT_DEG * np.cos( lat1d * DEG_2_RAD )

    # Computing derivative on left boundary
    output[0,:,:] = (
        (field3d[1,:,:] - field3d[0,:,:])
        / ( dist_per_deg_lon[0] * dlon[0] )
    )

    # Computing derivative on right boundary
    output[-1,:,:] = (
        (field3d[-1,:,:] - field3d[-2,:,:])
        / ( dist_per_deg_lon[-1] * dlon[-1] )
    )

    # Compute derivatives on in-between points, then interpolate to interior points
    inbtwn_lon = ( lon1d[1:] + lon1d[:-1] )/2
    for j in range( field3d.shape[1] ):
        # Compute derivative at in-btwn locations
        inbtwn_derivative = (
            (field3d[1:,j,:] - field3d[:-1,j,:]).T
            / ( dist_per_deg_lon[j] * dlon[:] )
        ).T
        # Interpolate derivative to interior points
        for k in range( field3d.shape[2] ):
            output[1:-1,j,k] = np.interp(
                lon1d[1:-1], inbtwn_lon, inbtwn_derivative[:,k]
            )
        # --- End of loop over model levels
    # --- End of loop over latitude dimension

    # Return 3D array of x-direction derivatives
    return output









'''
    Function to compute (df/dy)_N ("y-direction partial derivative of f on eta surface")

    Inputs:
    -------
    1) field3d (lon, lat, level)
        3D NumPy array to take partial derivative on.
    2) lon1d (lon)
            1D NumPy array of longitude values (in degrees).
    3) lat1d (lat)
            1D NumPy array of latitude values (in degrees).

    Returns a 3D NumPy array (lon, lat, level) containing those derivative values
'''
@njit( float64[:,:,:]( float64[:,:,:], float64[:], float64[:] ), cache=jit_cache_flag )
def compute_df_dy_on_eta_surface( field3d, lon1d, lat1d ):

    # Useful constants
    DEG_2_RAD = PI/180
    METERS_PER_LAT_DEG = 6371*1000.*2*PI/360.

    # Longitude interval
    dlat = lat1d[1:] - lat1d[:-1]

    # Initialize array to hold derivatives
    output = np.empty( field3d.shape, dtype='f8')

    # Compute derivative southern boundary
    output[:,0,:] = (
        (field3d[:,1,:] - field3d[:,0,:])
        / ( METERS_PER_LAT_DEG * dlat[0] )
    )

    # Compute derivative northern boundary
    output[:,-1,:] = (
        (field3d[:,-1,:] - field3d[:,-2,:])
        / ( METERS_PER_LAT_DEG * dlat[-1] )
    )

    # Compute derivatives on in-between points, then interpolate to interior points
    inbtwn_lat = ( lat1d[1:] + lat1d[:-1] )/2
    for i in range(field3d.shape[0]):
        inbtwn_derivative = (
            (field3d[i,1:,:] - field3d[i,:-1,:]).T
            / ( METERS_PER_LAT_DEG * dlat[:] )
        ).T
        # Interpolate derivative to interior points
        for k in range( field3d.shape[2] ):
            output[i,1:-1,k] = np.interp(
                lat1d[1:-1], inbtwn_lat, inbtwn_derivative[:,k]
            )
        # --- End of loop over model levels
    # --- End of loop over longitude dimension

    # Return 3D array of y-direction derivatives
    return output












'''
    Function to compute (df/dN) ("eta-direction partial derivative of f")

    Inputs:
    -------
    1) field3d (lon, lat, level)
        3D NumPy array to take partial derivative on.
    2) eta1d (level)
            1D NumPy array of eta values.

    Returns a 3D NumPy array (lon, lat, level) containing those derivative values
'''
@njit( float64[:,:,:]( float64[:,:,:], float64[:] ), cache=jit_cache_flag )
def compute_df_dN( field3d, eta1d ):

    # Eta intervals
    deta = eta1d[1:] - eta1d[:-1]

    # Check: does eta increase or decrease with height
    if ( np.all( deta < 0 ) ):
        deta_sign = -1.0
    elif ( np.all( deta > 0) ):
        deta_sign = 1.0
    else:
        deta_sign = np.nan

    # Init array to hold output derivatives
    output = np.zeros( field3d.shape, dtype='f8' )

    # Compute derivative on lower boundary
    output[:,:,0] =  (field3d[:,:,1] - field3d[:,:,0])/deta[0]

    # Compute derivative on upper boundary
    output[:,:,-1] =  (field3d[:,:,-1] - field3d[:,:,-2])/deta[-1]

    # Compute derivative in-between levels, then interpolate to interior levels
    inbtwn_eta = ( eta1d[1:] + eta1d[:-1] )/2
    df_dN = (field3d[:,:,1:] - field3d[:,:,:-1]) / deta[:]
    for i in range(field3d.shape[0]):
        for j in range( field3d.shape[1] ):
            output[i,j,1:-1] = np.interp(
                deta_sign * eta1d[1:-1], 
                deta_sign * inbtwn_eta, df_dN[i,j,:]
            )
        # --- End of loop over model levels
    # --- End of loop over longitude dimension

    # Return 3D array of y-direction derivatives
    return output













































'''
    COMPUTE PARTIAL DERIVATIVES ON PRESSURE COORDINATES
'''


'''
    Function to compute partial derivative with respect to pressure.

    Inputs:
    -------
    1) field3d  (lon, lat, level)
            3D NumPy array to take pressure partial derivative on.
    2) pres3d (lon, lat, level)
            3D NumPy array of pressure values.

    Returns a 3D NumPy array (lon, lat, level) containing the partial derivative with respect to pressure. 
'''
@njit( float64[:,:,:]( float64[:,:,:], float64[:,:,:] ), cache=jit_cache_flag )
def compute_df_dP( field3d, pres3d ):

    # Does pressure increase or decrease with level?
    dPres = pres3d[:,:,1:] - pres3d[:,:,:-1]
    if ( np.all( dPres < 0 ) ):
        dPres_sign = -1
    elif (np.all( dPres > 0) ):
        dPres_sign = 1
    else:
        dPres_sign = np.nan

    # Init array to hold derivative field (output)
    output = np.zeros( field3d.shape, dtype='f8')

    # Compute derivative for upper boundary points
    output[:,:,-1] = (
        ( field3d[:,:,-1] - field3d[:,:,-2] )
        / ( pres3d[:,:,-1] - pres3d[:,:,-2] )
    )

    # Compute derivative for lower boundary points
    output[:,:,0] = (
        ( field3d[:,:,1] - field3d[:,:,0] )
        / ( pres3d[:,:,1] - pres3d[:,:,0] )
    )

    # Compute derivative for non-boundary points
    # This is complicated by the fact that pressure often varies unevenly with model level.
    inter_level_derivatives = (
        ( field3d[:,:,1:] - field3d[:,:,:-1] )
        / ( pres3d[:,:,1:] - pres3d[:,:,:-1] )
    )
    inter_level_pres = (
        ( pres3d[:,:,1:] + pres3d[:,:,:-1] ) / 2
    )
    for i in range( output.shape[0] ):
        for j in range( output.shape[1] ):
            output[i,j,1:-1] = np.interp(
                pres3d[i,j,1:-1] * dPres_sign, 
                inter_level_pres[i,j,:] * dPres_sign,
                inter_level_derivatives[i,j,:]
            )
    
    return output

# --- End of function to compute vertical derivatives with respect to pressure










'''
    Function to compute ( df/dx )_P

    Inputs:
    -------
    1) field3d  (lon, lat, level)
            3D NumPy array to take partial derivative on.
            This field must be defined on the eta levels!!!
    2) alpha_x  (lon, lat, level)
            3D NumPy array containing alpha_x values.
    3) lon1d (lon)
            1D NumPy array of longitude values (in degrees).
    4) lat1d (lat)
            1D NumPy array of latitude values (in degrees).
    5) eta1d (level)
            1D NumPy array of eta coordinate values.

    Returns a 3D NumPy array (lon, lat, level) containing the partial derivative with respect to pressure. 

    Uses the approach of Kasahara 1974 to compute (df/dx)_P when the data is actually defined on terrain-following coordinates.
'''
@njit( float64[:,:,:]( float64[:,:,:], float64[:,:,:], float64[:], float64[:], float64[:] ), cache=jit_cache_flag )
def compute_df_dx_on_pres_surface( field3d, alpha_x, lon1d, lat1d, eta1d ):

    # Compute x-derivative on eta surface
    output = compute_df_dx_on_eta_surface( field3d, lon1d, lat1d )

    # Compute alpha offset term
    output -= alpha_x * compute_df_dN( field3d, eta1d )

    return output










'''
    Function to compute ( df/dy )_P

    Inputs:
    -------
    1) field3d  (lon, lat, level)
            3D NumPy array to take partial derivative on.
            This field must be defined on the eta levels!!!
    2) alpha_y  (lon, lat, level)
            3D NumPy array containing alpha_x values.
    3) lon1d (lon)
            1D NumPy array of longitude values (in degrees).
    4) lat1d (lat)
            1D NumPy array of latitude values (in degrees).
    5) eta1d (level)
            1D NumPy array of eta coordinate values.

    Returns a 3D NumPy array (lon, lat, level) containing the partial derivative with respect to pressure. 

    Uses the approach of Kasahara 1974 to compute (df/dx)_P when the data is actually defined on terrain-following coordinates.
'''
@njit( float64[:,:,:]( float64[:,:,:], float64[:,:,:], float64[:], float64[:], float64[:] ), cache=jit_cache_flag )
def compute_df_dy_on_pres_surface( field3d, alpha_y, lon1d, lat1d, eta1d ):

    # Compute x-derivative on eta surface
    output = compute_df_dy_on_eta_surface( field3d, lon1d, lat1d )

    # Compute alpha offset term
    output -= alpha_y * compute_df_dN( field3d, eta1d )

    return output









































'''
    FUNCTIONS TO COMPUTE THERMODYNAMIC QUANTITIES NEEDED TO SOLVE NONLINEAR BALANCE EQUATIONS ON TERRAIN-FOLLOWING GRID
'''



'''
    Function to compute (dN/dP) * (dP/dx)_N (i.e., the constant field alpha_x)

    Inputs:
    -------
    1) eta1d  (level)
            1D NumPy array of eta coordinate values (recall: eta is the terrain-following vertical coordinate)
    2) pres3d (lon, lat, level)
            3D NumPy array of pressure values.
    3) lon1d (lon)
            1D NumPy array of longitude values (in degrees).
    4) lat1d (lat)
            1D NumPy array of latitude values (in degrees).

    Returns a 3D NumPy array (lon, lat, level) of alpha_x values
'''
@njit( float64[:,:,:]( float64[:,], float64[:,:,:], float64[:], float64[:] ), cache=jit_cache_flag )
def compute_alpha_x( eta1d, pres3d, lon1d, lat1d ):

    # Generate 3D array of eta values
    eta3d = np.empty( pres3d.shape, dtype='f8' )
    for k, eta in enumerate( eta1d ):
        eta3d[:,:,k] = eta

    # Compute dN/dP
    dN_dP = compute_df_dP( eta3d, pres3d )

    # Compute (dP/dx)_N
    dP_dx_on_eta_lvls = compute_df_dx_on_eta_surface( 
        pres3d, lon1d, lat1d
    )

    # Return the alpha_x values
    return dN_dP * dP_dx_on_eta_lvls









'''
    Function to compute (dN/dP) * (dP/dy)_N (i.e., the constant field alpha_y)

    Inputs:
    -------
    1) eta1d  (level)
            1D NumPy array of eta coordinate values (recall: eta is the terrain-following vertical coordinate)
    2) pres3d (lon, lat, level)
            3D NumPy array of pressure values.
    3) lon1d (lon)
            1D NumPy array of longitude values (in degrees).
    4) lat1d (lat)
            1D NumPy array of latitude values (in degrees).

    Returns a 3D NumPy array (lon, lat, level) of alpha_y values
'''
@njit( float64[:,:,:]( float64[:,], float64[:,:,:], float64[:], float64[:] ), cache=jit_cache_flag )
def compute_alpha_y( eta1d, pres3d, lon1d, lat1d ):

    # Generate 3D array of eta values
    eta3d = np.empty( pres3d.shape, dtype='f8' )
    for k, eta in enumerate( eta1d ):
        eta3d[:,:,k] = eta1d[k]

    # Compute dN/dP
    dN_dP = compute_df_dP( eta3d, pres3d )

    # Compute (dP/dy)_N
    dP_dy_on_eta_lvls = compute_df_dy_on_eta_surface( 
        pres3d, lon1d, lat1d
    )

    # Return the alpha_x values
    return dN_dP * dP_dy_on_eta_lvls







'''
    Function to sanity check derivatives
'''
def SANITY_CHECK_spatial_derivatives():
    
    # Generate test case
    lat = np.linspace(-4,4,21).astype('f8')
    lon = np.linspace(-4,4,41).astype('f8')
    latmesh, lonmesh = np.meshgrid( lat, lon)
    psurf = 1000 - lonmesh*8 - latmesh*12
    eta_lvls = np.linspace(0,1,61)
    eta_lvls = (eta_lvls[1:] + eta_lvls[:-1])/2.
    test_pres = np.empty( [41,21,60], dtype='f8')
    for k in range(60):
        test_pres[:,:,k] = eta_lvls[k] * psurf
    test_arr = test_pres * 2

    # Checking pressure partial derivative function
    test_derivative = compute_df_dP( test_arr, test_pres)
    print( 'Sanity checking pressure derivative function (correct value is 2) ')
    print( test_derivative.min(), test_derivative.max())
    print("")

    # Checking x-direction partial derivative on eta surface
    df_dx = compute_df_dx_on_eta_surface( test_pres, lon, lat )
    correct_val = (test_pres[21,10,5] - test_pres[20,10,5])/(( lon[21]-lon[20]) * 111000) 
    print( 'Sanity checking x-derivative function (correct value is approximately %e) ' % (correct_val))
    print( df_dx[:,10,5].min(), df_dx[:,10,5].max() )
    print("")


    # Checking y-direction partial derivative on eta surface
    df_dy = compute_df_dy_on_eta_surface( test_pres, lon, lat )
    correct_val = (test_pres[21,11,5] - test_pres[21,10,5])/(( lat[11]-lat[10]) * 111000) 
    print( 'Sanity checking y-derivative function (correct value is approximately %e) ' % (correct_val))
    print( df_dy[21,:,5].min(), df_dy[21,:,5].max() )
    print("")



    # Checking isobaric derivatives
    # ------------------------------

    # Checking testing array
    data_on_eta_lvl = test_pres.transpose(2,0,1) + 1e5*lonmesh**2 + 5e4*latmesh**2 + 3e5*latmesh*lonmesh
    data_on_eta_lvl = data_on_eta_lvl.transpose(1,2,0)

    # Compute all alpha values
    alpha_x  =  compute_alpha_x( eta_lvls, test_pres, lon, lat )
    alpha_y  =  compute_alpha_y( eta_lvls, test_pres, lon, lat )

    # Location to execute isobaric calculations is [10,5,30]
    plvl_targ = test_pres[10,5,4]

    # Generate sanity checking data
    data_on_plvls = np.zeros( (len(lon), len(lat), 3), dtype='f8' )
    plvls = np.array( [ plvl_targ - 0.1, plvl_targ, plvl_targ + 0.1 ] )
    for i in range( data_on_plvls.shape[0] ):
        for j in range( data_on_plvls.shape[1] ):
            data_on_plvls[i,j,:] = np.interp( 
                plvls*-1, 
                test_pres[i,j,:]*-1, data_on_eta_lvl[i,j,:]
            )
    
    # Check (df/dx)_P at desired location
    test_val = compute_df_dx_on_pres_surface( data_on_eta_lvl, alpha_x, lon, lat, eta_lvls)
    true_val = compute_df_dx_on_eta_surface( data_on_plvls, lon, lat )
    print( 'Sanity checking df/dx on isobaric surface (correct value is approximately %e) ' % (true_val[10,5,1]) )
    print( test_val[10,5,4] )
    print("")


    # Check (df/dy)_P at desired location
    test_val = compute_df_dy_on_pres_surface( data_on_eta_lvl, alpha_x, lon, lat, eta_lvls)
    true_val = compute_df_dy_on_eta_surface( data_on_plvls, lon, lat )
    print( 'Sanity checking df/dy on isobaric surface (correct value is approximately %e) ' % (true_val[10,5,1]) )
    print( test_val[10,5,4] )
    print("")

    return












































'''
    FUNCTION TO PAD ARRAY DEFINED ON SPHERICAL SHELL

    Inputs:
    --------
    1) field3d  (lon, lat, level)
            3D NumPy array to pad
    2) lon1d (lon)
            1D NumPy array of longitude values (in degrees).
    3) lat1d (lat)
            1D NumPy array of latitude values (in degrees).

    Returns a 3D NumPy array with dimensions (lon+2, lat+2, level) 

'''
@njit( nbtuple( (float64[:], float64[:], float64[:,:,:]) )( float64[:,:,:], float64[:], float64[:] ), cache=jit_cache_flag )
def pad_field_due_to_spherical_symmetry( field3d, lon1d, lat1d ): 

    # Init array to hold padded values
    padded_field3d = np.empty(
        ( field3d.shape[0]+2, field3d.shape[1]+2, field3d.shape[2] ),
        dtype='f8'
    )

    # Populate interior points
    padded_field3d[1:-1,1:-1,:] = field3d[:,:,:]

    # Pad the westmost boundary
    padded_field3d[0,1:-1,:] = field3d[-1,:,:]

    # Pad the eastmost boundary
    padded_field3d[-1,1:-1,:] = field3d[0,:,:]

    # Determining longitudes beyond the north/south poles
    corresponding_lon = lon1d + 180.
    while ( np.sum(corresponding_lon > 180) > 0):
        corresponding_lon[corresponding_lon > 180] -= 360

    # Pad the northmost boundary
    for k in range( field3d.shape[2] ):
        padded_field3d[1:-1,-1,k] = np.interp(
            corresponding_lon,
            lon1d, field3d[:,-1,k]
        )
    # --- End of loop over model levels

    # Pad the Southmost boundary
    for k in range( field3d.shape[2] ):
        padded_field3d[1:-1,0,k] = np.interp(
            corresponding_lon,
            lon1d, field3d[:,0,k]
        )
    # --- End of loop over model levels    


    # Pad the southwest corner
    padded_field3d[0,0,:] = field3d[-1,0,:]

    # Pad the southeast corner
    padded_field3d[-1,0,:] = field3d[0,0,:]

    # Pad the northwest corner
    padded_field3d[0,-1,:] = field3d[-1,-1,:]

    # Pad the northeast corner
    padded_field3d[-1,-1,:] = field3d[0,-1,:]

    # Padded longitude
    padded_lon1d = np.empty( len(lon1d)+2, dtype='f8')
    padded_lon1d[1:-1] = lon1d
    dlon = lon1d[1] - lon1d[0]
    padded_lon1d[ 0] = lon1d[ 0] - dlon
    padded_lon1d[-1] = lon1d[-1] + dlon


    # Padded latitude
    padded_lat1d = np.empty( len(lat1d)+2, dtype='f8')
    padded_lat1d[1:-1] = lat1d
    dlat = lat1d[1] - lat1d[0]
    padded_lat1d[ 0] = lat1d[ 0] - dlat
    padded_lat1d[-1] = lat1d[-1] + dlat

    return padded_lon1d, padded_lat1d, padded_field3d







'''
    Function to sanity check the spherical symmetry padding function
'''
def SANITY_CHECK_pad_field_due_to_spherical_symmetry():

    import matplotlib.pyplot as plt

    # Grid settings (must be even numbers)
    nlat = 10
    nlon = nlat * 2
    nlvl = 4


    # Generate grid
    lat1d =  ( (np.arange( nlat )+0.5 - int(nlat/2)) * 180/nlat ).astype('f8')
    lon1d =  ( (np.arange( nlon )+0.5 - int(nlon/2)) * 360/nlon ).astype('f8')



    # Generate 2D meshes for lat, lon and pressure
    latmesh, lonmesh = np.meshgrid(lat1d, lon1d)
    # Generate some plot to check if the spherical padding is working
    arr2d = np.zeros( (nlon, nlat, nlvl), dtype='f8' )
    arr2d[:,:,0] = np.cos( 2*latmesh * PI/180 ) * np.cos( 2*lonmesh * PI/180 )
    arr2d[:,:,1] = np.cos( 2*latmesh * PI/180 ) * np.cos( 2*lonmesh * PI/180 )*2
    plon1d, plat1d, parr2d = pad_field_due_to_spherical_symmetry( arr2d, lon1d, lat1d)
    plt.contour( lon1d, lat1d, arr2d[:,:,1].T, [-1.2, 1.2], colors='k', linestyles=['--','-'] )
    plt.contourf( plon1d, plat1d, parr2d[:,:,1].T, 11, cmap = 'RdBu_r' )
    plt.colorbar()


    plt.axvline( 180, color='white', linestyle='-', linewidth=5 )
    plt.axvline(-180, color='white', linestyle='-', linewidth=5 )
    plt.axhline( 90, color='white', linestyle='-', linewidth=5 )
    plt.axhline(-90, color='white', linestyle='-', linewidth=5 )

    plt.axvline( 180, color='k', linestyle='-' )
    plt.axvline(-180, color='k', linestyle='-' )
    plt.axhline( 90, color='k', linestyle='-' )
    plt.axhline(-90, color='k', linestyle='-' )


    plt.savefig('check_spherical_array_padding.png')
    plt.close()

    return






































































'''
    FUNCTION TO DIAGNOSE GEOSTROPHIC FLOW ON TERRAIN-FOLLOWING COORDINATES

    All inputs must be in SI units!!

    Inputs:
    -------
    1) pres3d    (lon, lat, layer)
            3D NumPy array of pressure values. Must monotonically decrease with increasing altitude.
    2) psurf2d   (lon, lat)
            2D NumPy array of pressure values at the ground surface.
    3) ptop2d    (lon, lat)
            2D NumPy array of pressure values at the top of the top model layer.
    4) hgt3d  (lon, lat, layer) 
            3D NumPy array of model layer height values.
    5) terrain2d (lon, lat)
            2D NumPy array of terrain heights
    6) hgttop2d    (lon, lat)
            2D NumPy array of model top heights
    7) lon1d (lon)
            1D NumPy array of longitude values (in degrees).
    8) lat1d (lat)
            1D NumPy array of latitude values (in degrees).
'''
# @njit( nbtuple( (float64[:,:,:], float64[:,:,:], float64[:,:,:]) )( float64[:,:,:], float64[:,:], float64[:,:], float64[:,:,:], float64[:,:], float64[:,:], float64[:], float64[:]) )
def diagnose_geostrophic_flow( pres3d, psurf2d, ptop2d, hgt3d, terrain2d, hgttop2d, lon1d, lat1d ):


    # Useful constants
    DEG_2_RAD = PI/180
    EARTH_ANGULAR_SPEED = 7.2921159e-5
    GRAVITY_ACCEL = 9.80665 

    # Dimensions of unpadded arrays
    nlons, nlats, nlayers = pres3d.shape

    # Generate padded "eta" levels
    peta1d = np.arange(nlayers+2).astype('f8')


    # Generate array of padded pressure values
    # -----------------------------------------

    # Does P increase with increasing layer index, or decrease with increasing layer index?
    dp = pres3d[:,:,1:] - pres3d[:,:,:-1]
    if ( np.all( dp <= 0) ):
        sfc_ilayer = 0
        top_ilayer = -1
    elif ( np.all( dp >= 0) ):
        sfc_ilayer = -1
        top_ilayer = 0
    # --- endif

    # Top-and-bottom pad the pressure array
    ppres3d = np.empty( (nlons, nlats, nlayers+2), dtype='f8' )
    ppres3d[:,:,1:-1] = pres3d
    ppres3d[:,:,sfc_ilayer] = psurf2d
    ppres3d[:,:,top_ilayer] = ptop2d

    # horizontally pad the pressure array
    plon1d, plat1d, ppres3d = pad_field_due_to_spherical_symmetry( ppres3d, lon1d, lat1d )



    # Generate array of padded geopotential values
    # ---------------------------------------------

    # Top-and-bottom pad the height array
    phgt3d = np.empty( (nlons, nlats, nlayers+2), dtype='f8' )
    phgt3d[:,:,1:-1] = hgt3d
    phgt3d[:,:,sfc_ilayer] = terrain2d
    phgt3d[:,:,top_ilayer] = hgttop2d

    # horizontally pad the height array
    plon1d, plat1d, phgt3d = pad_field_due_to_spherical_symmetry( phgt3d, lon1d, lat1d )

    # convert padded height array to padded geopotential array
    pgeopot3d = phgt3d * GRAVITY_ACCEL 



    # Compute coefficients needed to diagnose geostrophic flow
    # ---------------------------------------------------------

    # Compute all alpha values
    alpha_x  =  compute_alpha_x( peta1d, ppres3d, plon1d, plat1d )
    alpha_y  =  compute_alpha_y( peta1d, ppres3d, plon1d, plat1d )

    # Generate coriolis parameter at all locations
    platmesh, plonmesh = np.meshgrid( plat1d, plon1d )
    coriolis_param3d = np.empty( ppres3d.shape, dtype='f8' )
    for k in range( len(peta1d) ):
        coriolis_param3d[:,:,k] =  2 * EARTH_ANGULAR_SPEED * np.sin( DEG_2_RAD * platmesh )
    

    # Compute geostrophic flow
    # ------------------------
    u3d = (
        compute_df_dy_on_pres_surface( pgeopot3d, alpha_y, plon1d, plat1d, peta1d )[1:-1,1:-1,1:-1]
        / coriolis_param3d[1:-1,1:-1,1:-1]
    ) * (-1.0)
    v3d = (
        compute_df_dx_on_pres_surface( pgeopot3d, alpha_x, plon1d, plat1d, peta1d )[1:-1,1:-1,1:-1]
        / coriolis_param3d[1:-1,1:-1,1:-1]
    ) 


    # # Remove equatorial geostrophic flow
    # # ----------------------------------
    # flag_lat_eq = ( np.abs( lat1d )< 5)
    # u3d[:,flag_lat_eq,:] = np.nan
    # v3d[:,flag_lat_eq,:] = np.nan


    return u3d, v3d







'''
    SANITY CHECK DIAGNOSIS OF GEOSTROPHIC FLOW
'''
def SANITY_CHECK_geostrophic_flow_diagnosis():

    # Load python packages for plotting
    # from matplotlib import use as mpl_use
    # mpl_use('agg')
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize

    # Grid settings (must be even numbers)
    nlat = 180
    nlon = nlat * 2
    nlvl = 11


    # Generate grid
    lat1d =  ( (np.arange( nlat )+0.5 - int(nlat/2)) * 180/nlat ).astype('f8')
    lon1d =  ( (np.arange( nlon )+0.5 - int(nlon/2)) * 360/nlon ).astype('f8')
    pres1d = ( ( (np.arange( nlvl +1 ) )[1:] * 980/(nlvl+1) )[::-1] + 20 ).astype('f8') * 100


    # Generate 2D meshes for lat, lon and pressure
    latmesh, lonmesh = np.meshgrid(lat1d, lon1d)
    psurf2d = ( (latmesh / latmesh) * 1000. ).astype('f8') * 100
    ptop2d = psurf2d * 0. + 20*100


    # Generate zonally and vertically symmetric height fields
    height3d = np.empty( [nlon, nlat, nlvl], dtype='f8')
    amplitude = np.log( pres1d[0]/100000 ) * -12000/1.7 / 2
    meridional_variation = amplitude * np.exp( -0.5 * (latmesh)**2/(30**2) ) 
    zonal_variation = np.abs(np.sin( 0.5*latmesh * PI/180)) * np.cos(4* lonmesh * PI/180) * amplitude /2

    for kk in range(nlvl):
        ref_height = np.log( pres1d[kk]/100000 ) * -12000/1.7
        height3d[:,:,kk] = ref_height + meridional_variation + zonal_variation
    # --- End of loop over model layers

    # Generate meshes for terrain and model top height
    terrain2d = latmesh*0
    hgttop2d = np.log( 2000./100000 ) * -12000/1.7 + meridional_variation + zonal_variation
    
    # Construct 3d pressure field
    pres3d = np.empty( (nlon, nlat, nlvl), dtype='f8')
    for kk in range(nlvl):
        pres3d[:,:,kk] = pres1d[kk]


    # Generate geostrophic flow
    u3d, v3d = diagnose_geostrophic_flow( 
        pres3d, psurf2d, ptop2d, height3d, terrain2d, hgttop2d, lon1d, lat1d 
    )


    # Visualize geostrophic flow and geopotential height
    geopot3d = 9.81 * height3d
    cnf = plt.contourf( lon1d, lat1d, geopot3d[:,:,5].T, 11, cmap = 'RdBu_r')
    cbar = plt.colorbar(cnf)
    cbar.ax.set_ylabel('Geopotential (J/kg)')
    # plt.quiver( lonmesh[::], latmesh, norm_u[:,:,2], norm_v[:,:,2])
    plt.streamplot( lon1d, lat1d, 
        (u3d[:,:,5] / (111000 * np.cos( latmesh*PI/180 ))).T, 
        (v3d[:,:,5] / 111000).T,
         color='k', density=2
    )
    plt.savefig('check_geostrophic_flow.png')
    plt.close()


    return























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