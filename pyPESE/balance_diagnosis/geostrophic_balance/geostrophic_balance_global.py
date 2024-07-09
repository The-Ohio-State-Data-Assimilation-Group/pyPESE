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
    inbtwn_lon = ( lon1d[1:] + lon1d[:-1] )
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
    inbtwn_lat = ( lat1d[1:] + lat1d[:-1] )
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
    inbtwn_eta = ( eta1d[1:] + eta1d[:-1] )
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
    Function to compute (d2f/dN2) ("eta-direction second partial derivative of f")

    Inputs:
    -------
    1) field3d (lon, lat, level)
        3D NumPy array to take partial derivative on.
    2) eta1d (level)
            1D NumPy array of eta values.

    Returns a 3D NumPy array (lon, lat, level) containing those derivative values,
    except at the topmost and bottommost boundaries (values are NaN there.)
'''
@njit( float64[:,:,:]( float64[:,:,:], float64[:] ), cache=jit_cache_flag )
def compute_df2_dN2_on_eta_surface( field3d, eta1d ):

    # Eta intervals
    deta = eta1d[1:] - eta1d[:-1]

    # Check: does eta increase or decrease with height
    if ( np.all( deta < 0 ) ):
        deta_sign = -1.0
    elif ( np.all( deta > 0) ):
        deta_sign = 1.0
    else:
        deta_sign = np.nan


    # Compute first derivative over eta
    # ---------------------------------
    # Eta interval and location
    deta = eta1d[1:] - eta1d[:-1]
    eta_inbtwn = ( eta1d[1:] + eta1d[:-1] )/2

    # Compute derivative
    df_dN = ( field3d[:,:,1:] - field3d[:,:,:-1] ) / deta
    # Dimensions of df_dN are (lon, lat, level - 1)


    # Compute second derivative over eta
    # ----------------------------------
    # Eta interval and location
    deta = eta_inbtwn[1:] - eta_inbtwn[:-1]
    eta_inbtwn = ( eta_inbtwn[1:] + eta_inbtwn[:-1] )/2

    # Compute derivative
    d2f_dN2 = ( df_dN[:,:,1:] - df_dN[:,:,:-1] ) / deta
    # Dimensions of df_dN are (lon, lat, level - 2)


    # Init array to hold derivatives
    output = np.empty( field3d.shape, dtype='f8')

    
    # Interpolate second derivative to desired locations
    # --------------------------------------------------
    for i in range( field3d.shape[0] ):
        for j in range( field3d.shape[1] ):
            output[i,j,1:-1] = np.interp(
                deta_sign * eta1d[1:-1], 
                deta_sign * eta_inbtwn, d2f_dN2[i,j,:]
            )
        # --- End of loop over latitudes
    # --- End of loop over longitudes

    output[:,:,0] = np.nan
    output[:,:,-1] = np.nan

    # Return 3D array of eta-direction derivatives
    return output











'''
    Function to compute ( d2 f/ (dx dy) )_N 

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
def compute_d2f_dxdy_on_eta_surface( field3d, lon1d, lat1d ):

    # Compute x-direction derivative
    df_dx = compute_df_dx_on_eta_surface(
        field3d, lon1d, lat1d
    )

    # Compute y-direction derivative
    d2f_dxdy = compute_df_dy_on_eta_surface(
        df_dx, lon1d, lat1d
    )

    return d2f_dxdy








'''
    Function to compute ( d2 f/ (dx^2) )_N 

    Inputs:
    -------
    1) field3d (lon, lat, level)
        3D NumPy array to take partial derivative on.
    2) lon1d (lon)
            1D NumPy array of longitude values (in degrees).
    3) lat1d (lat)
            1D NumPy array of latitude values (in degrees).

    Returns a 3D NumPy array (lon, lat, level) containing those derivative values,
    except at the eastmost and westmost boundaries (values are NaN there.)
'''
@njit( float64[:,:,:]( float64[:,:,:], float64[:], float64[:] ), cache=jit_cache_flag )
def compute_d2f_dx2_on_eta_surface( field3d, lon1d, lat1d ):

    # Useful constants
    DEG_2_RAD = PI/180
    METERS_PER_LAT_DEG = 6371*1000.*2*PI/360.


    # Compute first x-derivative
    # --------------------------
    # Prepare coordinates
    dlon = lon1d[1:] - lon1d[:-1]
    lon_inbtwn = (lon1d[1:] + lon1d[:-1])/2

    # Compute derivative over lon
    df_dx = (field3d[1:,:,:] - field3d[:-1,:,:]).transpose(1,2,0) / dlon
    # dims of df_dlon are (lat, level, lon-1)

    # Convert df_dlon to df_dx
    for j in range( len(lat1d) ):
        meters_per_lon_deg = METERS_PER_LAT_DEG * np.cos( lat1d[j] * DEG_2_RAD )
        df_dx[j,:,:] /= meters_per_lon_deg
    # dims of dfdx are (lat, level, lon-1)


    # Compute second x derivative
    # ---------------------------
    # Prepare interval
    dlon = lon_inbtwn[1:] - lon_inbtwn[:-1]
    new_lons = (lon_inbtwn[1:] + lon_inbtwn[:-1])/2

    # Compute derivative over lon
    d2f_dx2 = (df_dx[:,:,1:] - df_dx[:,:,:-1]) / dlon
    # dims of d2f_dlon2 are (lat, level, lon-2)

    # Convert d2f_dlon2 to d2f_dx2
    for j in range( len(lat1d) ):
        meters_per_lon_deg = METERS_PER_LAT_DEG * np.cos( lat1d[j] * DEG_2_RAD )
        d2f_dx2[j,:,:] /= meters_per_lon_deg
    # dims of d2f_dx2 are (lat, level, lon-2)

    # Rearrange dimensions for convenience
    d2f_dx2 = d2f_dx2.transpose(2,0,1)
    # Dimensions  of d2f_dx2 are now (lon-2, lat, level)

    # Interpolate second derivate to the interior longitude points
    for j in range( d2f_dx2.shape[1] ):
        for k in range( d2f_dx2.shape[2] ):
            d2f_dx2[:,j,k] = np.interp(
                lon1d[1:-1], new_lons, d2f_dx2[:,j,k]
            )
        # --- End of loop over model levels
    # --- End of loop over latitudes

    # Init output array and store derivatives in it
    output = np.empty( field3d.shape, dtype='f8')
    output[1:-1,:,:] = d2f_dx2
    output[0,:,:] = np.nan
    output[-1,:,:] = np.nan

    return output









'''
    Function to compute ( d2 f/ (dy^2) )_N 

    Inputs:
    -------
    1) field3d (lon, lat, level)
        3D NumPy array to take partial derivative on.
    2) lon1d (lon)
            1D NumPy array of longitude values (in degrees).
    3) lat1d (lat)
            1D NumPy array of latitude values (in degrees).

    Returns a 3D NumPy array (lon, lat, level) containing those derivative values,
    except at the northmost and southmost boundaries (values are NaN there.)
'''
@njit( float64[:,:,:]( float64[:,:,:], float64[:], float64[:] ), cache=jit_cache_flag )
def compute_d2f_dy2_on_eta_surface( field3d, lon1d, lat1d ):

    # Useful constants
    DEG_2_RAD = PI/180
    METERS_PER_LAT_DEG = 6371*1000.*2*PI/360.


    # Compute first y-derivative
    # --------------------------
    # Prepare coordinates
    dlat = lat1d[1:] - lat1d[:-1]
    lat_inbtwn = (lat1d[1:] + lat1d[:-1])/2

    # Compute df/dy
    df_dy = (field3d[:,1:,:] - field3d[:,:-1,:]).transpose(0,2,1) / (METERS_PER_LAT_DEG * dlat )
    # dims of df_dy are (lon, level, lat-1)


    # Compute second y derivative
    # ---------------------------
    # Prepare interval
    dlat = lat_inbtwn[1:] - lat_inbtwn[:-1]
    lat_inbtwn = (lat_inbtwn[1:] + lat_inbtwn[:-1])/2

    # Compute derivative over lon
    d2f_dy2 = (df_dy[:,:,1:] - df_dy[:,:,:-1])  / (METERS_PER_LAT_DEG * dlat )
    # dims of d2f_dy2 are (lon, level, lat-2)

    # Rearrange dimensions for convenience
    d2f_dy2 = d2f_dy2.transpose(0,2,1)
    # Dimensions  of d2f_dx2 are now (lon, lat-2, level)

    # Interpolate second derivate to the interior latitude points
    for i in range( d2f_dy2.shape[0] ):
        for k in range( d2f_dy2.shape[2] ):
            d2f_dy2[i,:,k] = np.interp(
                lat1d[1:-1], lat_inbtwn, d2f_dy2[i,:,k]
            )
        # --- End of loop over model levels
    # --- End of loop over latitudes

    # Init output array and store derivatives in it
    output = np.empty( field3d.shape, dtype='f8')
    output[:,1:-1,:] = d2f_dy2
    output[:,0,:] = np.nan
    output[:,-1,:] = np.nan

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
                pres3d[i,j,1:-1], 
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
    Function to compute ( (d2 f)/(dx dy) )_P

    Inputs:
    -------
    1) field3d  (lon, lat, level)
            3D NumPy array to take partial derivative on.
            This field must be defined on the eta levels!!!
    2) alpha_x  (lon, lat, level)
            3D NumPy array containing alpha_x values.
    3) alpha_y  (lon, lat, level)
            3D NumPy array containing alpha_y values.
    4) alpha_xy  (lon, lat, level)
            3D NumPy array containing alpha_xy values.
    5) lon1d (lon)
            1D NumPy array of longitude values (in degrees).
    6) lat1d (lat)
            1D NumPy array of latitude values (in degrees).
    7) eta1d (level)
            1D NumPy array of eta coordinate values.

    Returns a 3D NumPy array (lon, lat, level) containing the partial derivative desired.

    Important Notes:
    ----------------
    1) The derivative values at the following locations are set to NaN for safety reasons:
        * Topmost and bottommost layers
        * Eastmost and westmost boundaries
        * Northmost and southmost boundaries

    2) The approach of Kasahara 1974 is used to compute derivatives on isobars when the 
       data is actually defined on terrain-following coordinates. The formula used here is
            ( d2f / dxdy )_P =  ( d2f / dxdy )_N 
                                - alpha_x (d/dy)_N (d/dN) 
                                - alpha_y (d/dx)_N (d/dN)
                                + alpha_x * alpha_y (d2/dN2)
                                - alpha_xy (d/dN)

'''
@njit( float64[:,:,:]( float64[:,:,:], float64[:,:,:], float64[:,:,:], float64[:,:,:], float64[:], float64[:], float64[:] ), cache=jit_cache_flag )
def compute_d2f_dxdy_on_pres_surface( field3d, alpha_x, alpha_y, alpha_xy, lon1d, lat1d, eta1d ):

    # Compute the ( d2f / dxdy )_N term
    output = compute_d2f_dxdy_on_eta_surface( 
        field3d, lon1d, lat1d
    )

    # Compute df/dN since it shows up often
    df_dN = compute_df_dN( field3d, eta1d )

    # Compute the (- alpha_x (d/dy)_N (d/dN)) term
    output -= alpha_x * compute_df_dy_on_eta_surface( df_dN, lon1d, lat1d )

    # Compute the (- alpha_y (d/dx)_N (d/dN)) term
    output -= alpha_y * compute_df_dx_on_eta_surface( df_dN, lon1d, lat1d )

    # Compute the (+ alpha_x * alpha_y (d2/dN2)) term
    output += (
        alpha_x * alpha_y 
        * compute_df2_dN2_on_eta_surface( field3d, eta1d )
    )

    # Compute the (- alpha_xy * (d/dN)) term
    output -= (alpha_xy * df_dN)

    # Guard-rails to prevent weird derivative values
    output[ 0,:,:] = np.nan
    output[-1,:,:] = np.nan    
    output[:, 0,:] = np.nan
    output[:,-1,:] = np.nan
    output[:,:, 0] = np.nan
    output[:,:,-1] = np.nan

    return output








'''
    Function to compute ( (d2 f)/(dx2) )_P

    Inputs:
    -------
    1) field3d  (lon, lat, level)
            3D NumPy array to take partial derivative on.
            This field must be defined on the eta levels!!!
    2) alpha_x  (lon, lat, level)
            3D NumPy array containing alpha_x values.
    3) alpha_y  (lon, lat, level)
            3D NumPy array containing alpha_y values.
    4) alpha_xx  (lon, lat, level)
            3D NumPy array containing alpha_xx values.
    5) lon1d (lon)
            1D NumPy array of longitude values (in degrees).
    6) lat1d (lat)
            1D NumPy array of latitude values (in degrees).
    7) eta1d (level)
            1D NumPy array of eta coordinate values.

    Returns a 3D NumPy array (lon, lat, level) containing the partial derivative desired.

    Important Notes:
    ----------------
    1) The derivative values at the following locations are set to NaN for safety reasons:
        * Topmost and bottommost layers
        * Eastmost and westmost boundaries
        * Northmost and southmost boundaries

    2) The approach of Kasahara 1974 is used to compute derivatives on isobars when the 
       data is actually defined on terrain-following coordinates. The formula used here is
            ( d2f / d2 )_P =  ( d2f / dx2 )_N 
                                - alpha_x (d/dx)_N (d/dN) 
                                - alpha_x (d/dx)_N (d/dN)
                                + alpha_x * alpha_x (d2/dN2)
                                - alpha_xx (d/dN)

'''
@njit( float64[:,:,:]( float64[:,:,:], float64[:,:,:], float64[:,:,:], float64[:,:,:], float64[:], float64[:], float64[:] ), cache=jit_cache_flag )
def compute_d2f_dx2_on_pres_surface( field3d, alpha_x, alpha_y, alpha_xx, lon1d, lat1d, eta1d ):

    # Compute the ( d2f / dx2 )_N term
    output = compute_d2f_dx2_on_eta_surface( 
        field3d, lon1d, lat1d
    )

    # Compute df/dN since it shows up often
    df_dN = compute_df_dN( field3d, eta1d )

    # Compute the (- alpha_x (d/dx)_N (d/dN)) term
    output -= alpha_x * compute_df_dx_on_eta_surface( df_dN, lon1d, lat1d ) *2

    # Compute the (+ alpha_x * alpha_x (d2/dN2)) term
    output += (
        alpha_x * alpha_x
        * compute_df2_dN2_on_eta_surface( field3d, eta1d )
    )

    # Compute the (- alpha_xy * (d/dN)) term
    output -= (alpha_xx * df_dN)

    # Guard-rails to prevent weird derivative values
    output[ 0,:,:] = np.nan
    output[-1,:,:] = np.nan    
    output[:, 0,:] = np.nan
    output[:,-1,:] = np.nan
    output[:,:, 0] = np.nan
    output[:,:,-1] = np.nan

    return output










'''
    Function to compute ( (d2 f)/(dy2) )_P

    Inputs:
    -------
    1) field3d  (lon, lat, level)
            3D NumPy array to take partial derivative on.
            This field must be defined on the eta levels!!!
    2) alpha_x  (lon, lat, level)
            3D NumPy array containing alpha_x values.
    3) alpha_y  (lon, lat, level)
            3D NumPy array containing alpha_y values.
    4) alpha_yy  (lon, lat, level)
            3D NumPy array containing alpha_yy values.
    5) lon1d (lon)
            1D NumPy array of longitude values (in degrees).
    6) lat1d (lat)
            1D NumPy array of latitude values (in degrees).
    7) eta1d (level)
            1D NumPy array of eta coordinate values.

    Returns a 3D NumPy array (lon, lat, level) containing the partial derivative desired.

    Important Notes:
    ----------------
    1) The derivative values at the following locations are set to NaN for safety reasons:
        * Topmost and bottommost layers
        * Eastmost and westmost boundaries
        * Northmost and southmost boundaries

    2) The approach of Kasahara 1974 is used to compute derivatives on isobars when the 
       data is actually defined on terrain-following coordinates. The formula used here is
            ( d2f / dy2 )_P =  ( d2f / dy2 )_N 
                                - alpha_y (d/dy)_N (d/dN) 
                                - alpha_y (d/dy)_N (d/dN)
                                + alpha_y * alpha_y (d2/dN2)
                                - alpha_yy (d/dN)

'''
@njit( float64[:,:,:]( float64[:,:,:], float64[:,:,:], float64[:,:,:], float64[:,:,:], float64[:], float64[:], float64[:] ), cache=jit_cache_flag )
def compute_d2f_dy2_on_pres_surface( field3d, alpha_x, alpha_y, alpha_yy, lon1d, lat1d, eta1d ):

    # Compute the ( d2f / dy2 )_N term
    output = compute_d2f_dy2_on_eta_surface( 
        field3d, lon1d, lat1d
    )

    # Compute df/dN since it shows up often
    df_dN = compute_df_dN( field3d, eta1d )

    # Compute the (- alpha_y (d/dy)_N (d/dN)) term
    output -= alpha_y * compute_df_dy_on_eta_surface( df_dN, lon1d, lat1d ) *2

    # Compute the (+ alpha_y * alpha_y (d2/dN2)) term
    output += (
        alpha_y * alpha_y
        * compute_df2_dN2_on_eta_surface( field3d, eta1d )
    )

    # Compute the (- alpha_xy * (d/dN)) term
    output -= (alpha_yy * df_dN)

    # Guard-rails to prevent weird derivative values
    output[ 0,:,:] = np.nan
    output[-1,:,:] = np.nan    
    output[:, 0,:] = np.nan
    output[:,-1,:] = np.nan
    output[:,:, 0] = np.nan
    output[:,:,-1] = np.nan

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
        eta3d[:,:,k] = eta1d[k]

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
    Function to compute (dN/dP) * ( (d2P)/(dy dx) )_N (i.e., the constant field alpha_xy)

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
def compute_alpha_xy( eta1d, pres3d, lon1d, lat1d ):

    # Generate 3D array of eta values
    eta3d = np.empty( pres3d.shape, dtype='f8' )
    for k, eta in enumerate( eta1d ):
        eta3d[:,:,k] = eta1d[k]

    # Compute dN/dP
    dN_dP = compute_df_dP( eta3d, pres3d )

    # Compute (dP/dx)_N
    dP_dx_on_eta_lvls = compute_df_dx_on_eta_surface( 
        pres3d, lon1d, lat1d
    )

    # Compute ( (d2P)/(dx dy) )_N
    d2P_dxdy_on_eta_lvls = compute_df_dy_on_eta_surface( 
        dP_dx_on_eta_lvls, lon1d, lat1d
    )

    # Return the alpha_x values
    return dN_dP * d2P_dxdy_on_eta_lvls








'''
    Function to compute (dN/dP) * ( (d2P)/(dx2) )_N (i.e., the constant field alpha_xx)

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
def compute_alpha_xx( eta1d, pres3d, lon1d, lat1d ):

    # Generate 3D array of eta values
    eta3d = np.empty( pres3d.shape, dtype='f8' )
    for k, eta in enumerate( eta1d ):
        eta3d[:,:,k] = eta1d[k]

    # Compute dN/dP
    dN_dP = compute_df_dP( eta3d, pres3d )

    # Compute (dP/dx)_N
    dP_dx_on_eta_lvls = compute_df_dx_on_eta_surface( 
        pres3d, lon1d, lat1d
    )

    # Compute ( (d2P)/(dx2) )_N
    d2P_dx2_on_eta_lvls = compute_df_dx_on_eta_surface( 
        dP_dx_on_eta_lvls, lon1d, lat1d
    )

    # Return the alpha_x values
    return dN_dP * d2P_dx2_on_eta_lvls








'''
    Function to compute (dN/dP) * ( (d2P)/(dy2) )_N (i.e., the constant field alpha_yy)

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
def compute_alpha_yy( eta1d, pres3d, lon1d, lat1d ):

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

    # Compute ( (d2P)/(dy2) )_N
    d2P_dy2_on_eta_lvls = compute_df_dy_on_eta_surface( 
        dP_dy_on_eta_lvls, lon1d, lat1d
    )

    # Return the alpha_x values
    return dN_dP * d2P_dy2_on_eta_lvls




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
    alpha_xx = compute_alpha_xx( eta_lvls, test_pres, lon, lat )
    alpha_yy = compute_alpha_yy( eta_lvls, test_pres, lon, lat )
    alpha_xy = compute_alpha_xy( eta_lvls, test_pres, lon, lat )

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


    # Check (d2f/dx2)_P at desired location
    test_val = compute_d2f_dx2_on_pres_surface( data_on_eta_lvl, alpha_x, alpha_y, alpha_xx, lon, lat, eta_lvls)
    true_val = compute_d2f_dx2_on_eta_surface( data_on_plvls, lon, lat )
    print( 'Sanity checking d2f/dx2 on isobaric surface (correct value is approximately %e) ' % (true_val[10,5,1]) )
    print( test_val[10,5,4] )
    print("")

    # Check (d2f/dy2)_P at desired location
    test_val = compute_d2f_dy2_on_pres_surface( data_on_eta_lvl, alpha_x, alpha_y, alpha_yy, lon, lat, eta_lvls)
    true_val = compute_d2f_dy2_on_eta_surface( data_on_plvls, lon, lat )
    print( 'Sanity checking d2f/dy2 on isobaric surface (correct value is approximately %e) ' % (true_val[10,5,1]) )
    print( test_val[10,5,4] )
    print("")


    # Check (d2f/dxdy)_P at desired location
    test_val = compute_d2f_dxdy_on_pres_surface( data_on_eta_lvl, alpha_x, alpha_y, alpha_xy, lon, lat, eta_lvls)
    true_val = compute_d2f_dxdy_on_eta_surface( data_on_plvls, lon, lat )
    print( 'Sanity checking d2f/dxdy on isobaric surface (correct value is approximately %e) ' % (true_val[10,5,1]) )
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
    UNDER-THE-HOOD EAGER-COMPILABLE FUNCTIONS USED TO EVALUATE COST FUNCTION AND GRADIENT OF COST FUNCTION
'''



'''
    Function to evaluate residuals of Charney (1955) nonlinear balance

    Input:
    ------
    1) pstreamfunc3d                (lon+2, lat+2, layer+2)
            3D NumPy array of padded streamfunction values

    2) alpha_x3d                    (lon+2, lat+2, layer+2)
            3D NumPy array of padded alpha_x values (array in padded_constants_dict['alpha_x'])

    3) alpha_y3d                    (lon+2, lat+2, layer+2)
            3D NumPy array of padded alpha_y values (array in padded_constants_dict['alpha_y'])

    4) alpha_xx3d                   (lon+2, lat+2, layer+2)
            3D NumPy array of padded alpha_xx values (array in padded_constants_dict['alpha_xx'])

    5) alpha_yy3d                   (lon+2, lat+2, layer+2)
            3D NumPy array of padded alpha_xy values (array in padded_constants_dict['alpha_yy'])

    6) alpha_xy3d                   (lon+2, lat+2, layer+2)
            3D NumPy array of padded alpha_xy values (array in padded_constants_dict['alpha_xy'])

    7) coriolis_param3d             (lon+2, lat+2, layer+2)
            3D NumPy array of padded coriolis parameter values (array in padded_constants_dict['coriolis'])

    8) ygrad_coriolis_param3d       (lon+2, lat+2, layer+2)
            3D NumPy array of padded coriolis beta parameter values (array in padded_constants_dict['coriolis ygrad'])

    9) fscaled_laplacian_geopot3d   (lon+2, lat+2, layer+2)
            3D NumPy array of laplacian(geopot)/f (array in padded_constants_dict['f-scaled laplacian geopot'])
    
    10) plon1d                      (lon+2)
            1D NumPy array of padded longitude values (in degrees).

    11) plat1d                      (lat+2)
            1D NumPy array of padded latitude values (in degrees).
    
    12) peta1d                      (layers+2)
            1D NumPy array of padded eta values.    


    Returns a 3D NumPy array (lon, lat, layer) containing the residuals of the nonlinear balance equation at
    every point on the lat x lon x layer grid.
'''
@njit( float64[:,:,:]( float64[:,:,:], float64[:,:,:], float64[:,:,:], float64[:,:,:], float64[:,:,:], float64[:,:,:], float64[:,:,:], float64[:,:,:], float64[:,:,:], float64[:], float64[:], float64[:] ) )
def eval_nonlinear_balance_residuals( pstreamfunc3d, alpha_x3d, alpha_y3d, alpha_xx3d, alpha_xy3d, alpha_yy3d, coriolis_param3d, ygrad_coriolis_param3d, fscaled_laplacian_geopot3d, plon1d, plat1d, peta1d  ):

    # Compute derivatives of the streamfunction
    # -----------------------------------------
    dpsi_dy = compute_df_dy_on_pres_surface( 
                    pstreamfunc3d, alpha_x3d, plon1d, plat1d, peta1d
    )
    d2psi_dy2 = compute_d2f_dy2_on_pres_surface(  
                    pstreamfunc3d, alpha_x3d, alpha_y3d, alpha_yy3d, plon1d, plat1d, peta1d 
    )
    d2psi_dxdy = compute_d2f_dxdy_on_pres_surface(  
                    pstreamfunc3d, alpha_x3d, alpha_y3d, alpha_xy3d, plon1d, plat1d, peta1d 
    )
    d2psi_dx2 = compute_d2f_dx2_on_pres_surface(  
                    pstreamfunc3d, alpha_x3d, alpha_y3d, alpha_xx3d, plon1d, plat1d, peta1d 
    )


    # Unpadded dimensions
    nlon = len(plon1d) - 2
    nlat = len(plat1d) - 2
    neta = len(peta1d) - 2


    # Evaluate residuals of nonlinear balance equations
    # --------------------------------------------------
    # Init array
    residuals = np.zeros( (nlon, nlat, neta), dtype='f8' )
    
    # Evaluate laplacian of streamfunction
    residuals += d2psi_dx2[1:-1,1:-1,1:-1] + d2psi_dy2[1:-1,1:-1,1:-1]

    # # Evaluate grad(psi) dot grad(f)
    # residuals += dpsi_dy[1:-1,1:-1,1:-1] * ygrad_coriolis_param3d[1:-1,1:-1,1:-1]

    # Evaluate "advection term"
    term = np.power( d2psi_dxdy[1:-1,1:-1,1:-1], 2) - d2psi_dx2[1:-1,1:-1,1:-1] * d2psi_dy2[1:-1,1:-1,1:-1] 
    term *= 2/coriolis_param3d[1:-1,1:-1,1:-1]
    residuals -= term

    # Evaluate geopotential term
    residuals -= fscaled_laplacian_geopot3d[1:-1,1:-1,1:-1]

    # print( residuals.min(), np.percentile( residuals, 25), np.median( residuals), np.percentile(residuals, 75), residuals.max())
    

    # Remap the nonlinear balance evaluations into vector and return
    # --------------------------------------------------------------
    rescale_factor = np.mean( np.abs( fscaled_laplacian_geopot3d[1:-1,1:-1,1:-1] ) ) 

    return residuals / rescale_factor


   















'''
    Function to pad streamfunction
'''
@njit( float64[:,:,:]( float64[:,:,:], float64[:], float64[:] ) )
def pad_streamfunc( streamfunc3d, lon1d, lat1d ):


    # Dimensions of unpadded arrays
    nlon , nlat, neta = streamfunc3d.shape


    # Generate padded 3D streamfunciton array
    # ---------------------------------------
    # Init array
    pstreamfunc3d = np.empty( (nlon+2, nlat+2, neta+2), dtype='f8' )
    pstreamfunc3d[1:-1,1:-1,1:-1] = streamfunc3d

    # Topmost boundary values for streamfunction (no slip condition)
    pstreamfunc3d[1:-1,1:-1,-1] = pstreamfunc3d[1:-1,1:-1,-2]

    # Bottommost boundary values for streamfunction (no slip condition)
    pstreamfunc3d[1:-1,1:-1,0] = pstreamfunc3d[1:-1,1:-1,1]

    # Spherical symmetry padding
    plon1d, plat1d, pstreamfunc3d[:,:,:] = pad_field_due_to_spherical_symmetry( pstreamfunc3d[1:-1,1:-1,:], lon1d, lat1d)
    
    return pstreamfunc3d









'''
    Function to evaluate gradient of cost function

    Input:
    ------
    1) pstreamfunc3d                (lon+2, lat+2, layer+2)
            3D NumPy array of padded streamfunction values

    2) alpha_x3d                    (lon+2, lat+2, layer+2)
            3D NumPy array of padded alpha_x values (array in padded_constants_dict['alpha_x'])

    3) alpha_y3d                    (lon+2, lat+2, layer+2)
            3D NumPy array of padded alpha_y values (array in padded_constants_dict['alpha_y'])

    4) alpha_xx3d                   (lon+2, lat+2, layer+2)
            3D NumPy array of padded alpha_xx values (array in padded_constants_dict['alpha_xx'])

    5) alpha_yy3d                   (lon+2, lat+2, layer+2)
            3D NumPy array of padded alpha_xy values (array in padded_constants_dict['alpha_yy'])

    6) alpha_xy3d                   (lon+2, lat+2, layer+2)
            3D NumPy array of padded alpha_xy values (array in padded_constants_dict['alpha_xy'])

    7) coriolis_param3d             (lon+2, lat+2, layer+2)
            3D NumPy array of padded coriolis parameter values (array in padded_constants_dict['coriolis'])

    8) ygrad_coriolis_param3d       (lon+2, lat+2, layer+2)
            3D NumPy array of padded coriolis beta parameter values (array in padded_constants_dict['coriolis ygrad'])

    9) fscaled_laplacian_geopot3d   (lon+2, lat+2, layer+2)
            3D NumPy array of laplacian(geopot)/f (array in padded_constants_dict['f-scaled laplacian geopot'])
    
    10) plon1d                      (lon+2)
            1D NumPy array of padded longitude values (in degrees).

    11) plat1d                      (lat+2)
            1D NumPy array of padded latitude values (in degrees).
    
    12) peta1d                      (layers+2)
            1D NumPy array of padded eta values.    


    Returns a 1D NumPy array (lon * lat * layer) containing the gradient of the cost function
'''
@njit( float64[:]( float64[:,:,:], float64[:,:,:], float64[:,:,:], float64[:,:,:], float64[:,:,:], float64[:,:,:], float64[:,:,:], float64[:,:,:], float64[:,:,:], float64[:], float64[:], float64[:] ) )
def fast_eval_cost_grad( pstreamfunc3d, alpha_x3d, alpha_y3d, alpha_xx3d, alpha_xy3d, alpha_yy3d, coriolis_param3d, ygrad_coriolis_param3d, fscaled_laplacian_geopot3d, plon1d, plat1d, peta1d  ):

    # Unpadded dimensions
    nlon = len(plon1d) - 2
    nlat = len(plat1d) - 2
    neta = len(peta1d) - 2

    # Prep padded residuals
    presiduals3d = np.empty( (nlon+2, nlat+2, neta+2), dtype='f8' )
    presiduals3d_pert = np.empty( (nlon+2, nlat+2, neta+2), dtype='f8' )


    # First, evaluate current residuals and spherical-pad the array
    tmp1, tmp2, presiduals3d[:,:,1:-1] = pad_field_due_to_spherical_symmetry(
        eval_nonlinear_balance_residuals( pstreamfunc3d, alpha_x3d, alpha_y3d, 
            alpha_xx3d, alpha_xy3d, alpha_yy3d, coriolis_param3d, ygrad_coriolis_param3d, 
            fscaled_laplacian_geopot3d, plon1d, plat1d, peta1d ),
        plon1d[1:-1], plat1d[1:-1]
    )
    # --- residuals3d has dimensions (lon, lat, layer)

    # Pad residuals
    presiduals3d[:,:,0] = presiduals3d[:,:,1]
    presiduals3d[:,:,-1] = presiduals3d[:,:,-2]

    # Init array to hold cost function gradient
    cost_grad3d = np.empty( (nlon, nlat, neta), dtype='f8' )

    # Make copies of streamfunction
    streamfunc3d = pstreamfunc3d[1:-1,1:-1,1:-1] * 1
    streamfunc3d_pert = streamfunc3d *1
    pstreamfunc3d_pert = np.empty( pstreamfunc3d.shape, dtype='f8')


    # Perform "stencil" calculation of cost gradient
    # -----------------------------------------------
    # Due to the calculation stencil of the spatial derivatives, it is possible to evaluate
    # the cost function gradient at multiple locations simultaneously.
    for i in range(3):
        for j in range(3):
            for k in range(3):
                
                # Perturb streamfunction
                streamfunc3d_pert[i::3, j::3, k::3] += 1000

                # Pad the perturbed streamfunction
                pstreamfunc3d_pert[:,:,:] = pad_streamfunc( streamfunc3d_pert, plon1d[1:-1], plat1d[1:-1] )

                # Compute perturbed residuals and pad the perturbed residuals
                tmp1, tmp2, presiduals3d_pert[:,:,1:-1] = pad_field_due_to_spherical_symmetry(
                    eval_nonlinear_balance_residuals( pstreamfunc3d_pert, alpha_x3d, alpha_y3d, 
                        alpha_xx3d, alpha_xy3d, alpha_yy3d, coriolis_param3d, ygrad_coriolis_param3d, 
                        fscaled_laplacian_geopot3d, plon1d, plat1d, peta1d ),
                    plon1d[1:-1], plat1d[1:-1] 
                )
                presiduals3d_pert[:,:,0] = presiduals3d_pert[:,:,1]
                presiduals3d_pert[:,:,-1] = presiduals3d_pert[:,:,-2]

                # Compute summand of the cost function gradient
                summand = ( (presiduals3d_pert - presiduals3d)/1000 ) * presiduals3d*2
                # print( np.abs(summand).max())

                # Compute gradient at chosen locations by summing up the summands
                cost_grad3d[i::3, j::3, k::3] = 0.
                for i1 in np.arange(i, nlon)[::3]:
                    for j1 in np.arange(j, nlat)[::3]:
                        for k1 in np.arange(k, neta)[::3]:
                            for di in range(-1,2):
                                for dj in range(-1,2):
                                    for dk in range(-1,2):
                                        cost_grad3d[i1,j1,k1]+= (summand[1+i1+di,1+j1+dj,1+k1+dk])

                # ------ End of cost function gradient calculation for selected locations

                # Reset value
                streamfunc3d_pert[i::3, j::3, k::3] = streamfunc3d[i::3, j::3, k::3]

            # --- End of loop over layer index
        # --- End of loop over latitude index
    # --- End of loop over longitude index
    
    # Return gradient of cost function as a 1D array
    return cost_grad3d.reshape( nlon*nlat*neta ) #*1e8






















































'''
    FUNCTIONS TO EVALUATE NONLINEAR BALANCE COST FUNCTION AND ITS GRADIENT
'''




'''
    Function to compute the constants needed to evaluate nonlinear flow equation

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
def compute_constants_needed_to_evaluate_nonlinear_flow_equation( pres3d, psurf2d, ptop2d, hgt3d, terrain2d, hgttop2d, lon1d, lat1d ):

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


    # Compute all alpha values
    alpha_x  =  compute_alpha_x( peta1d, ppres3d, plon1d, plat1d )
    alpha_y  =  compute_alpha_y( peta1d, ppres3d, plon1d, plat1d )
    alpha_xx = compute_alpha_xx( peta1d, ppres3d, plon1d, plat1d )
    alpha_yy = compute_alpha_yy( peta1d, ppres3d, plon1d, plat1d )
    alpha_xy = compute_alpha_xy( peta1d, ppres3d, plon1d, plat1d )


    # Compute laplacian of geopotential field
    laplacian_geopot  = compute_d2f_dx2_on_pres_surface( pgeopot3d, alpha_x, alpha_y, alpha_xx, plon1d, plat1d, peta1d )
    laplacian_geopot += compute_d2f_dy2_on_pres_surface( pgeopot3d, alpha_x, alpha_y, alpha_yy, plon1d, plat1d, peta1d )

    # Generate coriolis parameter at all locations
    platmesh, plonmesh = np.meshgrid( plat1d, plon1d )
    coriolis_param3d = np.empty( ppres3d.shape, dtype='f8' )
    coriolis_param3d_xgrad = np.zeros( ppres3d.shape, dtype='f8' )
    coriolis_param3d_ygrad = np.empty( ppres3d.shape, dtype='f8' )
    for k in range( len(peta1d) ):
        coriolis_param3d[:,:,k] =  2 * EARTH_ANGULAR_SPEED * np.sin( DEG_2_RAD * platmesh )

    # Generate y-gradient of coriolis parameter
    coriolis_param3d_ygrad = compute_df_dy_on_eta_surface( 
            coriolis_param3d, plon1d, plat1d
    )
    

    # Dictionary of padded constants
    padded_constants_dict = {}
    padded_constants_dict['coriolis']                   = coriolis_param3d * 1
    padded_constants_dict['coriolis xgrad']             = coriolis_param3d_xgrad * 1
    padded_constants_dict['coriolis ygrad']             = coriolis_param3d_ygrad * 1
    padded_constants_dict['f-scaled laplacian geopot']  = laplacian_geopot*1 /coriolis_param3d * 1
    padded_constants_dict['alpha_x' ]                   = alpha_x * 1
    padded_constants_dict['alpha_y' ]                   = alpha_y * 1
    padded_constants_dict['alpha_xx']                   = alpha_xx * 1
    padded_constants_dict['alpha_yy']                   = alpha_yy * 1
    padded_constants_dict['alpha_xy']                   = alpha_xy * 1
    padded_constants_dict['plat1d']                     = plat1d * 1
    padded_constants_dict['plon1d']                     = plon1d * 1
    padded_constants_dict['peta1d']                     = peta1d * 1

    # # Monge-Ampere condition
    # flag_violated = (
    #     padded_constants_dict['f-scaled laplacian geopot'] + padded_constants_dict['coriolis'] / 2 <= 0
    # )
    # padded_constants_dict['corrected f-scaled laplacian geopot'] = padded_constants_dict['f-scaled laplacian geopot']*1
    # padded_constants_dict['corrected f-scaled laplacian geopot'][flag_violated] = -0.49 * padded_constants_dict['coriolis'][flag_violated] 

    return padded_constants_dict
















'''
    Function to compute cost function for nonlinear balance

    Note that this function interfaces with an accelerated function to compute the gradient.
    
    Inputs:
    --------
    1) streamfunc1d (lon * lat * layer)
            1D NumPy array containing flattened version of streamfunction on global gaussian grid.
    2) padded_constants_dict
            Python dictionary containing padded arrays of constants

    Returns the squared sum of all nonlinear balance residuals
'''
def cost_function_nonlinear_balance(streamfunc1d, padded_constants_dict ):

    # Extract constants
    lon1d  = padded_constants_dict['plon1d'][1:-1]
    lat1d  = padded_constants_dict['plat1d'][1:-1]
    peta1d = padded_constants_dict['peta1d']


    # Dimensions of unpadded arrays
    nlon = len(lon1d)
    nlat = len(lat1d)
    neta = len(peta1d) -2


    # Generate padded 3D streamfunciton array
    # ---------------------------------------
    # Init array
    pstreamfunc3d = np.empty( (nlon+2, nlat+2, neta+2), dtype='f8' )
    pstreamfunc3d[1:-1,1:-1,1:-1] = streamfunc1d.reshape([nlon, nlat, neta])

    # Topmost boundary values for streamfunction (no slip condition)
    pstreamfunc3d[1:-1,1:-1,-1] = pstreamfunc3d[1:-1,1:-1,-2]

    # Bottommost boundary values for streamfunction (no slip condition)
    pstreamfunc3d[1:-1,1:-1,0] = pstreamfunc3d[1:-1,1:-1,1]

    # Spherical symmetry padding
    plon1d, plat1d, pstreamfunc3d[:,:,:] = pad_field_due_to_spherical_symmetry( pstreamfunc3d[1:-1,1:-1,:], lon1d, lat1d)
    


    # Evaluate residuals
    residuals3d = eval_nonlinear_balance_residuals( 
        pstreamfunc3d, 
        padded_constants_dict['alpha_x'],  padded_constants_dict['alpha_y'], 
        padded_constants_dict['alpha_xx'], padded_constants_dict['alpha_xy'], 
        padded_constants_dict['alpha_yy'], padded_constants_dict['coriolis'], 
        padded_constants_dict['coriolis ygrad'], 
        padded_constants_dict['f-scaled laplacian geopot'], 
        plon1d, plat1d, peta1d
    )

    # Return cost function
    return np.sum( residuals3d**2 ) #/ (nlon*nlat*neta )
















'''
    Function to evaluate the cost function gradient efficiently
    
    Inputs:
    --------
    1) streamfunc1d (lon * lat * layer)
            1D NumPy array containing flattened version of streamfunction on global gaussian grid.
    2) padded_constants_dict
            Python dictionary containing padded arrays of constants

    Returns the gradient of the cost function in the form of a 1D NumPy array (lon * lat * layer).
'''
def cost_gradient_nonlinear_balance(streamfunc1d, padded_constants_dict ):

    # Extract constants
    lon1d  = padded_constants_dict['plon1d'][1:-1]
    lat1d  = padded_constants_dict['plat1d'][1:-1]
    peta1d = padded_constants_dict['peta1d']

    # Dimensions of unpadded arrays
    nlon = len(lon1d)
    nlat = len(lat1d)
    neta = len(peta1d) -2

    # Generate padded 3D streamfunction array
    pstreamfunc3d = pad_streamfunc( 
        streamfunc1d.reshape( [nlon, nlat, neta] ), lon1d, lat1d 
    )
    
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

    # Compute cost function gradient
    cost_grad1d = fast_eval_cost_grad( 
        pstreamfunc3d, 
        padded_constants_dict['alpha_x'],  padded_constants_dict['alpha_y'], 
        padded_constants_dict['alpha_xx'], padded_constants_dict['alpha_xy'], 
        padded_constants_dict['alpha_yy'], padded_constants_dict['coriolis'], 
        padded_constants_dict['coriolis ygrad'], 
        padded_constants_dict['f-scaled laplacian geopot'], 
        padded_lon1d, padded_lat1d, peta1d
    )

    return cost_grad1d #/ (nlon*nlat*neta )


















'''
    Function to compute U and V field from streamfunction

    Inputs:
    --------
    1) streamfunc3d (lon, lat, layer)
            3D NumPy array containing flattened version of streamfunction on global gaussian grid.
    2) padded_constants_dict
            Python dictionary containing padded arrays of constants

    Returns two 3D NumPy arrays (lon, lat, layer): U wind (m/s) and V wind (m/s)
'''
def convert_streamfunc_to_u_and_v( streamfunc3d, padded_constants_dict ):

    # Pad the streamfunction for easy computation of gradients
    pstreamfunc3d = pad_streamfunc( 
                        streamfunc3d, padded_constants_dict['plon1d'][1:-1], 
                        padded_constants_dict['plat1d'][1:-1]
    )

    # Compute u wind (defined as negative y-gradient of streamfunction on pressure surface)
    u3d = compute_df_dy_on_pres_surface(
        pstreamfunc3d, padded_constants_dict['alpha_y'], padded_constants_dict['plon1d'][1:-1], 
        padded_constants_dict['plat1d'][1:-1], 
    )[1:-1, 1:-1, 1:-1]
    u3d *= -1

    # Compute v wind (defined as x-gradient of streamfunction on pressure surface)
    v3d = compute_df_dx_on_pres_surface(
        pstreamfunc3d,  padded_constants_dict['alpha_x'], padded_constants_dict['plon1d'][1:-1], 
        padded_constants_dict['plat1d'][1:-1], 
    )[1:-1, 1:-1, 1:-1]

    return u3d, v3d








'''
    Function to sanity check cost function gradient calculation.
'''
def SANITY_CHECK_cost_function_gradient():

    # Load python packages for plotting
    import matplotlib.pyplot as plt

    # Grid settings (must be even numbers)
    nlat = 8
    nlon = nlat * 2
    nlvl = 4

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
    for kk in range(nlvl):
        ref_height = np.log( pres1d[kk]/100000 ) * -12000/1.7
        height3d[:,:,kk] = ref_height + amplitude * np.cos( 2*latmesh * PI/180 ) * np.cos( 2*lonmesh * PI/180 )
    # --- End of loop over model layers

    # Generate meshes for terrain and model top height
    terrain2d = latmesh*0
    hgttop2d = np.log( 2000./100000 ) * -12000/1.7 + amplitude * np.cos( 2*latmesh * PI/180 ) * np.cos( 2*lonmesh * PI/180 )
    
    # Construct 3d pressure field
    pres3d = np.empty( (nlon, nlat, nlvl), dtype='f8')
    for kk in range(nlvl):
        pres3d[:,:,kk] = pres1d[kk]

    # Generate dictionary of padded constants
    padded_constants_dict = compute_constants_needed_to_evaluate_nonlinear_flow_equation( 
        pres3d, psurf2d, ptop2d, height3d, terrain2d, hgttop2d, lon1d, lat1d 
    )

    # Compute gradient and make plot
    streamfunc1d = np.zeros( (nlon*nlat*nlvl), dtype='f8') 
    grad1d = cost_gradient_nonlinear_balance(streamfunc1d, padded_constants_dict)
    grad3d = grad1d.reshape([nlon, nlat, nlvl])

    # Compare laplacian of geopotential and gradient
    lap_geopot = padded_constants_dict['f-scaled laplacian geopot'][1:-1,1:-1,2]
    crange = np.linspace(-1e-5, 1e-5, 11)
    plt.contourf( lon1d, lat1d, lap_geopot.T, crange, cmap='RdBu_r', extend='both' )
    cbar = plt.colorbar()
    cbar.ax.set_ylabel( r'$\dfrac{1}{f} \nabla^2 \phi$')
    
    plt.contour( lon1d, lat1d, grad3d[:,:,2].T,  [-1e-9,0,1e-9], colors='k', linestyles=[':','--','-'])

    plt.savefig('check_cost_func_grad.png')
    plt.close()

    return




























'''
    SANITY CHECK: Diagnosing nonlinearly balanced flow from zonally-symmetric height fields
'''
def sanity_check_nonlinear_flow_estimation():

    # Load python packages for plotting
    # from matplotlib import use as mpl_use
    # mpl_use('agg')
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize

    # Grid settings (must be even numbers)
    nlat = 10
    nlon = nlat * 2
    nlvl = 2


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
    meridional_variation = amplitude * np.cos( latmesh * PI/180) #np.exp( -0.5 * (latmesh)**2/(10**2) ) 
    zonal_variation = 0 #np.abs(np.sin( 0.5*latmesh * PI/180)) * np.cos(2* lonmesh * PI/180) * amplitude /2

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


    # Generate dictionary of padded constants
    padded_constants_dict = compute_constants_needed_to_evaluate_nonlinear_flow_equation( 
        pres3d, psurf2d, ptop2d, height3d, terrain2d, hgttop2d, lon1d, lat1d 
    )

    plt.contour( lon1d, lat1d, height3d[:,:,-1].T, [10500, 11000, 11500], colors='k', linestyles=[':','--','-'])
    plt.contourf( lon1d, lat1d, padded_constants_dict['f-scaled laplacian geopot'][1:-1,1:-1,-2].T , 11, cmap='RdBu_r')
    plt.colorbar()
    plt.savefig('check_laplacian.png')
    plt.close()

    # Visualize geopotential
    geopot3d = 9.81 * height3d
    plt.contourf( lon1d, lat1d, geopot3d[:,:,1].T, 11, cmap = 'RdBu_r')
    plt.colorbar()
    plt.savefig('check_geopotential.png')
    plt.close()

    streamfunc3d = geopot3d / padded_constants_dict['coriolis'][1:-1,1:-1,1:-1]
    streamfunc1d = streamfunc3d.reshape(nlon*nlat*nlvl)
    #np.zeros( (nlon*nlat*nlvl), dtype='f8') #(height3d*1).reshape(nlon*nlat*nlvl)
    print( 'cost func', cost_function_nonlinear_balance(streamfunc1d, padded_constants_dict) )
    # grad = cost_gradient_nonlinear_balance(streamfunc1d, padded_constants_dict)
    # print( 'cost grad', np.sqrt(np.mean( np.power( grad,2 ) ) ), np.abs(grad).max() )

    # Visualize first guess streamfunction
    plt.contourf( lon1d, lat1d, streamfunc3d[:,:,1].T, 11, cmap = 'RdBu_r')
    plt.colorbar()
    plt.savefig('check_streamfunc_firstguess.png')
    plt.close()
    # quit()


    # Derive streamfunction by minimizing residuals of nonlinear balance equation
    print('starting to solve streamfunction')
    res = minimize( 
        cost_function_nonlinear_balance, streamfunc1d,
        args = (padded_constants_dict), # jac = cost_gradient_nonlinear_balance,
        options = {'disp' : True} #{'maxiter' : 10, 'disp' : True}
    )
    print('managed to run streamfunction solver')

    print(res.x.min(), res.x.max())
    print( dir(res))

    # Visualize solution
    plt.contourf( lon1d, lat1d, res.x.reshape((nlon, nlat, nlvl))[:,:,1].T, 11, cmap = 'RdBu_r')
    plt.colorbar()
    plt.savefig('check_streamfunc_soln.png')
    plt.close()

    # u3d, v3d = convert_streamfunc_to_u_and_v(  )


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

    # SANITY_CHECK_cost_function_gradient()

    sanity_check_nonlinear_flow_estimation()


    print('meow')