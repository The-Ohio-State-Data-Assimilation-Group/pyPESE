#!/bin/python3
'''
    PYTHON FUNCTIONS TO DIAGNOSE GEOSTROPHICALLY-BALANCED FLOW OR GEOPOTENTIAL
    Written by: Man-Yau (Joseph) Chan

    DESCRIPTION:
    ------------
    Much of the atmospheric/oceanic flow is in geostrophic balance. The functions
    diagnoses either (1) geostrophic flow from given thermodynamic data or (2) 
    geostrophic geopotential from given horizontal velocity data.

    
    
    Note that the geostrophic equation is written in (x,y,P) coordinates. However, 
    many weather models use terrain-following vertical coordinates instead of isobaric
    vertical coordinates. The approach of Kasahara (1974) is used to estimate geostrophic
    flow on terrain-following coordinates.
    
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

# Load all functions relating to taking spatial derivatives & spherical padding
from pyPESE.utilities.global_latlon_grid import *

# Load functions relating to vertical interpolation
from pyPESE.utilities.vertical_interp import interp_geopotential_to_plvls, basic_interpolate_to_pressure_levs, basic_interpolate_to_eta_levs


t0 = time()

# flag for caching
jit_cache_flag = False

































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
#njit( nbtuple( (float64[:,:,:], float64[:,:,:], float64[:,:,:]) )( float64[:,:,:], float64[:,:], float64[:,:], float64[:,:,:], float64[:,:], float64[:,:], float64[:], float64[:]) )
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
    FUNCTIONS TO DIAGNOSE GEOSTROPHIC HEIGHTS FROM NON-DIVERGENT FLOW ON PRESSURE
    SURFACES
'''



'''
    FUNCTION TO COMPUTE VORTICITY (NEEDED TO ESTIMATE STREAMFUNCTION)

    Inputs:
    -------
    1) uwind3d (lon, lat, level)
        3D NumPy array of zonal wind velocities on PRESSURE SURFACES.
    2) vwind3d (lon, lat, level)
        3D NumPy array of meridional wind velocities on PRESSURE SURFACES.
    3) lon1d (lon)
            1D NumPy array of longitude values (in degrees).
    4) lat1d (lat)
            1D NumPy array of latitude values (in degrees).

    Returns a 3D NumPy array (lon, lat, level) containing those vorticity values
'''
#njit( float64[:,:,:]( float64[:,:,:], float64[:,:,:], float64[:,:], float64[:,:] ) )
def compute_vorticity( uwind3d, vwind3d, lon1d, lat1d ):

    plon1d, plat1d, puwind3d = pad_field_due_to_spherical_symmetry( uwind3d, lon1d, lat1d )
    plon1d, plat1d, pvwind3d = pad_field_due_to_spherical_symmetry( vwind3d, lon1d, lat1d )

    vort3d = ( 
        compute_df_dx_on_eta_surface( pvwind3d, plon1d, plat1d )[1:-1,1:-1,:]
        - compute_df_dy_on_eta_surface( puwind3d, plon1d, plat1d )[1:-1,1:-1,:]
    )

    return vort3d









'''
    FUNCTION TO COMPUTE STREAMFUNCTION VIA SPHERICAL HARMONICS ON PRESSURE SURFACES

    Inputs:
    -------
    1) uwind3d (lon, lat, level)
        3D NumPy array of zonal wind velocities on PRESSURE SURFACES.
    2) vwind3d (lon, lat, level)
        3D NumPy array of meridional wind velocities on PRESSURE SURFACES.
    3) lon1d (lon)
            1D NumPy array of longitude values (in degrees).
    4) lat1d (lat)
            1D NumPy array of latitude values (in degrees).

    Returns a 3D NumPy array (lon, lat, level) containing streamfunction values
'''
def compute_streamfunc( uwind3d, vwind3d, lon1d, lat1d ):

    # Array dimensions
    nlon, nlat, nlvl = uwind3d.shape
    
    # First, compute vorticity on lat-lon grid
    vort3d = compute_vorticity( uwind3d, vwind3d, lon1d, lat1d )

    # Invert Poisson's equation
    streamfunc = spherical_invert_poisson_equation(vort3d)

    return streamfunc









'''
    FUNCTION TO SANITY CHECK STREAMFUNCTION CALCULATION ON PRESSURE SURFACES

    Inputs:
    -------
    1) uwind3d (lon, lat, level)
        3D NumPy array of zonal wind velocities on PRESSURE SURFACES.
    2) vwind3d (lon, lat, level)
        3D NumPy array of meridional wind velocities on PRESSURE SURFACES.
    3) lon1d (lon)
            1D NumPy array of longitude values (in degrees).
    4) lat1d (lat)
            1D NumPy array of latitude values (in degrees).
'''
def SANITY_CHECK_compute_streamfunc( uwind3d, vwind3d, lon1d, lat1d ):

    import matplotlib.pyplot as plt

    # Test function used to compute streamfunction
    streamfunc = compute_streamfunc( 
        uwind3d, vwind3d, lon1d, lat1d
    ) 

    # Compute vorticity from wind fields
    vort3d_real = compute_vorticity( uwind3d, vwind3d, lon1d, lat1d )

    # Compute vorticity from streamfunction
    vort3d_test = compute_d2f_dx2_on_eta_surface(
        streamfunc, lon1d, lat1d
    )
    vort3d_test += compute_d2f_dy2_on_eta_surface(
        streamfunc, lon1d, lat1d
    )


    # Make figure to compare the streamfunction's vorticity to the original values
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8,8) )
    axs[0,0].contourf( lon1d, lat1d, vort3d_test[:,:,0].T, np.linspace(-1e-4,1e-4,10), cmap='RdBu_r', extend='both')
    axs[0,0].set_title('Streamfunction-produced vorticity')
    axs[0,1].contourf( lon1d, lat1d, vort3d_real[:,:,0].T, np.linspace(-1e-4,1e-4,10), cmap='RdBu_r', extend='both')
    axs[0,1].set_title('Original vorticity')

    diff_vort = vort3d_test[:,:,0] - vort3d_real[:,:,0]
    diff_vort[np.abs(diff_vort)<1e-7] = np.nan
    axs[1,1].contourf( lon1d, lat1d, diff_vort.T, np.linspace(-1e-4,1e-4,10), cmap='RdBu_r', extend='both')
    axs[1,1].set_title('Test minus Original')
    plt.savefig('test_streamfunc_calc.png')


    return











'''
    FUNCTION TO COMPUTE GEOSTROPHICALLY BALANCED ZERO-MEAN GEOPOTENTIAL HEIGHTS
    
    Note: This height has a spatial mean of zero! Need to add a constant value to it.

    Inputs:
    -------
    1) uwind3d (lon, lat, level)
        3D NumPy array of zonal wind velocities on PRESSURE SURFACES.
    2) vwind3d (lon, lat, level)
        3D NumPy array of meridional wind velocities on PRESSURE SURFACES.
    3) lon1d (lon)
            1D NumPy array of longitude values (in degrees).
    4) lat1d (lat)
            1D NumPy array of latitude values (in degrees). 
'''
def compute_geostrophic_heights_from_horizontal_flow( uwind3d, vwind3d, lon1d, lat1d ):

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

    # Generate coriolis parameter gradient
    coriolis_beta3d = compute_df_dy_on_eta_surface( coriolis_param3d, lon1d, lat1d )


    # Estimate streamfunction
    streamfunc3d = compute_streamfunc(uwind3d, vwind3d, lon1d, lat1d)


    # Compute coriolis term in geostrophic poisson equation
    coriolis_terms = compute_vorticity( uwind3d, vwind3d, lon1d, lat1d ) * coriolis_param3d
    # (
    #     compute_d2f_dx2_on_eta_surface( streamfunc3d, lon1d, lat1d )
    #     + compute_d2f_dy2_on_eta_surface( streamfunc3d, lon1d, lat1d )
    # ) 
    # coriolis_terms += ( 
    #     compute_df_dy_on_eta_surface( streamfunc3d, lon1d, lat1d) 
    #     * (coriolis_beta3d / coriolis_param3d)
    # )
    # coriolis_terms *= coriolis_param3d
    

    # Invert Poisson equation to obtain mean-zero geopotential field
    zero_mean_geopot3d = spherical_invert_poisson_equation( coriolis_terms )
    zero_mean_geopot3d -= np.mean( np.mean( zero_mean_geopot3d, axis=0), axis=0 )


    # Return mean-zero geopotential heights
    return zero_mean_geopot3d / GRAVITY_ACCEL













'''
    FUNCTION TO COMPUTE GEOSTROPHICALLY BALANCED GEOPOTENTIAL HEIGHTS ON ETA LEVELS

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
def compute_geostrophic_heights_on_eta_lvls( uwind3d, vwind3d, pres3d, hgt3d, psurf2d, tsurf2d, terrain2d, lon1d, lat1d ):

    # Detect dimensions
    nlon, nlat, neta = uwind3d.shape
    nplvl = 101

    # Useful constants
    GRAVITY_ACCEL = 9.80665 
    
    # Pressure layers to do calculations on 
    plvls1d = np.linspace( pres3d.min(), pres3d.max(), nplvl)[::-1]

    # Interpolate U & V to desired pressure levels
    uwind_plvl = basic_interpolate_to_pressure_levs( pres3d, uwind3d, plvls1d)
    vwind_plvl = basic_interpolate_to_pressure_levs( pres3d, vwind3d, plvls1d)


    # Interpolate geopotential to pressure levels
    hgt_plvl = interp_geopotential_to_plvls( 
        hgt3d*GRAVITY_ACCEL, pres3d, psurf2d, tsurf2d, terrain2d, plvls1d.astype('f8')
    )
    hgt_plvl /= GRAVITY_ACCEL


    # Compute geostrophic heights with mean zero
    mean_zero_hgt = compute_geostrophic_heights_from_horizontal_flow( uwind_plvl, vwind_plvl, lon1d, lat1d )

    # Adjust the layerwise mean of the geostrophic height
    geostrophic_hgt_plvl = mean_zero_hgt + np.mean( np.mean( hgt_plvl, axis=0), axis=0)

    # Interpolate geostrophic heights from plvls back to eta lvls
    geostrophic_hgt3d = basic_interpolate_to_eta_levs( plvls1d, geostrophic_hgt_plvl, pres3d )


    return geostrophic_hgt3d







































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