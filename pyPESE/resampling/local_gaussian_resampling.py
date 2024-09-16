'''
    FUNCTIONS RELATING TO FAST LOCALIZED GAUSSIAN RESAMPLING
    =========================================================



'''

import numpy as np
from math import pi as PI
from numba import njit, float64 as nbf64, int64 as nbi64
import random
from pyPESE.resampling.gaussian_resampling import prep_transform_matrix_for_gaussian_coefficients, compute_unlocalized_gaussian_resampling_coefficients
# from gaussian_resampling import prep_transform_matrix_for_gaussian_coefficients, compute_unlocalized_gaussian_resampling_coefficients
from time import time



t0 = time()







'''
    Function to compute Gaspari-Cohn localization coefficients

    Inputs:
    --------
    1) dist_arr1d
            1D float64 NumPy array containing distances.
    2) obs_roi
            Scalar radius-of-influence to use with the GC99 function.
            Must have the same units as dist_arr1d.

'''
@njit( nbf64[:]( nbf64[:], nbf64 ) )
def GC99( dist_arr1d, obs_roi):


    c = obs_roi/2.0
    z = dist_arr1d * 1.
    
    coeff = dist_arr1d * 0.

    flags = (z<=c)
    coeff[ flags ] = - 0.25  *np.power(z[flags]/c, 5) \
                      + 0.5  *np.power(z[flags]/c, 4) \
                      +5./8.0*np.power(z[flags]/c, 3) \
                      -5./3.0*np.power(z[flags]/c, 2) \
                      + 1

    flags = z > c
    flags *= ( z<=2*c)
    coeff[ flags ] =   1.0/12  *np.power(z[flags]/c, 5) \
                       -1.0/2.0 *np.power(z[flags]/c, 4) \
                       +5.0/8.0 *np.power(z[flags]/c, 3) \
                       +5.0/3.0 *np.power(z[flags]/c, 2) \
                       -5.0 *z[flags]/c \
                       + 4 \
                       -(2.0/3.0)*np.power(z[flags]/c, -1)
    
    return  coeff











'''
    Function to compute Haversine distance (aka, great circle distance)

    Hard-coded value of R is Earth's radius of 6,378,000 meters

    Inputs:
    -------
    1) lon1_in_rad1d
            1D float64 NumPy array of longitude values (radians)
    2) lon2_in_rad
            Float64 scalar reference longitude value (radians) to compute distances from
    3) lat1_in_rad1d
            1D float64 NumPy array of latitude values (radians)
    4) lat2_in_rad
            Float64 scalar reference latitude value (radians) to compute distances from

    Note that lon1_in_rad1d and lat1_in_rad1d must have the same array dimensions

    Outputs:
    --------
    1) dist_in_meters
            1D NumPy array of great circle distances in meters.
            Has the same array dimension as lon1_in_rad1d & lat1_in_rad1d

'''
@njit( nbf64[:]( nbf64[:], nbf64, nbf64[:], nbf64 ) )
def compute_great_circle_dist( lon1_in_rad1d, lon2_in_rad, lat1_in_rad1d, lat2_in_rad ):

    R = 6378*1000.

    # Compute squared angular distance
    dist_in_meters = ( 
        1 - np.cos( lat2_in_rad - lat1_in_rad1d ) 
        + np.cos( lat1_in_rad1d ) * np.cos( lat2_in_rad ) * (1 - np.cos( lon2_in_rad - lon1_in_rad1d) )
    ) / 2

    # Convert squared angular distance to distance in meters
    dist_in_meters = 2 * R * np.arcsin( np.sqrt( dist_in_meters ) )

    return dist_in_meters





























'''
    FUNCTIONS TO GENERATE GRIDDED NOISE NEEDED FOR LOCALIZED RESAMPLING

    Basic idea:
    1) Start with white noise defined on all model grid points
    2) Convolve white noise with localization function to construct spatially correlated noise

    However, because we need lots of noise samples per grid point, storing all that noise
    is likely memory intensive.

    Instead, we will have an "implicit" white noise grid.
    
'''



'''
    Function to materialize the implicit noise grid

    Inputs:
    -------
    1) lon1d (1D NumPy float array)
            Longitudes of all desired locations in degrees East
            Basically a flattened version of longitude meshgrid
    2) lat1d (1D NumPy float array)
            Latitudes of all desired locations in degrees North
            Basically a flattened version of latitude meshgrid
    3) ens_size
            Number of noise draws to make
            This is a scalar value of the NumPy 64-bit integer type
'''
@njit( nbf64[:,:]( nbf64[:], nbf64[:], nbi64 ) )
def materialize_implicit_noise_grid( lon1d, lat1d, ens_size ):

    # Shape of things
    n_pts = lon1d.shape[0]

    # Init noise array
    noise_arr = np.empty( (ens_size, n_pts ), dtype='f8')

    # Sample!
    for ipt in range( n_pts ):
        lat = lat1d[ipt]
        lon = lon1d[ipt]
        rng_seed = ((90-lat)*360 + (lon+180))*100
        np.random.seed( int(rng_seed) )
        noise_arr[:,ipt] = np.random.normal( loc=0, scale=1, size=ens_size )
    # --- End of loop over all positions
    
    return noise_arr






















'''
    Function generate noise at a location that is IMPLICIT correlated to noise 
    at other locations.

    The explicit version of this treatment is to generate spatially-correlated noise
    on a 2D grid via convolving a grid of white noise with the GC99 kernel.

    The advantages of the implicit treatment over the explicit treatment are:
    1) Less memory overhead
    2) Can be evoked in parallel

    The disadvantages of the implicit treatment over the explicit treatment are:
    1) Harder to understand
    2) If noise is being generated for multiple nearby locations, there will be 
       lots of redundant calculations.

    Inputs:
    -------
    1) all_lon1d (1D NumPy float64 array)
            Longitudes of ALL model locations in degrees East
            Basically a flattened version of longitude meshgrid
    2) all_lat1d (1D NumPy float64 array)
            Latitudes of ALL model locations in degrees North
            Basically a flattened version of latitude meshgrid
    3) targ_lon (scalar NumPy float64)
            Targetted location's longitude in degrees East
    4) targ_lat (scalar NumPy float64)
            Targetted location's latitude in degrees North
    5) roi_in_km (scalar NumPy float64)
            Radius of influence in kilometers
    6) ens_size
            Number of noise draws to make
            This is a scalar value of the NumPy 64-bit integer type
    
    Output:
    -------
    1) sample (1D NumPy float64 array)
            This 1D array contains ens_size draws of the implicitly correlated noise.
    
'''
@njit( nbf64[:]( nbf64[:], nbf64[:], nbf64, nbf64, nbf64, nbi64 ) )
def implicit_gen_spatially_correlated_sample( all_lon1d, all_lat1d, targ_lon, targ_lat, roi_in_km, ens_size ):

    # Constants
    DEG2PI = PI/180
    
    # Compute great circle distances between target site and all other locations
    dist_in_km = compute_great_circle_dist( all_lon1d*DEG2PI, targ_lon*DEG2PI, all_lat1d*DEG2PI, targ_lat*DEG2PI )/1000

    # Determine all points that are within 1 ROI of the target site (aka, "nearby points")
    flag_in_roi = dist_in_km <= roi_in_km
    nearby_lon1d = all_lon1d[flag_in_roi]
    nearby_lat1d = all_lat1d[flag_in_roi]
    nearby_dists  = dist_in_km[flag_in_roi]

    # Materialize the noise field on nearby points
    nearby_random = materialize_implicit_noise_grid( nearby_lon1d, nearby_lat1d, ens_size )

    # Generate localization coefficients on nearby locations
    nearby_loc_coeff = GC99( nearby_dists, roi_in_km )

    # Construct sample from the materialized nearby noise field
    sample = np.sum( nearby_random * nearby_loc_coeff, axis=1 )

    return sample










# Function to sanity check gen_spatially_correlated_sample
def SANITY_CHECK_gen_spatially_correlated_sample():

    print('Running sanity check function for gen_spatially_correlated_sample')

    import matplotlib.pyplot as plt

    # Generate lat lon grid
    latmesh, lonmesh = np.meshgrid( 
        np.linspace(-88, 88, 45 ),
        np.linspace(-178, 178, 90)
    )
    ens_size=800

    # Choose an ROI in kilometers
    roi_in_km = 111 * 40               # Basically 40 degree ROI on equator

    
    t0 = time()
    # Compute many many samples
    all_samples = njit_draw_spatially_localized_samples( 
        lonmesh.astype('f8'), 
        latmesh.astype('f8'), 
        np.float64(roi_in_km), 
        np.int64(ens_size)
    )
    print('Large sample draws completed in ', time() - t0, ' seconds.')
    
    # Converting samples to standard normal
    all_samples = (
        ( all_samples.transpose( 2, 0, 1 ) - np.mean( all_samples, axis =-1) )
        / np.std( all_samples, axis=-1, ddof=1)
    ).transpose( 2, 1, 0)


    # Compute correlation statistics
    ref_ilon = 4
    ref_ilat = 4
    ref_samples = all_samples[ref_ilon, ref_ilat,:]
    cov = np.mean( all_samples * ref_samples, axis=-1 ) * ens_size / (ens_size-1)
    plt.contourf( lonmesh, latmesh, cov, np.linspace(-0.9, 0.9, 10), extend='both', cmap='RdBu_r' )
    plt.colorbar()
    plt.savefig('covariance_of_localized_noise.png')
    plt.close()

    return




# Function to accelerate the drawing process done in the sanity check
@njit( nbf64[:,:,:] ( nbf64[:,:], nbf64[:,:], nbf64, nbi64 ) )
def njit_draw_spatially_localized_samples( lonmesh, latmesh, roi_in_km, ens_size):

    # Flatten stuff!
    all_lon1d = lonmesh.flatten()
    all_lat1d = latmesh.flatten()

    # Compute dimensions
    nlat, nlon = latmesh.shape
    
    # Init samples
    all_samples = np.empty( (nlon, nlat, ens_size) )

    # Loop over all locations
    for ilon in range(nlon):
        for ilat in range(nlat):

            # Determine target location
            targ_lon = lonmesh[ilat, ilon] 
            targ_lat = latmesh[ilat, ilon]

            # Determine samples
            all_samples[ilon, ilat, :] = (
                implicit_gen_spatially_correlated_sample( 
                    all_lon1d, all_lat1d, targ_lon, targ_lat, roi_in_km, ens_size
                )
            )
        # --- End of loop over latitudes
    # --- End of loop over longitudes

    return all_samples



    

            























'''
    FUNCTION TO GENERATE LOCALIZED GAUSSIAN RESAMPLING COEFFICIENTS THAT GUARANTEES
    LOCAL VARIANCE CONSERVATION AND LOCALIZED SPATIAL COVARIANCES

    The basic procedure is the same as that of matrix E described in Chan et al 2020 
    MWR paper on BGEnKF.

    The only modification is that the W matrix of Chan et al 2020 Appendix B varies
    spatially. 

    Note that the resulting virtual members are not iid.

    Inputs:
    -------
    1) all_lon1d (1D NumPy float64 array)
            Longitudes of ALL model locations in degrees East
            Basically a flattened version of longitude meshgrid
    2) all_lat1d (1D NumPy float64 array)
            Latitudes of ALL model locations in degrees North
            Basically a flattened version of latitude meshgrid
    3) targ_lon (scalar NumPy float64)
            Targetted location's longitude in degrees East
    4) targ_lat (scalar NumPy float64)
            Targetted location's latitude in degrees North
    5) roi_in_km (scalar NumPy float64)
            Radius of influence in kilometers
    6) ens_size_original (scalar integer)
            Original number of ensemble members
    7) ens_size_virtual (scalar integer)
            Number of virtual members to create
    
    Outputs:
    --------
    1) resampling_matrix ( ens_size_original x ens_size_virtual float matrix)
            matrix of linear combination coefficients (matrix E in MWR paper)
'''
def compute_localized_gaussian_resampling_coefficients( all_lon1d, all_lat1d, targ_lon, targ_lat, roi_in_km, ens_size_original, ens_size_virtual, rng_seed=0 ):

    # Rename variables to use a different notation.
    N = ens_size_original
    M = ens_size_virtual

    # Seed the NumPy random number generator -- useful for parallelization.
    np.random.seed(rng_seed)

    # Materialize spatially-correlated noise at the targetted location
    # This is the "localized" version of Chan et al 2020 Appendix B Step 1.
    W = implicit_gen_spatially_correlated_sample( 
            all_lon1d, all_lat1d, targ_lon, targ_lat, roi_in_km, N*M
    ).reshape(N,M)
    
    # Appendix B Step 2
    W = np.matrix( (W.T - np.mean(W, axis=1)).T )

    # Appendix B Step 3 and 4
    C_W = W * W.T
    inv_L_W = np.linalg.inv(np.linalg.cholesky( C_W ))
    inv_L_W = np.matrix( inv_L_W )

    # Appendix B Step 5
    L_E = prep_transform_matrix_for_gaussian_coefficients( ens_size_original, ens_size_virtual )
    k = np.sqrt( ( M+N-1. )/(N-1.) )
    
    # Appendix B Step 6
    E_prime = L_E * inv_L_W * W

    # Appendix B Step 7
    resampling_matrix = E_prime + (k-1)/M

    return np.matrix( resampling_matrix )





# Sanity checking compute_localized_gaussian_resampling_coefficients
def SANITY_CHECK_compute_localized_gaussian_resampling_coefficients():

    # Generate lat lon grid
    latmesh, lonmesh = np.meshgrid( 
        np.linspace(-88, 88, 45 ),
        np.linspace(-178, 178, 90)
    )
    fcst_ens_size = np.int64(10)
    virt_ens_size = np.int64(100)

    # Choose an ROI in kilometers
    roi_in_km = 111 * 40               # Basically 40 degree ROI on equator

    # Does the code run?
    all_lon1d = lonmesh.flatten()
    all_lat1d = latmesh.flatten()
    targ_lon = -60.0
    targ_lat = 40.0
    E = compute_localized_gaussian_resampling_coefficients( 
        all_lon1d, all_lat1d, targ_lon, targ_lat, roi_in_km, fcst_ens_size, virt_ens_size 
    )

    return




















'''
    FUNCTION TO GENERATE LOCALIZED GAUSSIAN RESAMPLING COEFFICIENTS THAT GUARANTEES
    LOCAL VARIANCE CONSERVATION AND APPROXIMATELY LOCALIZED SPATIAL COVARIANCES

    The basic procedure is the same as that of matrix E described in Chan et al 2020 
    MWR paper on BGEnKF.

    The only modification is that the W matrix of Chan et al 2020 Appendix B varies
    spatially. 

    Note that the resulting virtual members are not iid.

    Inputs:
    -------
    1) all_lon1d (1D NumPy float64 array)
            Longitudes of ALL model locations in degrees East
            Basically a flattened version of longitude meshgrid
    2) all_lat1d (1D NumPy float64 array)
            Latitudes of ALL model locations in degrees North
            Basically a flattened version of latitude meshgrid
    3) targ_lon (scalar NumPy float64)
            Targetted location's longitude in degrees East
    4) targ_lat (scalar NumPy float64)
            Targetted location's latitude in degrees North
    5) roi_in_km (scalar NumPy float64)
            Radius of influence in kilometers
    6) ens_size_original (scalar integer)
            Original number of ensemble members
    7) ens_size_virtual (scalar integer)
            Number of virtual members to create
    
    Outputs:
    --------
    1) resampling_matrix ( ens_size_original x ens_size_virtual float matrix)
            matrix of linear combination coefficients (matrix E in MWR paper)
'''
def compute_POOR_MAN_localized_gaussian_resampling_coefficients( all_lon1d, all_lat1d, targ_lon, targ_lat, roi_in_km, ens_size_original, ens_size_virtual, E_special, rng_seed=0 ):

    # Rename variables to use a different notation.
    N = ens_size_original
    M = ens_size_virtual

    # Seed the NumPy random number generator -- useful for parallelization.
    np.random.seed(rng_seed)
    
    # Hard-coded upper bound on the number of random samples to materialize
    if ( N*M > 600 ):
        num_samples = 100
    else:
        num_samples = N*M

    # t1 = time()
    # Materialize spatially-correlated noise at the targetted location
    # This is the "localized" version of Chan et al 2020 Appendix B Step 1.
    W_poor = implicit_gen_spatially_correlated_sample( 
            all_lon1d, all_lat1d, targ_lon, targ_lat, roi_in_km, num_samples
    )
    # print('materialization happened in ', time()-t1, 'seconds')

    # Use Gaussian resampling to boost the number of samples
    W = np.array( np.matmul( W_poor , E_special ).reshape(N,M) )
    
    # Appendix B Step 2
    W = np.matrix( (W.T - np.mean(W, axis=1)).T )

    # t1 = time()
    # Appendix B Step 3 and 4
    C_W = W * W.T
    inv_L_W = np.linalg.inv(np.linalg.cholesky( C_W ))
    inv_L_W = np.matrix( inv_L_W )

    # Appendix B Step 5
    L_E = prep_transform_matrix_for_gaussian_coefficients( ens_size_original, ens_size_virtual )
    k = np.sqrt( ( M+N-1. )/(N-1.) )
    
    # Appendix B Step 6
    E_prime = L_E * inv_L_W * W

    # Appendix B Step 7
    resampling_matrix = E_prime + (k-1)/M
    # print('E-matrix computation happened in ', time()-t1, 'seconds')

    return np.matrix( resampling_matrix )






# Sanity checking compute_localized_gaussian_resampling_coefficients
def SANITY_CHECK_compute_POOR_MAN_localized_gaussian_resampling_coefficients():

    # Generate lat lon grid
    latmesh, lonmesh = np.meshgrid( 
        np.linspace(-88, 88, 45 ),
        np.linspace(-178, 178, 90)
    )
    fcst_ens_size = np.int64(80)
    virt_ens_size = np.int64(240)

    # Choose an ROI in kilometers
    roi_in_km = 111 * 40               # Basically 40 degree ROI on equator

    # Does the code run?
    all_lon1d = lonmesh.flatten()
    all_lat1d = latmesh.flatten()
    targ_lon = -60.0
    targ_lat = 40.0
    E = compute_POOR_MAN_localized_gaussian_resampling_coefficients( 
        all_lon1d, all_lat1d, targ_lon, targ_lat, roi_in_km, fcst_ens_size, virt_ens_size 
    )

    return
















'''
    Sanity checking
'''
if __name__ == '__main__':

    print('AOT compiles completed in ', time() - t0, ' seconds.')

    # SANITY_CHECK_gen_spatially_correlated_sample()
    t1 = time()
    SANITY_CHECK_compute_POOR_MAN_localized_gaussian_resampling_coefficients()
    print('Constructed all E matrices in ', time() - t1, ' seconds.')
