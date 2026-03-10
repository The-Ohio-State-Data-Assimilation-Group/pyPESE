'''
    FUNCTION TO GENERATE FOR LOCAL WEIGHTS NEEDED FOR LOCALIZED PESE ON 3D LATITUDE-LONGITUDE
    -PRESSURE GRID. 

    * The grid must be regular in lat and lon. 
    

    Description:
    ------------
    Locally-correlated Gaussian noise samples are needed for localized PESE.
    This package contains generic functions needed to perform this localization.
    
    Both horizontal & vertical localization are supported.
    
    The Gaspari-Cohn 1999 5th order rational function is used as the 
    localization function. 

'''

# Package imports
import numpy as np
from numba import njit,  float64 as nbf64
from copy import deepcopy

# Loading functions relating to gaspari-cohn
from pyPESE.resampling.local_gaussian_resampling import GC99

# Handy constant
from math import pi as PI
DEG2PI = PI/180
















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
    FUNCTION TO GENERATE LOCALIZED NOISE ON A 3D LAT-LON-LOGPRES GRID

    Assumptions:
    1) Horizontal coordinates are latitudes and longitudes
    2) dx & dy are the same
    
    Note that vetical coordinate can vary spatially

    Inputs:
    1) lon1d
            1D array of longitude values in degrees
    2) lat1d
            1D array of latitude values in degrees
    3) hroi_in_km
            Horizontal radius-of-influence (ROI) for horizontal localization in kilometers
            hroi must be specified in either one of the following ways:
            a)  Scalar float value that specifies a constant HROI for all locations
            TODO: 3D NumPy float array specifying HROI at every grid box
                  Shape must be (nz, nlat, nlon)
    4) pres1d
            1D array of pressure values in Pa
    5) vroi_in_LogP
            Vertical radius-of-influence (ROI) for vertical localization in Log-P coordinates
            vroi must be specified in either one of the following ways:
            a)  Scalar float value that specifies a constant VROI for all locations.
            TODO: 3D NumPy float array specifying HROI at every grid box
                  Shape must be (nz, nlat, nlon)

    6) rng_seed_list
            Seed used by numpy random number generator

    NOTE: Specifying negative hroi deactivates horizontal localization.
    NOTE: Specifying negative vroi deactivates vertical localization.
            
'''

def gen_local_noise_for_3d_reg_LatLonLogP_grid( lon1d, lat1d, pres1d, hroi_in_km, vroi_in_LogP, rng_seed ):

    # Grid dimensions
    nlon = len(lon1d)
    nlat = len(lat1d)
    npres = len(pres1d)

    # Seed random num generator
    np.random.seed(rng_seed)


    # GENERATE WHITE NOISE
    # --------------------
    # Case where horizontal & vertical localization is active
    if np.sum(hroi_in_km > 0) > 0 and np.sum(vroi_in_LogP > 0) > 0:
        noise3d = np.random.normal( size = (npres, nlat, nlon))
    
    # Case where only horizontal localization is active
    elif np.sum(hroi_in_km > 0) > 0 and np.sum(vroi_in_LogP > 0) == 0:
        noise3d = np.random.normal( size = (1, nlat, nlon))

    # Case where only vertical localization is active
    elif np.sum(hroi_in_km > 0) == 0 and np.sum(vroi_in_LogP > 0) > 0:
        noise3d = np.random.normal( size = (npres, 1, 1))

    # Case where no localization is active at all
    # -- This is a trivial case -- can immediately generate output
    elif np.sum(hroi_in_km > 0) == 0 and np.sum(vroi_in_LogP > 0) == 0:
        return np.ones( (npres, nlat, nlon) ) * np.random.normal(size=1)




    # HORIZONTAL LOCALIZATION
    # -----------------------
    # Situation: HROI is a scalar and horizontal localization is active
    if isinstance( hroi_in_km, (int, float) ) and (np.sum(hroi_in_km > 0) > 0):

        # Compute horizontal kernel length
        kern_hlen_in_km = hroi_in_km / np.sqrt(2)

        # Convolve with GC99 kernel
        input_noise3d = deepcopy(noise3d)
        noise3d = horiz_local_noise_gen_NJIT_ACCEL( lon1d, lat1d, input_noise3d, kern_hlen_in_km)

    # --- End of handling situation where the HROI is scalar and horizontal localization is active


    # VERTICAL LOCALIZATION
    # ----------------------
    # Situation: VROI is a scalar and vertical localization is active
    if isinstance( vroi_in_LogP, (int, float) ) and (np.sum(vroi_in_LogP > 0) > 0):

        # Compute vertical kernel length
        kern_vlen_in_Pa = vroi_in_LogP / np.sqrt(2)
        input_noise3d = deepcopy(noise3d)

        # Convolve with GC99 kernel
        noise3d = vert_local_noise_gen_NJIT_ACCEL( np.log(pres1d), input_noise3d, kern_vlen_in_Pa)

    # --- End of handling situation where the VROI is scalar and vertical localization is active



    # RETURN NOISE SAMPLES 
    # --------------------
    # The return statement depends on whether horiziontal and/or vertical localization is done)

    # Situation where both vertical and horizontal localization are done
    if np.sum(hroi_in_km > 0) > 0 and np.sum(vroi_in_LogP > 0) > 0:
        return noise3d

    # Situation where only horizontal localization is active
    if np.sum(hroi_in_km > 0) > 0 and np.sum(vroi_in_LogP > 0) == 0:
        return np.zeros( [npres, nlat, nlon], dtype='f8' ) + noise3d[0]

    # Situation where only vertical localization is active 
    if np.sum(hroi_in_km > 0) == 0 and np.sum(vroi_in_LogP > 0) > 0:
        noise3d = np.zeros( [nlon, nlat, npres], dtype='f8' ) + noise3d[:,0,0]
        return np.swapaxes( noise3d, 0, 2)    

        

    # Final exception statement
    return 'ERROR: Final return statement triggered'










'''
    FUNCTION TO GENERATE LOCALIZED NOISE ON A 3D LAT-LON-PRES GRID

    Assumptions:
    1) Horizontal coordinates are latitudes and longitudes
    2) dx & dy are the same
    
    Note that vetical coordinate can vary spatially

    Inputs:
    1) lon1d
            1D array of longitude values in degrees
    2) lat1d
            1D array of latitude values in degrees
    3) hroi_in_km
            Horizontal radius-of-influence (ROI) for horizontal localization in kilometers
            hroi must be specified in either one of the following ways:
            a)  Scalar float value that specifies a constant HROI for all locations
            TODO: 3D NumPy float array specifying HROI at every grid box
                  Shape must be (nz, nlat, nlon)
    4) pres1d
            1D array of pressure values in Pa
    5) vroi_in_Pa
            Vertical radius-of-influence (ROI) for horizontal localization in Pascals
            vroi must be specified in either one of the following ways:
            a)  Scalar float value that specifies a constant VROI for all locations.
            TODO: 3D NumPy float array specifying HROI at every grid box
                  Shape must be (nz, nlat, nlon)

    6) rng_seed_list
            Seed used by numpy random number generator

    NOTE: Specifying negative hroi deactivates horizontal localization.
    NOTE: Specifying negative vroi deactivates vertical localization.
            
'''

def gen_local_noise_for_3d_reg_LatLonPres_grid( lon1d, lat1d, pres1d, hroi_in_km, vroi_in_Pa, rng_seed ):

    # Grid dimensions
    nlon = len(lon1d)
    nlat = len(lat1d)
    npres = len(pres1d)

    # Seed random num generator
    np.random.seed(rng_seed)


    # GENERATE WHITE NOISE
    # --------------------
    # Case where horizontal & vertical localization is active
    if np.sum(hroi_in_km > 0) > 0 and np.sum(vroi_in_Pa > 0) > 0:
        noise3d = np.random.normal( size = (npres, nlat, nlon))
    
    # Case where only horizontal localization is active
    elif np.sum(hroi_in_km > 0) > 0 and np.sum(vroi_in_Pa > 0) == 0:
        noise3d = np.random.normal( size = (1, nlat, nlon))

    # Case where only vertical localization is active
    elif np.sum(hroi_in_km > 0) == 0 and np.sum(vroi_in_Pa > 0) > 0:
        noise3d = np.random.normal( size = (npres, 1, 1))

    # Case where no localization is active at all
    # -- This is a trivial case -- can immediately generate output
    elif np.sum(hroi_in_km > 0) == 0 and np.sum(vroi_in_Pa > 0) == 0:
        return np.ones( (npres, nlat, nlon) ) * np.random.normal(size=1)




    # HORIZONTAL LOCALIZATION
    # -----------------------
    # Situation: HROI is a scalar and horizontal localization is active
    if isinstance( hroi_in_km, (int, float) ) and (np.sum(hroi_in_km > 0) > 0):

        # Compute horizontal kernel length
        kern_hlen_in_km = hroi_in_km / np.sqrt(2)

        # Convolve with GC99 kernel
        input_noise3d = deepcopy(noise3d)
        noise3d = horiz_local_noise_gen_NJIT_ACCEL( lon1d, lat1d, input_noise3d, kern_hlen_in_km)

    # --- End of handling situation where the HROI is scalar and horizontal localization is active


    # VERTICAL LOCALIZATION
    # ----------------------
    # Situation: VROI is a scalar and vertical localization is active
    if isinstance( vroi_in_Pa, (int, float) ) and (np.sum(vroi_in_Pa > 0) > 0):

        # Compute vertical kernel length
        kern_vlen_in_Pa = vroi_in_Pa / np.sqrt(2)
        input_noise3d = deepcopy(noise3d)

        # Convolve with GC99 kernel
        noise3d = vert_local_noise_gen_NJIT_ACCEL( pres1d, input_noise3d, kern_vlen_in_Pa)

    # --- End of handling situation where the VROI is scalar and vertical localization is active



    # RETURN NOISE SAMPLES 
    # --------------------
    # The return statement depends on whether horiziontal and/or vertical localization is done)

    # Situation where both vertical and horizontal localization are done
    if np.sum(hroi_in_km > 0) > 0 and np.sum(vroi_in_Pa > 0) > 0:
        return noise3d

    # Situation where only horizontal localization is active
    if np.sum(hroi_in_km > 0) > 0 and np.sum(vroi_in_Pa > 0) == 0:
        return np.zeros( [npres, nlat, nlon], dtype='f8' ) + noise3d[0]

    # Situation where only vertical localization is active 
    if np.sum(hroi_in_km > 0) == 0 and np.sum(vroi_in_Pa > 0) > 0:
        noise3d = np.zeros( [nlon, nlat, npres], dtype='f8' ) + noise3d[:,0,0]
        return np.swapaxes( noise3d, 0, 2)    

        

    # Final exception statement
    return 'ERROR: Final return statement triggered'





'''
    Accelerated convolution of horizontal localization function
'''
@njit( nbf64[:,:,:]( nbf64[:], nbf64[:], nbf64[:,:,:], nbf64 ) )
def horiz_local_noise_gen_NJIT_ACCEL( lon1d, lat1d, raw_noise3d, kern_hlen_in_km):

    # Grid dimensions
    nlon = len(lon1d)
    nlat = len(lat1d)

    # Generate mesh of latlon
    lon2d = np.zeros( (nlat, nlon), dtype='f8') + lon1d
    lat2d = (np.zeros( (nlon, nlat), dtype='f8') + lat1d).T        

    # Output noise
    output_noise3d = np.zeros_like( raw_noise3d )

    # Convert kern_hlen_in_km from km into deg lat
    kern_hlen_in_deglat = kern_hlen_in_km / 111.3


    # Loop over all locations
    for ilon in np.arange(nlon):
        for ilat in np.arange(nlat):

            # Subset grid
            targ_lat = lat1d[ilat]
            targ_lon = lon1d[ilon]
            lat_flag1d = (
                ( lat1d > targ_lat - kern_hlen_in_deglat*1.1 )
                * ( lat1d < targ_lat + kern_hlen_in_deglat*1.1 )
            )

            # Compute great circle distance from current location
            sub_lat1d = lat2d[lat_flag1d,:].flatten()
            sub_lon1d = lon2d[lat_flag1d,:].flatten()
            dist_in_km = compute_great_circle_dist(
                sub_lon1d*DEG2PI, targ_lon*DEG2PI, sub_lat1d*DEG2PI, targ_lat*DEG2PI 
            )/1000

            # Generate kernel weights 
            flag_in_roi = dist_in_km <= kern_hlen_in_km
            nearby_dists  = dist_in_km[flag_in_roi]
            gc_weights = GC99( nearby_dists, kern_hlen_in_km )

            # Convolve
            for ipres in np.arange( raw_noise3d.shape[0] ):
                nearby_noise = ((raw_noise3d[ipres,lat_flag1d,:]).flatten())[flag_in_roi]
                output_noise3d[ipres, ilat, ilon] = np.sum( nearby_noise * gc_weights )
            # --- End of loop over pressure
        # --- End of loop over latitudes
    # --- End of loop over longitudes
            
        
    return output_noise3d





'''
    Accelerated convolution of vertical localization function
'''
@njit( nbf64[:,:,:]( nbf64[:], nbf64[:,:,:], nbf64 ) )
def vert_local_noise_gen_NJIT_ACCEL( pres1d, raw_noise3d, kern_vlen_in_Pa):

    # Grid dimensions
    npres = len(pres1d)

    # Output noise
    output_noise3d = np.zeros_like( raw_noise3d )

    # Loop over p levels
    for ipres in np.arange(npres):

        # Compute GC weights
        delta_pres = np.abs( pres1d - pres1d[ipres] )
        gc_weights = GC99( delta_pres, kern_vlen_in_Pa )
        
        # Convolve!
        for jpres in np.arange(npres):
            output_noise3d[ipres,:,:] += (
                gc_weights[jpres] * raw_noise3d[jpres,:,:]
            )
        # -- end of convolution
    # --- end of loop over p levels
        
    return output_noise3d