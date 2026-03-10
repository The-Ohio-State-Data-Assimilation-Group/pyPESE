'''
    FUNCTION TO GENERATE FOR LOCAL WEIGHTS NEEDED FOR LOCALIZED PESE ON WRF-LIKE GRID

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
from copy import deepcopy
from scipy.ndimage import convolve as scipy_convolve

# Loading functions relating to gaspari-cohn
from pyPESE.resampling.local_gaussian_resampling import GC99, vertical_convolve_2d_noise_GC99


'''
    FUNCTION TO GENERATE LOCALIZED NOISE ON A 3D CARTESIAN GRID

    Assumptions:
    1) Horizontal coordinates are Cartesian
    2) dx & dy are the same
    
    Note that vetical coordinate can vary spatially

    Inputs:
    1) nx
            Scalar integer specifying number of grid points in the x-direction
    2) ny
            Scalar integer specifying number of grid points in the z-direction
    3) dx
            Scalar float specifying horizontal grid spacing
    4) hroi
            Horizontal radius-of-influence (ROI) for horizontal localization. 
            Must be in the same unit as inputted dx value. hroi must be specified
            in either one of the following ways:
            a)  Scalar float value that specifies a constant HROI for all horizontal
                locations.
            TODO: 3D NumPy float array specifying HROI at every grid box
                  Shape must be (nz, ny, nx)
    5) z3d
            NumPy array specifying vertical coordinate of WRF-like grid. Must be a 3D NumPy 
            float array of shape (nz, ny, nx)

            where nz is the number of model layers
    6) vroi
            Vertical ROI for vertical localization. Must be in the same unit as the inputted
            z_arr. vroi must be specified in one of the following ways:
            a)  Scalar float value that specifies a constant VROI for all grid boxes.
            TODO: 3D NumPy float array specifying a unique VROI for each model layer
                  Shape must be (nz, ny, nx)  

    7) rng_seed
            Seed used by numpy random number generator

    NOTE: Specifying negative hroi deactivates horizontal localization.
    NOTE: Specifying negative vroi deactivates vertical localization.
            
'''
def gen_local_noise_for_3d_cartesian_grid( nx, ny, dx, hroi, z3d, vroi, rng_seed ):

    # Detecting z dimension
    nz = z3d.shape[0]

    # RNG seeding
    np.random.seed(rng_seed)


    # Generate white noise
    # --------------------
    # Case where both horizontal and vertical localization are active
    if np.sum(hroi > 0) > 0 and np.sum(vroi > 0) > 0:
        noise3d = np.random.normal( size = (nz, ny, nx))

    # Case where only horizontal localization is active
    if np.sum(hroi > 0) > 0 and np.sum(vroi > 0) == 0:
        noise2d = np.random.normal( size = (ny, nx))
    
    # Case where only vertical localization is active
    if np.sum(hroi > 0) == 0 and np.sum(vroi > 0) > 0:
        noise1d = np.random.normal( size = nz )
        noise3d = np.zeros( size = (nz,ny,nx) )
        for iz in range(nz):
            noise3d[iz,:,:] = noise1d[iz]

        
    # Case where no localization is active at all
    # -- This is a trivial case -- can immediately generate output
    if np.sum(hroi > 0) == 0 and np.sum(vroi > 0) == 0:
        return np.ones( (nz,ny,nx) ) * np.random.normal(size=1)



    # HORIZONTAL LOCALIZATION
    # -----------------------

    # Situation: HROI is a scalar and horizontal localization is active
    if isinstance( hroi, (int, float) ) and (np.sum(hroi > 0) > 0):

        # Setting up kernel for locally smoothed noise
        kern_hlen_unitless = (hroi/dx) / np.sqrt(2.)

        # Horizontal kernel half-width in terms of grid boxes
        kernel_halfwidth_in_dx = int( kern_hlen_unitless ) + 3 # 3-pt padding for insurance.

        # Generate gc99 weights
        xy_offsets = np.arange( kernel_halfwidth_in_dx*-1, kernel_halfwidth_in_dx+1 )
        xmesh, ymesh = np.meshgrid( xy_offsets, xy_offsets)
        radius_mesh = np.sqrt( xmesh**2 + ymesh**2 )
        all_radius = (radius_mesh.flatten()).astype('f8')
        gc_weights_xy = ( GC99( all_radius, kern_hlen_unitless ) ).reshape( xmesh.shape )


        # Applying horizontal convolution onto white noise
        # ------------------------------------------------
        # Handling situations with and without vertical localization separately

        # Situation where vertical localization is active
        if np.sum(vroi>0) > 0: 
            # Convolution is applied one layer at a time
            for iz in range(nz):
                noise3d[iz,:,:] = scipy_convolve(
                    noise3d[iz,:,:], gc_weights_xy, mode='constant', cval=0.0, origin=0 
                )
            # --- end of loop over model layers
        
        # Situation where vertical localization is inactive
        if np.sum(vroi>0) == 0:
            noise2d[:,:] = scipy_convolve(
                noise2d[:,:], gc_weights_xy, mode='constant', cval=0.0, origin=0 
            )
    
    # --- End of horizontal localization section



    # VERTICAL LOCALIZATION
    # ----------------------

    # Situation: VROI is a scalar and vertical localization is active
    if isinstance( vroi, (int, float) ) and (np.sum(vroi > 0) > 0):

        # Kernel for vertical localziation
        kern_vlen = vroi / np.sqrt(2.)

        noise3d[:,:,:] = vertical_convolve_2d_noise_GC99( 
            noise3d.reshape([nz, ny*nx]), z3d.reshape([nz,ny*nx]),
            np.ones( [nz, ny*nx] , dtype='f8' ) * kern_vlen
        ).reshape([nz,ny,nx])

    # --- End of vertical localization section




    # Returning localized noise arrays
    # --------------------------------
    # Case where both horizontal and vertical localization are active
    if np.sum(vroi > 0) > 0:
        return noise3d

    # Case where only horizontal localization is active
    elif np.sum(hroi > 0) > 0 and np.sum(vroi > 0) == 0:
        noise3d = np.zeros( (nz,ny,nx) )
        for iz in range(nz):
            noise3d[iz,:,:] = noise2d
        return noise3d


# --- End of function to generate local noise on wrflike grid