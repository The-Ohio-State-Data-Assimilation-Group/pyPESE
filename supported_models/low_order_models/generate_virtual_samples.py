'''
    Library of functions to execute pyPESE
'''

# Import standard packages
import numpy as np
from sys import argv
from copy import deepcopy
from math import ceil as ceiling
from scipy.ndimage import convolve as scipy_convolve
import pickle
from os.path import isfile
from gc import collect as gc_collect
from shutil import copyfile as shutil_copyfile

# Import netCDF package
from netCDF4 import Dataset as ncopen

# Import useful parts of PyPESE package
from pyPESE.resampling.gaussian_resampling import fast_unlocalized_gaussian_resampling 
from pyPESE.distributions.distributions import all_dist_class_dict
from pyPESE.distributions.gaussian import STANDARD_NORMAL_INSTANCE as std_norm_dist



'''
    MAIN FUNCTION

    Function to apply pyPESE onto samples/ensemble of state vectors.

    Inputs:
    -------
    - orig_samples2d:
            2D NumPy array containing the samples/ensemble of state vectors.
            Dimensions: ( state, ens or sample )
    - num_virt_samples:
            Integer value indicating the number of virtual samples to construct.
            Must be at least 2 times the initial ensemble/sample of state vectors.
    - dist_name:
            String indicating the distribution family to use. See dictionary keys
            of all_dist_class_dict under pyPESE/distributions.py for the list of
            available distributions.

    Output:
    -------
    - virt_samples:
            2D NumPy array containing the virtual samples/ensemble of state vectors
            Dimensions: ( state, ens or sample )
'''
def generate_virtual_samples( orig_samples2d, num_virt_samples, dist_name ):

    Nx, Ne = orig_samples2d.shape

    # Sanity check: Does the number of virtual samples make sense?
    if (Nx > Ne) and (num_virt_samples < Nx *2):
        print( 'ERROR: Too few virtual members/samples requested. The minimum'
                + 'number of virtual members/samples is 2 times the original'
                + 'value.'
        )
    # --- End of sanity check.

    # Fit distributions and transform into probit-space
    orig_probits2d = np.zeros_like(orig_samples2d)
    fitted_dists_dict = {}
    for ix in range( Nx ):

        orig_samples1d = orig_samples2d[ix,:]


        # Step 1: fit MUWE distribution at current location
        #     Note that the user-specified distribution in muwe is the one 
        #     specified in  config_CAM_pyPESE.py
        params = all_dist_class_dict['muwe'].fit(
            orig_samples1d, all_dist_class_dict[dist_name]
        )
        fitted_dists_dict[ix] = all_dist_class_dict['muwe']( *params )


        # Step 2: Transform to probit space
        orig_probits2d[ix,:] = std_norm_dist.ppf(
            fitted_dists_dict[ix].cdf( orig_samples1d )
        )
        orig_probits2d[ix,:] -= np.mean( orig_probits2d[ix,:] )
        orig_probits2d[ix,:] /= np.std( orig_probits2d[ix,:], ddof=1)

    # --- End of loop over variables to map into probit space


    # Step 3: Perform Gaussian resampling in probit space
    # ---------------------------------------------------
    if Ne < Nx*10 :  # For scenarios where sample size is small
        virt_samples2d = fast_unlocalized_gaussian_resampling( 
            orig_probits2d, num_virt_samples
        )
    else:   # For scenarios where large samples are available
        virt_samples2d = traditional_unlocalized_gaussian_sampling(
            orig_probits2d, num_virt_samples
        )
    # --- End of Gaussian resampling
    

    # Step 4: Undo transforms applied in Step 2
    # -----------------------------------------
    for ix in range(Nx):
        virt_samples2d[ix,:] = fitted_dists_dict[ix].ppf(
            std_norm_dist.cdf( virt_samples2d[ix,:])
        )
    
    # Return virtual samples
    return virt_samples2d











'''
    Function to execute Gaussian sampling in the old way.

    Only efficient if Ne >> Nx.
'''
def traditional_unlocalized_gaussian_sampling( orig_samples2d, num_virt_samples ):

        Nx, Ne = orig_samples2d.shape

        # Compute square-root of sample covariance matrix
        orig_cov = np.cov( orig_samples2d, rowvar=True )
        sqrt_orig_cov = np.linalg.cholesky(orig_cov)

        # Generate white noise
        noise = np.random.normal( size=(Nx, num_virt_samples))
        noise = (noise.T - np.mean(noise, axis=1)).T
        noise_cov = np.cov( noise, rowvar=True)
        whitener_operator = np.linalg.inv( np.linalg.cholesky( noise_cov ) )
        white_noise = np.matmul( whitener_operator, noise )

        # Construct virtual samples
        virt_samples2d = np.matmul( sqrt_orig_cov, white_noise )

        # Rescaling to conserve expanded sample's covariance
        virt_samples2d *= np.sqrt( num_virt_samples/ (num_virt_samples-1) )

        # Handling mean state
        virt_samples2d[:,:] = (virt_samples2d.T + np.mean( orig_samples2d, axis=1 ) ).T

        return virt_samples2d








'''
    Sanity check for generate_virtual_samples
'''
def SANITY_CHECK_generate_virtual_samples(Nx = 10, Ne = 100):
    
    # Generate original samples via harmonics with some noise
    white_noise = np.random.normal( size = (Nx, Ne) )
    fft_amps = np.fft.fft( white_noise, axis=0 )
    abs_freq = np.abs( np.fft.fftfreq(Nx, d=1./Nx) )
    fft_amps[ abs_freq < 0.5 ] = 0.+0j
    orig_samples2d = np.real( np.fft.ifft( fft_amps, axis=0) ) 
    orig_samples2d[:,:] = (orig_samples2d.T/np.std( orig_samples2d, axis=1)).T
    orig_samples2d += np.random.normal( size = (Nx, Ne) )*0.1

    # Purrform resampling
    Nv = Ne*3
    virt_samples2d = generate_virtual_samples( orig_samples2d, Nv, 'gauss')

    # Mean check
    absdiff_mean = np.abs( np.mean(virt_samples2d, axis=1) - np.mean( orig_samples2d, axis=1) )
    print( 'Deviation of virtual mean from original mean' )
    print( np.sort( absdiff_mean)/np.std( orig_samples2d) )

    # Cov check
    expd_samples2d = np.zeros( (Nx, Nv+Ne))
    expd_samples2d[:,:Ne] = orig_samples2d
    expd_samples2d[:,Ne:] = virt_samples2d
    absdiff_cov = np.abs( 
        np.cov(expd_samples2d, rowvar=True) 
        - np.cov( orig_samples2d, rowvar=True) 
    )
    print( 'Deviation of expanded sample covariance from original covariance' )
    print( ( absdiff_cov.flatten()).max() )






if __name__ == '__main__':
    SANITY_CHECK_generate_virtual_samples()
    