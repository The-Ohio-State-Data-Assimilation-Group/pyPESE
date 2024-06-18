'''
    FUNCTIONS TO EVOKE PROBIT-SPACE ENSEMBLE SIZE EXPANSION FOR GAUSSIAN COPULAS
    ============================================================================

    
    List of functions:
    ------------------
    1) pese_gc
            Function to call PESE-GC.
'''

# Load standard Python packages
import numpy as np
from scipy.stats import norm

# Load pyPESE's custom packages
from pyPESE.resampling.gaussian_resampling import compute_unlocalized_gaussian_resampling_coefficients




'''
    MAIN FUNCTION TO EVOKE PESE-GC WITHOUT LOCALIZATION

    
    Mandatory Inputs:
    ------------------
    1) fcst_ens_2d (dimensions: num_variables x num_fcst_ens)
            Two-dimensional NumPy float array containing forecast ensemble data
            The leftmost dimension corresponds to model variables
            The rightmost dimension corresponds to the ensemble members.

    2) list_of_dist_classes 
            A Python List object containing the distribution classes for every ensemble variable.
            There must be num_variables entries in the list.
            Two kinds of distribution classes are currently supported:
                a) scipy.stats distribution classes (e.g., skewnorm, gamma, norm)
                b) pyPESE-defined distributions 

    3) num_virt_ens (scalar integer)
            Number of virtual members to create.
            Must be greater than num_fcst_ens. This condition is required to ensure that the 
            resampling coefficient generation algorithm works.

    4) rng_seed (scalar integer)
            Seed for NumPy's in-built random number generator.
            This seed is useful for parallelized asynchronous fast
            Gaussian resampling

'''
def pese_gc( fcst_ens_2d, list_of_dist_classes, num_virt_ens, rng_seed=0 ): 

    # Determine all dimension sizes
    num_variables, num_fcst_ens = fcst_ens_2d.shape


    # Check if number of virtual members is appropriate.
    if ( num_virt_ens <= num_fcst_ens ):
        print( 'ERROR: pese_gc')
        print( '    Number of virtual members must be more than number of original members')
        quit()


    # Generate Gaussian resampling coefficient matrix
    gauss_resamp_matrix = compute_unlocalized_gaussian_resampling_coefficients( 
        num_fcst_ens, num_virt_ens, rng_seed=rng_seed 
    )


    # Init array to hold the virtual ensemble
    virt_ens_2d = np.zeros( [num_variables, num_virt_ens] )

    # Init array to hold fcst and virtual probits
    fcst_probit = np.zeros( [1, num_fcst_ens] )


    # For memory efficiency, only applying PESE-GC on one variable at a time.
    for ivar in range( num_variables ):

        # Pre-PESE-GC check: any duplicate values?
        uniq_vals = np.unique( fcst_ens_2d[ivar,:] )
        if ( len(uniq_vals) < num_fcst_ens ):
            sigma = np.std( fcst_ens_2d[ivar], ddof=1) / 1e3
            if sigma == 0:
                sigma = np.sqrt(np.mean(np.power(fcst_ens_2d[ivar,:],2))) / 1e5
            fcst_ens_2d[ivar,:] += np.random.normal( scale=sigma, size=num_fcst_ens )
        # --- End of special treatment.


        # Step 1: Fit distribution to fcst ensembel for selected variable
        params = list_of_dist_classes[ivar].fit( fcst_ens_2d[ivar,:])
        fitted_dist = list_of_dist_classes[ivar]( *params )

        # Step 2: Transform forecast ensemble into probit space
        fcst_probit[0,:] = norm.ppf( 
            fitted_dist.cdf( fcst_ens_2d[ivar,:] )
        )
        fcst_probit -= np.mean(fcst_probit)
        fcst_probit /= np.std( fcst_probit, ddof=1)

        # Step 3: Apply fast Gaussian resampling
        virt_probit = ( np.matmul( fcst_probit, gauss_resamp_matrix ) )

        # Step 4: Invert PPI transforms on virtual probits
        virt_ens_2d[ivar,:] = (
            fitted_dist.ppf(
                norm.cdf( virt_probit[0,:] )
            )
        )

    # --- End of loop over variables

    
    return virt_ens_2d




