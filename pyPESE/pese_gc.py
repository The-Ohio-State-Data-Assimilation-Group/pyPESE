'''
    FUNCTIONS TO EVOKE PROBIT-SPACE ENSEMBLE SIZE EXPANSION FOR GAUSSIAN COPULAS
    ============================================================================

    
    List of functions:
    ------------------
    1) pese_gc
'''

# Load standard Python packages
import numpy as np

# Load pyPESE's custom packages
import distributions as dists
import resampling as resam





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


    # Step 1: Fit distributions
    # -------------------------
    list_of_fitted_dists = []
    for ivar in range( num_variables ):
        # Determine distribution parameters
        params = list_of_dist_classes[ivar].fit( fcst_ens_2d[ivar,:])
        # Specify parameters into distribution classes
        list_of_fitted_dists.append(
            list_of_dist_classes[ivar]( *params )
        )
    # --- End of loop over variables
    

    # Step 2: Transform forecast ensemble into probit space
    # ------------------------------------------------------
    fcst_probit_2d = np.zeros( (num_variables, num_fcst_ens) )

    # Map from native space to quantile space
    for ivar in range( num_variables ):
        # Map to quantile space
        fcst_probit_2d[ivar,:] = list_of_fitted_dists[ivar].cdf( fcst_ens_2d[ivar,:] )
    # --- End of loop over variables
    
    # Map from quantile space to probit space
    fcst_probit_2d[:,:] = norm.ppf( fcst_probit_2d )

    # Enforcing zero mean and unity variance conditions
    fcst_probit_2d[:,:] = ( 
        ( fcst_probit_2d.T - np.mean( fcst_probit_2d, axis=1) ) 
        / np.std( fcst_probit_2d, ddof=1, axis=1 )
    ).T


    # Step 3: Evoke fast Gaussian resampling
    # ---------------------------------------
    virt_probit_2d = resam.gaussian_resampling.fast_unlocalized_gaussian_resampling( 
        fcst_probit_2d, num_virt_ens, rng_seed = rng_seed 
    )

    
    # Step 4: Map virtual probits into native space
    # ---------------------------------------------

    # First, map to quantile space
    virt_ens_2d = norm.cdf( virt_probit_2d )

    # Then, use fitted distributions to map from quantile space to native space
    for ivar in range( num_variables ):
        virt_ens_2d[ivar,:] = list_of_fitted_dists[ivar].ppf( virt_ens_2d[ivar,:] )

    
    return virt_ens_2d




