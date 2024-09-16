'''
    FUNCTIONS RELATING TO FAST GAUSSIAN RESAMPLING
    ===============================================

    List of functions:
    ------------------

    1) fast_unlocalized_gaussian_resampling
            Function to execute Gaussian resampling in the subspace spanned by the ensemble
            perturbations.

    2) compute_unlocalized_gaussian_resampling_coefficients
            Generates a matrix of resampling coefficients that generates virtual probits with 
            the same covariance matrix as the forecast members' probits.

    3) prep_transform_matrix_for_gaussian_coefficients
            Utility function to generate transformation matrix. That matrix is used by function
            `compute_unlocalized_gaussian_resampling_coefficients`


'''

import numpy as np
from numba import njit












'''
    FUNCTION TO RESAMPLE ENSEMBLE VIA FAST GAUSSIAN RESAMPLING

    The expanded ensemble will have the same covariance matrix and mean vector as the original 
    ensemble members.
    This means that the resampling does not account for localization.

    
    Mandatory Inputs:
    -----------------
    1) ens2d ( variables x ens members )
            2D NumPy array containing the ensemble members

    2) num_virtual (scalar integer)
            Number of virtual members to create

            
    Optional Input:
    ---------------
    1) rng_seed (scalar integer)
            Seed for NumPy's in-built random number generator.
            This seed is useful for parallelized asynchronous fast
            Gaussian resampling

'''
def fast_unlocalized_gaussian_resampling( original_ens2d, ens_size_virtual, rng_seed = 0 ):

    # Determine number of variables and original ensemble size
    num_variables, ens_size_original = original_ens2d.shape

    # Ensuring that conditions to use resampling coefficients are satisfied
    if ( ens_size_virtual <= ens_size_original ):
        print( 'ERROR: fast_gaussian_resampling')
        print( '    Number of virtual members must be more than number of original members')
        quit()

    # Generate resampling matrix
    coeff_matrix = compute_unlocalized_gaussian_resampling_coefficients( 
                        ens_size_original, ens_size_virtual, rng_seed=rng_seed
                    )
    
    # Determine ensemble mean
    ens_mean = np.mean( original_ens2d, axis=1 )

    # Generate virtual members
    original_perts2d = np.matrix( original_ens2d.T - ens_mean ).T
    virtual_ens2d = np.array( original_perts2d * coeff_matrix )
    virtual_ens2d[:,:] = (virtual_ens2d.T + ens_mean).T

    return virtual_ens2d



        



















'''
    FUNCTION TO PREP TRANSFORMATION MATRIX NEEDED TO GENERATE RESAMPLING COEFFICIENTS
    
    The transformation matrix here refers to matrix L_E in Chan et al 2020 Monthly
    Weather Review paper on Bi-Gaussian EnKFs

    This transformation matrix is a deterministic function of original ensemble size and virtual 
    ensemble size.

    Note: assumes that the number of virtual members is more than 2x the original ensemble size.

    Mandatory Inputs:
    -----------------
    1) ens_size_original (scalar integer)
            Original number of ensemble members

    2) ens_size_virtual (scalar integer)
            Number of virtual members to create
    
    Outputs:
    --------
    1) transform_matrix ( ens_size_original x ens_size_virtual float matrix)
            Transformation matrix used to construct Gaussian resampling coefficients
        
'''
@njit
def prep_transform_matrix_for_gaussian_coefficients( ens_size_original, ens_size_virtual ):

    # Rename variables to use a different notation.
    N = ens_size_original
    M = ens_size_virtual

    # Appendix B Step 5
    k = np.sqrt( 
                ( M+N-1. )/(N-1.)
                )
    C_E = np.eye(N)* M/(N-1.)
    C_E -= (k-1)**2/M
    transform_matrix = np.linalg.cholesky( C_E )
    # transform_matrix = np.matrix(np.linalg.cholesky( C_E ))

    return transform_matrix

















'''
    FUNCTION TO GENERATE UNLOCALIZED GAUSSIAN RESAMPLING COEFFICIENTS THAT GUARANTEES
    COVARIANCE CONSERVATION

    This function assumes that there is no need to localize the resampling procedure.
    Results in the matrix E described in Chan et al 2020 MWR paper on BGEnKF.

    Note that the resulting virtual members are not iid.

    Mandatory Inputs:
    -----------------
    1) ens_size_original (scalar integer)
            Original number of ensemble members

    2) ens_size_virtual (scalar integer)
            Number of virtual members to create
    
    Outputs:
    --------
    1) resampling_matrix ( ens_size_original x ens_size_virtual float matrix)
            matrix of linear combination coefficients (matrix E in MWR paper)
'''
def compute_unlocalized_gaussian_resampling_coefficients( ens_size_original, ens_size_virtual, rng_seed=0 ):

    # Rename variables to use a different notation.
    N = ens_size_original
    M = ens_size_virtual

    # Seed the NumPy random number generator -- useful for parallelization.
    np.random.seed(rng_seed)

    # Generating matrix of noise (Appendix B Step 1)
    W = np.random.normal( size=[N,M] )

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





















# '''
#     FUNCTION TO GENERATE GAUSSIAN RESAMPLING COEFFICIENTS 

#     Based on M.-Y. Chan et al 2020 Monthly Weather Review paper on Bi-Gaussian 
#     EnKFs

#     Inputs:
#     1) N        -- Original ensemble size
#     2) M        -- Number of additional members
#     3) rng_seed -- Random number generator seed

#     Note that M must be greater than N!
# '''
# def IID_compute_unlocalized_gaussian_resampling_coefficients( N, M, rng_seed=0 ):

#     # Seed the NumPy random number generator -- useful for parallelization.
#     np.random.seed(rng_seed)

#     # Generating matrix of noise (Appendix B)
#     W = np.matrix( np.random.normal( size=[N,M] ) / np.sqrt(M) )

#     # Appendix B Step 5
#     k = np.sqrt( 
#                 ( M+N-1. )/(N-1.)
#                 )
#     C_E = np.eye(N)* M/(N-1.)
#     C_E -= (k-1)**2/M
#     L_E = np.matrix(np.linalg.cholesky( C_E ))
    
#     # Appendix B Step 6
#     E_prime = L_E * W

#     # Appendix B Step 7
#     E = E_prime + (k-1)/M

#     return np.matrix( E )












'''
    Sanity check
'''
if __name__ == '__main__':
    
    from scipy.ndimage import gaussian_filter1d  

    ens_size_original = 10
    ens_size_virtual  = 20

    original_ens = gaussian_filter1d( np.random.normal( size = [100, ens_size_original] ), sigma = 10., axis= 0 ) * 6
    original_mean = np.mean( original_ens, axis=1 )
    original_perts = (original_ens.T - original_mean).T

    print( np.cov( original_ens[::25] ))

    resampling_matrix = compute_unlocalized_gaussian_resampling_coefficients( 10, ens_size_virtual )

    expanded_ens = np.zeros( (100,ens_size_original+ens_size_virtual) ) 
    expanded_ens[:,:10] = original_ens
    expanded_ens[:,10:] = fast_gaussian_resampling( original_ens, ens_size_virtual, rng_seed = 0 )

    print('')
    print( np.cov( expanded_ens[::25] ))
