'''
    FUNCTIONS RELATING TO FAST GAUSSIAN RESAMPLING
    ===============================================

    List of functions:
    ------------------

    1) compute_gaussian_resampling_coefficients
            Generates a matrix of resampling coefficients that generates virtual probits with 
            the same covariance matrix as the forecast members' probits.


'''

import numpy as np
































'''
    FUNCTION TO GENERATE GAUSSIAN RESAMPLING COEFFICIENTS 

    Based on M.-Y. Chan et al 2020 Monthly Weather Review paper on Bi-Gaussian 
    EnKFs

    Inputs:
    1) N        -- Original ensemble size
    2) M        -- Number of additional members
    3) rng_seed -- Random number generator seed

    Note that M must be greater than N!
'''
def compute_unlocalized_gaussian_resampling_coefficients( N, M, rng_seed=0 ):

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
    k = np.sqrt( 
                ( M+N-1. )/(N-1.)
                )
    C_E = np.eye(N)* M/(N-1.)
    C_E -= (k-1)**2/M
    L_E = np.matrix(np.linalg.cholesky( C_E ))
    
    # Appendix B Step 6
    E_prime = L_E * inv_L_W * W

    # Appendix B Step 7
    E = E_prime + (k-1)/M

    return np.matrix( E )











'''
    FUNCTION TO GENERATE GAUSSIAN RESAMPLING COEFFICIENTS 

    Based on M.-Y. Chan et al 2020 Monthly Weather Review paper on Bi-Gaussian 
    EnKFs

    Inputs:
    1) N        -- Original ensemble size
    2) M        -- Number of additional members
    3) rng_seed -- Random number generator seed

    Note that M must be greater than N!
'''
def IID_compute_unlocalized_gaussian_resampling_coefficients( N, M, rng_seed=0 ):

    # Seed the NumPy random number generator -- useful for parallelization.
    np.random.seed(rng_seed)

    # Generating matrix of noise (Appendix B)
    W = np.matrix( np.random.normal( size=[N,M] ) / np.sqrt(M) )

    # Appendix B Step 5
    k = np.sqrt( 
                ( M+N-1. )/(N-1.)
                )
    C_E = np.eye(N)* M/(N-1.)
    C_E -= (k-1)**2/M
    L_E = np.matrix(np.linalg.cholesky( C_E ))
    
    # Appendix B Step 6
    E_prime = L_E * W

    # Appendix B Step 7
    E = E_prime + (k-1)/M

    return np.matrix( E )












'''
    Sanity check
'''
if __name__ == '__main__':
    
    from scipy.ndimage import gaussian_filter1d  

    original_ens = gaussian_filter1d( np.random.normal( size = [100, 10] ), sigma = 10., axis= 0 ) * 6
    original_mean = np.mean( original_ens, axis=1 )
    original_perts = (original_ens.T - original_mean).T

    print( np.cov( original_ens[10:12] ))

    resampling_matrix = compute_unlocalized_gaussian_resampling_coefficients( 10, 20000 )

    expanded_ens = np.zeros( (100,10+20000) ) 
    expanded_ens[:,:10] = original_ens
    expanded_ens[:,10:] = ( ( np.matrix( original_perts ) * np.matrix( resampling_matrix ) ).T + original_mean ).T

    print('')
    print( np.cov( expanded_ens[10:12] ))

    resampling_matrix = IID_compute_unlocalized_gaussian_resampling_coefficients( 10, 20000 )
    expanded_ens[:,:10] = original_ens
    expanded_ens[:,10:] = ( ( np.matrix( original_perts ) * np.matrix( resampling_matrix ) ).T + original_mean ).T
    
    print('')
    print( np.cov( expanded_ens[10:12] ))


    
