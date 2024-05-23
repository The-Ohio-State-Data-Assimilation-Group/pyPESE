'''
    FUNCTIONS RELATING TO FAST GAUSSIAN RESAMPLING
    -----------------------------------------------

    List of functions:
    ------------------

    1) COMPUTE_GAUSSIAN_RESAMPLING_COEFFICIENTS
       Generates a matrix of resampling coefficients that generates virtual probits with 
       the same covariance matrix as the forecast members' probits.

    2) 

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
def compute_unlocalized_gaussian_resampling_coefficients( N, M, rng_seed ):

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
    FUNCTION TO GENERATE LOCALIZED GAUSSIAN RESAMPLING COEFFICIENTS OVER
'''