'''
    GAUSSIAN DISTRIBUTION
    =================================================================================
    Basically the scipy.stats.norm distribution. Except that the fitting process
    uses ensemble mean and variance instead of the MLE approach.
'''


import numpy as np
from scipy.stats import norm



'''
    USEFUL GLOBAL VARIABLES
'''
STANDARD_NORMAL_COORDINATES = np.linspace(-5,5,1001)
STANDARD_NORMAL_CDF = norm.cdf( STANDARD_NORMAL_COORDINATES )
STANDARD_NORMAL_PDF = norm.pdf( STANDARD_NORMAL_COORDINATES )






'''
    SciPy-like class for Gaussian distribution.
'''
class gaussian:

    name = 'gaussian distribution'

    # Initialize 
    def __init__( self, mean, variance ):
        self.mean = mean
        self.sigma = np.sqrt(variance)
        return

    # Fit Gaussian distirbution to 1d data
    def fit( data1d ):
        return np.mean( data1d ), np.var( data1d, ddof=1 )

    # Function to evaluate CDF of fitted Gaussian. 
    def cdf(self, eval_pts):

        # Transform the evaluation points to standard normal coordinates
        transformed_coord = (eval_pts - self.mean)/self.sigma

        # Employ linear interpolation to evaluate the CDF
        cdf = np.interp( 
            transformed_coord, STANDARD_NORMAL_COORDINATES, STANDARD_NORMAL_CDF
        )
        return cdf


    # Function to evaluate inverse CDF of fitted Gaussian
    def ppf(self, eval_cdf):

        # Treat this problem in terms of standard normal first
        std_norm_ppf_vals = np.interp(
            eval_cdf, STANDARD_NORMAL_CDF, STANDARD_NORMAL_COORDINATES
        )
        
        # Transform from standard normal to actual distribution
        ppf_vals = std_norm_ppf_vals * self.sigma + self.mean

        return ppf_vals
    

    # Function to evaluate PDF of fitted Gaussian
    def pdf( self, eval_pts ):
        return norm( loc=self.mean, scale=self.sigma).pdf( eval_pts )
        

    # Function to draw samples consistent with fitted Gaussian
    def rvs( self, shape ):
        return norm( loc=self.mean, scale=self.sigma).rvs( size=shape )
    
# ------ End of Gaussian distribution SciPy-like class




'''
    Standard normal instance of the gaussian class
'''
STANDARD_NORMAL_INSTANCE = gaussian( 0, 1 )