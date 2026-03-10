'''
    GAMMA DISTRIBUTION
    =================================================================================
    Basically the scipy.stats.gamma distribution except that the fitting process uses 
    the data minimum, data mean and data variance, and the left bound is set to zero.
    
    Fitting process:
    ----------------
    ~ Offset = data minimum
    ~ shifted_data := data - offset
    ~ Scale = var( shifted_data ) / mean( shifted_data )
    ~ Shape = mean( shifted_data ) / scale
'''


import numpy as np
from scipy.stats import gamma as scipy_gamma





'''
    SciPy-like class for Gamma distribution.
'''
class gamma_leftbound_zero:

    name = 'gamma distribution'

    # Initialize 
    def __init__( self, shape, offset, scale ):
        self.shape = shape
        self.scale = scale
        self.offset = offset
        self.dist_inst = scipy_gamma( shape, loc = offset, scale = scale)
        return

    # Fit gamma distirbution to 1d data
    def fit( data1d ):
        offset = 0.
        shifted_data = data1d - offset
        scale = np.var(shifted_data, ddof=1) / np.mean( shifted_data )
        shape = np.mean( shifted_data ) / scale
        return shape, offset, scale

    # Function to evaluate CDF of fitted gamma distribution. 
    def cdf(self, eval_pts):
        out_cdf = np.zeros_like( eval_pts )
        minval = self.offset
        flag_eval = (minval <= eval_pts ) 
        out_cdf[flag_eval] = self.dist_inst.cdf(eval_pts[flag_eval])        
        return out_cdf



    # Function to evaluate inverse CDF of fitted Gamma
    def ppf(self, eval_cdf):
        return self.dist_inst.ppf(eval_cdf)
    

    # Function to evaluate PDF of fitted Gamma
    def pdf( self, eval_pts ):
        return self.dist_inst.pdf(eval_pts)
        

    # Function to draw samples consistent with fitted Gamma
    def rvs( self, shape ):
        return self.dist_inst.rvs( size=shape )
    
# ------ End of gamma distribution SciPy-like class







'''
    SANITY CHECK OF GAMMA DISTRIBUTION CLASS
'''
def SANITY_CHECK_gamma():

    import matplotlib.pyplot as plt

    # True parameters
    shape_param = 3
    loc = 0
    scale = 6
    true_gamma = scipy_gamma( shape_param, loc=loc, scale=scale)


    # Data
    data = true_gamma.rvs( size = 10000)

    # Fit
    params = gamma_leftbound_zero.fit( data )
    
    fitted_dist = gamma_leftbound_zero( *params )

    # Evaluate CDF, PPF & CDF for true distribution
    plot_coords = np.linspace( loc, loc + 10*scale, 101 )
    true_cdf = true_gamma.cdf( plot_coords )
    true_pdf = true_gamma.pdf( plot_coords )
    true_pdf /= true_pdf.max()

    # Evaluate CDF, PPF & CDF for fitted distribution
    fit_cdf = fitted_dist.cdf( plot_coords )
    fit_ppf = fitted_dist.ppf( true_cdf )
    fit_pdf = fitted_dist.pdf( plot_coords ) 
    fit_pdf /= fit_pdf.max()

    # Plot out stuff!
    plt.plot( plot_coords, true_cdf, '-k', label = 'True CDF', linewidth=6)
    plt.plot( plot_coords, true_pdf, '--k', label = 'True Scaled PDF')
    plt.plot( plot_coords, fit_cdf, '-r', label = 'Fitted CDF', linewidth=4)
    plt.plot( plot_coords, fit_pdf, '--r', label = 'Fitted Scaled PDF')
    plt.plot( fit_ppf, true_cdf, color='dodgerblue', label = 'Fitted PPF', linewidth=2)
    plt.legend()
    plt.title( 'True gamma( %3.1f, %3.1f, %3.1f ) vs fitted gamma( %3.1f, %3.1f, %3.1f ) ' % (shape_param, loc, scale, *params))
    plt.savefig( 'SANITY_CHECK_gamma_leftbound_zero.pdf')


    



if __name__ == '__main__':

    SANITY_CHECK_gamma()