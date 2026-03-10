'''
    EXPONENTIAL DISTRIBUTION
    =================================================================================
    Basically the scipy.stats.expon distribution except that the fitting process uses 
    the data minimum & data mean instead of MLE.
    
    Fitting process:
    ----------------
    ~ Offset = data minimum
    ~ Scale = data mean - offset
'''


import numpy as np
from scipy.stats import expon





'''
    SciPy-like class for Exponential distribution.
'''
class exponential:

    name = 'exponential distribution'

    # Initialize 
    def __init__( self, offset, scale ):
        self.offset = offset
        self.scale = scale
        self.dist_inst = expon( loc = offset, scale = scale)
        return

    # Fit exponential distirbution to 1d data
    def fit( data1d ):
        offset = ( np.array(data1d) ).min()
        scale = np.mean( data1d ) - offset
        return offset, scale

    # Function to evaluate CDF of fitted exponential distribution. 
    def cdf(self, eval_pts):
        out_cdf = np.zeros_like( eval_pts )
        minval = self.offset
        flag_eval = (minval <= eval_pts ) 
        out_cdf[flag_eval] = self.dist_inst.cdf(eval_pts[flag_eval])
        return out_cdf


    # Function to evaluate inverse CDF of fitted Gaussian
    def ppf(self, eval_cdf):
        return self.dist_inst.ppf(eval_cdf)
    

    # Function to evaluate PDF of fitted Gaussian
    def pdf( self, eval_pts ):
        return self.dist_inst.pdf(eval_pts)
        

    # Function to draw samples consistent with fitted Gaussian
    def rvs( self, shape ):
        return self.dist_inst.rvs( size=shape )
    
# ------ End of exponential distribution SciPy-like class







'''
    SANITY CHECK OF EXPONENTIAL DISTRIBUTION CLASS
'''
def SANITY_CHECK_exponential():

    import matplotlib.pyplot as plt

    # True parameters
    loc = 10
    scale = 3
    true_expon = expon( loc=loc, scale=scale)

    # Data
    data = true_expon.rvs( size = 1000)

    # Fit
    params = exponential.fit( data )
    fitted_dist = exponential( *params )

    # Evaluate CDF, PPF & CDF for true distribution
    plot_coords = np.linspace( 10, 30, 101 )
    true_cdf = true_expon.cdf( plot_coords )
    true_pdf = true_expon.pdf( plot_coords )
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
    plt.title( 'True expon( %3.1f, %3.1f ) vs fitted exponential( %3.1f, %3.1f ) ' % (loc, scale, *params))
    plt.savefig( 'SANITY_CHECK_exponential.pdf')


    



if __name__ == '__main__':

    SANITY_CHECK_exponential()