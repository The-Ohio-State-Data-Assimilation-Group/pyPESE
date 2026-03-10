'''
    BETA DISTRIBUTION
    =================================================================================
    Basically the scipy.stats.beta distribution except that the fitting process uses 
    the data minimum, data maximum, data mean and data variance instead of MLE.
    
    Fitting process:
    ----------------
    ~ Offset = data minimum
    ~ transformed_data := (data - offset) / (max - min)
    ~ mu = mean( transformed_data )
    ~ vr = var( transformed_data )
    ~ param_alpha = mu * (  mu*(1-mu)/vr - 1 )
    ~ param_beta = (1-mu)(  mu*(1-mu)/vr - 1 )
'''


import numpy as np
from scipy.stats import beta as scipy_beta





'''
    SciPy-like class for beta distribution.
'''
class beta:

    name = 'beta distribution'

    # Initialize 
    def __init__( self, offset, data_range, param_alpha, param_beta ):
        self.offset         = offset
        self.scale          = data_range
        self.data_range     = data_range
        self.param_alpha    = param_alpha
        self.param_beta     = param_beta
        self.dist_inst = scipy_beta( param_alpha, param_beta, loc = offset, scale = data_range )
        return

    # Fit beta distirbution to 1d data
    def fit( data1d ):
        data_copy = np.array( data1d )
        num_data = len(data_copy)
        data_range = data_copy.max() - data_copy.min()
        epsilon = data_range / (num_data)
        offset = data_copy.min() - epsilon
        data_range += 2*epsilon
        transformed_data = (data_copy - offset)/data_range
        mu = np.mean( transformed_data )
        vr = np.var( transformed_data, ddof=1 )
        param_alpha = mu * (  mu*(1-mu)/vr - 1 )
        param_beta = (1-mu)*(  mu*(1-mu)/vr - 1 )
        return offset, data_range, param_alpha, param_beta

    # Function to evaluate CDF of fitted beta distribution. 
    def cdf(self, eval_pts):
        out_cdf = np.zeros_like( eval_pts )
        minval = self.offset
        maxval = self.offset + self.data_range
        flag_eval = (minval <= eval_pts ) * ( eval_pts <= maxval )
        out_cdf[flag_eval] = self.dist_inst.cdf(eval_pts[flag_eval])
        return out_cdf


    # Function to evaluate inverse CDF of fitted beta
    def ppf(self, eval_cdf):
        return self.dist_inst.ppf(eval_cdf)
    

    # Function to evaluate PDF of fitted beta
    def pdf( self, eval_pts ):
        return self.dist_inst.pdf(eval_pts)
        

    # Function to draw samples consistent with fitted beta
    def rvs( self, shape ):
        return self.dist_inst.rvs( size=shape )
    
# ------ End of beta distribution SciPy-like class







'''
    SANITY CHECK OF beta DISTRIBUTION CLASS
'''
def SANITY_CHECK_beta():

    import matplotlib.pyplot as plt

    # True parameters
    loc = 4
    scale = 6
    true_beta = scipy_beta( 2,5, loc=loc, scale=scale)


    # Data
    data = true_beta.rvs( size = 10000)

    # Fit
    params = beta.fit( data )
    
    fitted_dist = beta( *params )

    # Evaluate CDF, PPF & CDF for true distribution
    plot_coords = np.linspace( loc, loc + scale, 101 )
    true_cdf = true_beta.cdf( plot_coords )
    true_pdf = true_beta.pdf( plot_coords )
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
    plt.title( 'True beta vs fitted beta')
    plt.savefig( 'SANITY_CHECK_beta.pdf')


    



if __name__ == '__main__':

    SANITY_CHECK_beta()