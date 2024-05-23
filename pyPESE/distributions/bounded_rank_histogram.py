'''
    BOUNDED RANK HISTOGRAM (BRH) DISTRIBUTION
    =================================================================================
    This is similar to Jeffrey Anderson's Bounded Normal Rank Histogram, except that
    I am assigning box-car tails outside of the data range.

    The width of the box-car tails is currently set to (maxval - minval) / num_data
    where minval is the data sample's minimum value, and maxval is the data sample's 
    maximum value.


    List of functions defined here:
    --------------------------------
    1) fit_brh_dist
            Fits BRH distribution to univariate samples
    2) eval_brh_cdf
            Evaluates the CDF of the bounded rank histogram distribution
            Note: Must first fit the BRH distribution
    3) eval_brh_inv_cdf
            Evaluates the inverse CDF of the bounded rank histogram distribution
            Note: Must first fit the BRH distribution
    4) eval_brh_pdf
            Evaluates the PDF of the bounded rank histogram distribution
            Note: Must first fit the BRH distribution
'''


import numpy as np
from numba import njit
from numba import float64 as nb_f64
from numba.types import Tuple as nb_tuple



# Hardcoded GLOBAL options controlling BRH
exterior_scaling = 0.1
left_bound = None
right_bound = None
# will incorporate these options into BRH fitting procedure in the future.





















'''
    SciPy-like class for BRH distribution.
    This is a univariate bounded rank histogram distribution!!!
'''
class bounded_rank_histogram:

    # Initialize 
    def __init__( self ):
        return

    # Fit BRH distirbution to 1d data
    def fit( self, data1d ):
        # For each variable, fit BRH distribution
        self.brh_pts, self.brh_cdf = fit_brh_dist( data1d )
        return

    # Function to evaluate CDF of fitted BRH. 
    def cdf(self, eval_pts):
        return eval_brh_cdf( eval_pts, self.brh_pts, self.brh_cdf )

    # Function to evaluate inverse CDF of fitted BRH
    def ppf(self, eval_cdf):
        return eval_brh_inv_cdf( eval_cdf, self.brh_pts, self.brh_cdf )
    
    # Function to evaluate PDF of fitted BRH
    def pdf( self, eval_pts ):
        return eval_brh_pdf( eval_pts, self.brh_pts, self.brh_cdf )
        
    # Function to draw samples consistent with fitted BRH
    def rvs( self, shape ):
        uniform_samples1d = np.random.uniform( size=np.prod(shape) )
        samples1d = ppf( uniform_samples1d )
        return samples1d.reshape(shape)
    
# ------ End of BRH distribution SciPy-like class

























'''
    FUNCTION TO FIT BOUNDED RANK HISTOGRAM TO UNIVARIATE SAMPLES

    
    Mandatory Arguments:
    -----------------
    1) data1d
        1D NumPy array containing data samples


    Outputs:
    --------
    1) brh_pts
            Locations defining the piecewise linear CDF of the BRH distribution.

    2) brh_cdf
            CDF values of the BRH distribution at locations brh_pts.
'''
@njit( nb_tuple((nb_f64[:], nb_f64[:]))( nb_f64[:] ) ) 
def fit_brh_dist( data1d ): 

    # Number of data points
    nPts = len( data1d )

    # Exterior scaling must be within (0,1].
    if ( exterior_scaling > 1 ) or ( exterior_scaling <= 0 ):
        print("ERROR: fit_brh_dist")
        print("    Exterior scaling factor must be between 0 and 1.")


    # Probability alloted beyond the boundaries of data's min-max range
    # (i.e., exterior zones). There are two exterior zones.
    exterior_prob = ( 1. / nPts  ) * exterior_scaling

    # Compute interior probability (i.e., probability within the range
    # of the data's min-max)
    internal_prob = 1. - 2.*exterior_prob

    # Interval probability (i.e., probability assigned to each interval 
    # between consecutive data points)
    interval_prob = internal_prob / (nPts - 1.)

    # Setup internal interval boundaries for the bounded rank histogram distribution
    brh_pts = np.zeros( nPts + 2 )
    brh_pts[1:-1] = np.sort(data1d)
    minmax_interval = data1d.max() - data1d.min()

    # Generate left boundary
    if left_bound == None:
        brh_pts[0]  = brh_pts[1] - minmax_interval/nPts
    elif left_bound < data1d.min():
        brh_pts[0] = left_bound 
    else: 
        brh_pts[0]  = brh_pts[1] - minmax_interval/nPts
    
    # Generate right boundary
    if right_bound == None:
        brh_pts[-1]  = brh_pts[-2] + minmax_interval/nPts
    elif right_bound > data1d.max():
        brh_pts[-1] = right_bound 
    else: 
        brh_pts[-1]  = brh_pts[-2] + minmax_interval/nPts


    # Evaluate BRH CDF at every bounding point
    brh_cdf = np.zeros( nPts + 2 )
    brh_cdf[1:-1] = np.arange(nPts) * interval_prob + exterior_prob
    brh_cdf[-1] = 1.

    return brh_pts, brh_cdf






# @njit( nb_tuple( (nb_f64[:,::1],nb_f64[:,::1]) )(nb_f64[:,::1]) )
# def multivariate_fit_brh( ens_data2d ):

#     # Setup desirable structures
#     nVar, nEns = ens_data2d.shape
#     brh_pts2d = np.zeros( (nVar, nEns+2) )
#     brh_cdf2d = np.zeros( (nVar, nEns+2) )

#     # Loop over variables
#     for vv in range( ens_data2d.shape[0] ):
#         brh_pts2d[vv,:], brh_cdf2d[vv,:] = fit_brh_dist( ens_data2d[vv,:] )
    
#     return brh_pts2d, brh_cdf2d





















'''
    FUNCTION TO EVALUATE BRH CDF

    Mandatory Arguments:
    --------------------
    1) eval_pts
            Locations to evaluate the BRH CDF
    
    2) brh_pts
            Locations defining the piecewise linear BRH CDF
    
    3) brh_cdf
            BRH CDF values at locations brh_pts

'''
@njit( nb_f64[:](nb_f64[:],nb_f64[:],nb_f64[:])  )
def eval_brh_cdf( eval_pts, brh_pts, brh_cdf ):

    return np.interp( eval_pts, brh_pts, brh_cdf)





















'''
    FUNCTION TO EVALUATE BRH QUANTILE FUNCTION
    This function applies the inverse of the BRH CDF.

    Mandatory Arguments:
    --------------------
    1) eval_cdfs
            Quantiles to apply the BRH inverse CDF on.
    
    2) brh_pts
            Locations defining the piecewise linear BRH CDF
    
    3) brh_cdf
            BRH CDF values at locations brh_pts

'''
@njit( nb_f64[:](nb_f64[:],nb_f64[:],nb_f64[:])  )
def eval_brh_inv_cdf( eval_cdf, brh_pts, brh_cdf ):
    
    return np.interp( eval_cdf, brh_cdf, brh_pts )




















'''
    FUNCTION TO EVALUATE BRH PROBABILITY DENSITY FUNCTION (PDF)
    This function uses the centered difference method on the BRH CDF
    to estimate the BRH PDF at specified locations

    Mandatory Arguments:
    --------------------
    1) eval_pts
            Locations to evaluate the BRH PDF 
    
    2) brh_pts
            Locations defining the piecewise linear BRH CDF
    
    3) brh_cdf
            BRH CDF values at locations brh_pts
'''
@njit( nb_f64[:](nb_f64[:],nb_f64[:],nb_f64[:])  )
def eval_brh_pdf( eval_pts, brh_pts, brh_cdf ):

    # Interval used to estimate BRH PDF values
    interval = (brh_pts[1:] - brh_pts[:-1]).min() * 1e-3

    # Identify left and right points used to evaluate 
    left_pts  = eval_pts - interval/2.
    right_pts = eval_pts + interval/2.

    # Evaluate CDF at left and right points
    left_cdf  = eval_brh_cdf(  left_pts, brh_pts, brh_cdf )
    right_cdf = eval_brh_cdf( right_pts, brh_pts, brh_cdf )

    # Evaluate PDF via centered difference
    return (right_cdf - left_cdf) / interval



    













































































'''
    SANITY CHECKS
'''

if __name__ == '__main__':

    from matplotlib import use as mpl_use
    mpl_use('agg')
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    
    samples = np.random.normal(size=500)

    brh_dist = bounded_rank_histogram() 
    brh_dist.fit(samples)
    
    many_pts = np.linspace( -4,4, 1000 )
    pdf_vals = brh_dist.pdf( many_pts )
    cdf_vals = brh_dist.cdf( many_pts )


    # Plot PDF and CDF
    fig, axs = plt.subplots( nrows=1, ncols=2, figsize = (6,3) )
    axs[0].plot( many_pts, pdf_vals, '-r')
    axs[0].set_title('BRH PDF')
    axs[1].plot( many_pts, cdf_vals, '-r')
    axs[1].set_title('BRH CDF')

    # Overlay with actual CDF
    gaussian_cdf = norm.cdf( many_pts )
    axs[1].plot( many_pts, gaussian_cdf, ':k' )
    plt.savefig('visualize_brh.png')
    plt.close()