'''
    FUNCTIONS RELATING TO COORDINATE TRANSFORMS
    -------------------------------------------
'''
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import use as mpl_use
mpl_use('agg')
import matplotlib.pyplot as plt





'''
    Quantile transform via bounded rank histogram CDF
    -------------------------------------------------
    This is similar to Jeffrey Anderson's Bounded Normal Rank Histogram, 
    except that I am assigning box-car distributions to the boundaries.
'''
def fit_bounded_rank_histogram_cdf( data1d, exterior_scaling = 1.0, left_bound = None, right_bound = None ):

    # Number of data points
    nPts = len( data1d )

    # Exterior scaling must be within (0,1].
    if ( exterior_scaling > 1 ) or ( exterior_scaling <= 0 ):
        print("ERROR: FIT_BOUNDED_RANK_HISTOGRAM_DISTRIBUTION")
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





def visualize_brh_distribution( brh_pts, brh_cdf ):

    # Generate PDF
    many_pts = np.linspace( brh_pts[0], brh_pts[-1], 1001 )
    dx = many_pts[1] - many_pts[0]
    many_cdf = np.interp( many_pts, brh_pts, brh_cdf )
    pdf_pts = ( many_pts[1:] + many_pts[:-1] )/2.
    pdf_val = (many_cdf[1:] - many_cdf[:-1])/dx

    # Plot PDF and CDF
    fig, axs = plt.subplots( nrows=1, ncols=2, figsize = (6,3) )
    axs[0].plot( pdf_pts, pdf_val, '-r')
    axs[0].set_title('BRH PDF')
    axs[1].plot( brh_pts, brh_cdf, '-r')
    axs[1].set_title('BRH CDF')
    
    
    return fig, axs















if __name__ == '__main__':
    
    # Plot out BRH CDF
    from scipy.stats import norm
    samples = np.random.normal(size=1000)
    brh_pts, brh_cdf = fit_bounded_rank_histogram_cdf( samples, exterior_scaling = 1.0, left_bound = None, right_bound = None )
    fig, axs = visualize_brh_distribution( brh_pts, brh_cdf )

    # Overlay with actual CDF
    cdf_check_pts = np.linspace(brh_pts[0], brh_pts[-1], 1001)
    gaussian_cdf = norm.cdf( cdf_check_pts )
    axs[1].plot( cdf_check_pts, gaussian_cdf, ':k' )
    plt.savefig('visualize_brh.png')
    plt.close()

    