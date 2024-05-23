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





















'''
    FUNCTION TO FIT BOUNDED RANK HISTOGRAM TO UNIVARIATE SAMPLES

    
    Mandatory Arguments:
    -----------------
    1) data1d
        1D NumPy array containing data samples

        
    Optional Arguments (i.e., kwargs):
    ----------------------------------
    1) exterior_scaling (default value: 1.0)
            Controls the probability assigned to the boxcar tails.
            Probability assigned to one boxcar tail is:
                exterior_scaling / number_of_samples

    2) left_bound (default value: None)
            Allows user to manually specify the left boundary of the BRH 
            distribution's support.
            If left_bound is set to None, the left boundary is:
                minval - ( maxval - minval ) / number_of_samples
            If left_bound is a float value and minval > left_bound, then the
            left boundary is set to left_bound.
            If minval < left_bound, the left boundary is set to:
                minval - ( maxval - minval ) / number_of_samples
    
    3) right_bound (default value: None)
            Allows users to manually specify the right boundary of the BRH
            distribution's support.
            If right_bound is set to None, the right boundary is:
                maxval + ( maxval - minval ) / number_of_samples
            If right_bound is a float value and maxval < right_bound, then the
            right boundary is set to right_bound.
            If maxval > right_bound, then the right boundary is set to:
                maxval + ( maxval - minval ) / number_of_samples


    Outputs:
    --------
    1) brh_pts
            Locations defining the piecewise linear CDF of the BRH distribution.

    2) brh_cdf
            CDF values of the BRH distribution at locations brh_pts.
'''

def fit_brh_dist( data1d, exterior_scaling = 1.0, left_bound = None, right_bound = None ):

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
def eval_brh_cdf( eval_pts, brh_pts, brh_cdf ):

    return np.interp( eval_pts, brh_pts, brh_cdf, left=0., right=1.)






















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
    Functions to perform sanity checks
'''

def visualize_brh_distribution( brh_pts, brh_cdf ):

    from matplotlib import use as mpl_use
    mpl_use('agg')
    import matplotlib.pyplot as plt

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



def demo_brh_distribution():
    
    # Plot out BRH CDF
    from scipy.stats import norm
    samples = np.random.normal(size=50)
    brh_pts, brh_cdf = fit_bounded_rank_histogram( samples, exterior_scaling = 0.1, left_bound = None, right_bound = None )
    fig, axs = visualize_brh_distribution( brh_pts, brh_cdf )

    # Overlay with actual CDF
    cdf_check_pts = np.linspace(brh_pts[0], brh_pts[-1], 1001)
    gaussian_cdf = norm.cdf( cdf_check_pts )
    axs[1].plot( cdf_check_pts, gaussian_cdf, ':k' )
    plt.savefig('visualize_brh.png')
    plt.close()

    return

