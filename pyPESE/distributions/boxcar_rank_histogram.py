'''
    BOXCAR RANK HISTOGRAM (BRH) DISTRIBUTION
    =================================================================================
    This is similar to Jeffrey Anderson's Gaussian-tailed Rank Histogram, except that
    I am assigning box-car tails outside of the data range.

    The widths and the probability masses of the box-car tails will be determined by
    moment-matching (this requires solving nonlinear equations)

    This BRH distribution will also handle degenerate samples

    List of functions defined here:
    --------------------------------
    1) fit_brh_dist
            Fits BRH distribution to univariate samples
    2) eval_brh_cdf
            Evaluates the CDF of the boxcar rank histogram distribution
            Note: Must first fit the BRH distribution
    3) eval_brh_inv_cdf
            Evaluates the inverse CDF of the boxcar rank histogram distribution
            Note: Must first fit the BRH distribution
    4) eval_brh_pdf
            Evaluates the PDF of the boxcar rank histogram distribution
            Note: Must first fit the BRH distribution
'''


import numpy as np
from copy import deepcopy
# from numba import njit
# from numba import float64 as nb_f64
# from numba.types import Tuple as nb_tuple





















# '''
#     FUNCTION TO FIT BOXCAR RANK HISTOGRAM TO UNIVARIATE SAMPLES
    
#     This function uses a moment-matching scheme to ascertain the boxcar tail widths
#     and boxcar tail probability masses.

    
#     Mandatory Arguments:
#     -----------------
#     1) raw_data1d
#             1D NumPy array containing data samples
                
#     Outputs:
#     --------
#     1) brh_pts
#             Locations defining the piecewise linear CDF of the BRH distribution.

#     2) brh_cdf
#             CDF values of the BRH distribution at locations brh_pts.
# '''
# # @njit( nb_tuple((nb_f64[:], nb_f64[:]))( nb_f64[:] ) ) 
# def fit_brh_dist( raw_data1d ): 

#     # Number of RAW data points
#     num_data = len( raw_data1d )

#     # Unbiased estimation of the first 3 central moments of the RAW data
#     moment1 = np.mean(raw_data1d)
#     moment2 = np.var( raw_data1d, ddof=1 )
#     moment3 = np.sum( np.power(raw_data1d - moment1, 3) ) * num_data / ( (num_data-1) * (num_data-2) )

#     # Map data to space where mean is zero
#     data1d = raw_data1d - moment1

#     # Determine unique data points
#     uniq_data1d = np.unique( data1d )
#     num_uniq_data = len( uniq_data1d )

#     # Seek degenerate data values -- this also determines the probability masses in BRH intervals
#     degen_count = np.ones( num_uniq_data, dtype = 'i4')
#     if ( num_uniq_data != num_data ):
#         for i, val in enumerate(uniq_data1d):
#             degen_count[i] = np.sum( data1d == val )
#         # --- End of loop search for degenerate data values
#     # --- End of check for degenerate data values
     






#     # Set data average to 0 -- this is just a useful thing to do
#     data1d_avg0 = data1d - moment1

#     # Ascertain unique


    



#     return brh_pts, brh_cdf













'''
    FUNCTION TO EVALUATE M-TH RAW MOMENT OF BRH DISTRIBUTION

    Mandatory Arguments:
    1) brh_pts  -- Locations where BRH is defined
    2) brh_cdf  -- CDF of BRH
    3) mom_ord  -- Order of desired raw moment

    This function is based on analytic theory of boxcar tail rank histogram
'''
def eval_brh_mth_raw_moment( brh_pts, brh_cdf, mom_ord ):

    # Number of intervals in BRH distribution
    num_intervals = len(brh_pts) - 1
    
    # Compute width & probability masses alloted to each BRH interval
    interval_masses = brh_cdf[1:] - brh_cdf[:-1]
    interval_widths = brh_pts[1:] - brh_pts[:-1]

    # Compute moment summand from each BRH interval
    interval_moment_summand = (
        np.power( brh_pts[1:], mom_ord + 1 ) - np.power( brh_pts[:-1], mom_ord + 1)
    ) * interval_masses / (mom_ord + 1)

    # Sum all interval moment summands to obtain raw moment
    return np.sum( interval_moment_summand )









'''
    FUNCTION TO SETUP BRH-DEFINING CDF VALUES

    The BRH CDF is a continuous piecewise-linear function. This function is ENTIRELY
    defined by (1) unique data values, (2) CDFs for unique data values, (3) number of 
    unique value occurrences, (4) location where left tail ends, (5) probability mass
    assigned to left tail, (6) location where right tail ends, and (7) probability mass
    assigned to the right tail

    Location where left tail ends is called "left boundary"
    Location where right tail ends is called "right boundary"

'''
def setup_brh_defining_cdf_vals( uniq_data1d, uniq_cnts1d, left_bound, right_bound, left_tail_mass, right_tail_mass ):

    # Determine interior probability mass (assuming boundaries are outside of uniq_data1d's span)
    interior_mass = 1. - (right_tail_mass + left_tail_mass)
  
    # Determine empirical distribution probability mass associated with each non-degenerate data point
    mass_per_point = interior_mass / np.sum(uniq_cnts1d)

    # Determine unique data points that lie within two boundaries
    flag_within_bnds = ( uniq_data1d > left_bound ) * (uniq_data1d < right_bound)
    uniq_data_within_bnds = deepcopy( uniq_data1d[flag_within_bnds] )
    cnts_within_bnds = deepcopy( uniq_cnts1d[flag_within_bnds] )

    # Count number of data points that are either (a) on the left boundary or (b) left of the left boundary
    flag_outside_left = ( uniq_data1d <= left_bound )
    cnt_outside_left = np.sum( uniq_cnts1d[flag_outside_left] )

    # Locations at which BRH CDF values are defined
    brh_pts = np.zeros( len(uniq_data_within_bnds)+2, dtype=np.float64 )
    brh_pts[0] = left_bound
    brh_pts[1:-1] = uniq_data_within_bnds[:]
    brh_pts[-1] = right_bound

    # Initialize BRH CDF values
    brh_cdf = np.zeros( len(brh_pts), dtype=np.float64)
    
    # Leftmost BRH CDF value
    brh_cdf[0] = cnt_outside_left * mass_per_point

    # Accumulate CDF values within interior (left to right)
    for i, cnt in enumerate( cnts_within_bnds ):

        # Accumulate probability mass
        brh_cdf[i+1] = brh_cdf[i] + cnt * mass_per_point

        # Special handling for the left tail mass
        if i == 0:
            brh_cdf[i+1] += left_tail_mass
    
    # Final CDF value
    brh_cdf[-1] = 1.

    return brh_pts, brh_cdf
    






# '''
#     FUNCTION TO EVALUATE BRH CDF

#     Mandatory Arguments:
#     --------------------
#     1) eval_pts
#             Locations to evaluate the BRH CDF
    
#     2) brh_pts
#             Locations defining the piecewise linear BRH CDF
    
#     3) brh_cdf
#             BRH CDF values at locations brh_pts

# '''
# # @njit( nb_f64[:](nb_f64[:],nb_f64[:],nb_f64[:])  )
# def eval_brh_cdf( eval_pts, brh_pts, brh_cdf ):

#     return np.interp( eval_pts, brh_pts, brh_cdf)





















# '''
#     FUNCTION TO EVALUATE BRH QUANTILE FUNCTION
#     This function applies the inverse of the BRH CDF.

#     Mandatory Arguments:
#     --------------------
#     1) eval_cdfs
#             Quantiles to apply the BRH inverse CDF on.
    
#     2) brh_pts
#             Locations defining the piecewise linear BRH CDF
    
#     3) brh_cdf
#             BRH CDF values at locations brh_pts

# '''
# # @njit( nb_f64[:](nb_f64[:],nb_f64[:],nb_f64[:])  )
# def eval_brh_inv_cdf( eval_cdf, brh_pts, brh_cdf ):
    
#     return np.interp( eval_cdf, brh_cdf, brh_pts )




















# '''
#     FUNCTION TO EVALUATE BRH PROBABILITY DENSITY FUNCTION (PDF)
#     This function uses the centered difference method on the BRH CDF
#     to estimate the BRH PDF at specified locations

#     Mandatory Arguments:
#     --------------------
#     1) eval_pts
#             Locations to evaluate the BRH PDF 
    
#     2) brh_pts
#             Locations defining the piecewise linear BRH CDF
    
#     3) brh_cdf
#             BRH CDF values at locations brh_pts
# '''
# # @njit( nb_f64[:](nb_f64[:],nb_f64[:],nb_f64[:])  )
# def eval_brh_pdf( eval_pts, brh_pts, brh_cdf ):

#     # Interval used to estimate BRH PDF values
#     interval = (brh_pts[1:] - brh_pts[:-1]).min() * 1e-3

#     # Identify left and right points used to evaluate 
#     left_pts  = eval_pts - interval/2.
#     right_pts = eval_pts + interval/2.

#     # Evaluate CDF at left and right points
#     left_cdf  = eval_brh_cdf(  left_pts, brh_pts, brh_cdf )
#     right_cdf = eval_brh_cdf( right_pts, brh_pts, brh_cdf )

#     # Evaluate PDF via centered difference
#     return (right_cdf - left_cdf) / interval



    













































































'''
    SANITY CHECKS
'''

if __name__ == '__main__':

    from matplotlib import use as mpl_use
    mpl_use('agg')
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    
    # Draw some values
    np.random.seed(0)
    samples = np.random.normal(size=10)
    samples -= np.mean(samples)
    samples /= np.std(samples)

    # Setup degenerate values
    samples[-1] = samples.min()

    # Unique values
    uniq_data, uniq_cnts = np.unique( samples, return_counts=True)

    # Setup BRH
    left_bound = -1
    right_bound= 1
    brh_pts, brh_cdf = setup_brh_defining_cdf_vals( uniq_data, uniq_cnts, left_bound, right_bound, left_tail_mass=0.1, right_tail_mass=0.1 )

    print(brh_pts)
    print( brh_cdf)

    # Plot out CDF!
    plt.plot( brh_pts, brh_cdf, '-r', zorder=100)
    for i, loc in enumerate(uniq_data):
        print( i,loc, uniq_cnts[i])
        plt.text( loc, 0, str(uniq_cnts[i]), color='red')
    
    plt.axvline( left_bound )
    plt.axvline( right_bound )
    plt.axhline( 0 )
    plt.axhline( 1 )
    plt.xlim([-2,2])
    plt.savefig('tmp.png')