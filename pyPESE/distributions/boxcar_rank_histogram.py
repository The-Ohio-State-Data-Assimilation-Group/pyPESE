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
from numba import njit
from numba import float64 as nb_f64
from numba.types import Tuple as nb_tuple





















'''
    FUNCTION TO FIT BOXCAR RANK HISTOGRAM TO UNIVARIATE SAMPLES
    
    This function uses a moment-matching scheme to ascertain the boxcar tail widths
    and boxcar tail probability masses.

    CURRENTLY MATCHES MEAN AND VARIANCE

    
    Mandatory Arguments:
    -----------------
    1) raw_data1d
            1D NumPy array containing data samples
                
    Outputs:
    --------
    1) brh_pts
            Locations defining the piecewise linear CDF of the BRH distribution.

    2) brh_cdf
            CDF values of the BRH distribution at locations brh_pts.
'''
# def fit_brh_dist( raw_data1d ):

#     # Number of RAW data points
#     num_data = len( raw_data1d )

#     # Unbiased estimation of the first 2 central moments of the RAW data
#     moment1 = np.mean(raw_data1d)
#     moment2 = np.var( raw_data1d, ddof=1 ) 

#     # Sort data
#     sorted1d = np.sort(raw_data1d)

#     # Determine 
















'''
    FUNCTION TO SETUP BRH-DEFINING CDF VALUES

    The BRH CDF is a continuous piecewise-linear function. 

    ASSUMPTIONS:
        1) All data values are unique
        2) Left and right bounds fall outside of data span.

        
    Mandatory function arguments:
    1) data1d
            1D NumPy array holding ensemble values. Every value is assumed to be unique
    2) left_bound
            Scalar float value indicating the left boundary of the left boxcar tail
    3) right_bound
            Scalar float value indicating the right boundary of the right boxcar tail
    4) left_tail_mass
            Scalar float value indicating the probability mass assigned to the left boxcar tail
    5) right_tail_mass
            Scalar float value indicating the probability mass assigned to the right boxcar tail

'''
# @njit
def setup_brh_defining_cdf_vals( data1d, left_bound, right_bound, left_tail_mass, right_tail_mass ):

    # Sort data values
    sorted_data1d = np.sort(data1d)

    # Number of inter-sample intervals
    num_interior_intervals = len(data1d)-1

    # Probability assigned to each inter-sample interval
    interior_mass = 1. - (right_tail_mass + left_tail_mass)
    mass_per_interval = interior_mass / num_interior_intervals

    # Setup CDF defining points
    brh_pts = np.zeros( num_interior_intervals + 3, dtype=np.float64)
    brh_pts[0] = left_bound
    brh_pts[1:-1] = sorted_data1d
    brh_pts[-1] = right_bound

    # Setup CDF values
    brh_cdf = np.zeros( num_interior_intervals + 3, dtype=np.float64)
    brh_cdf[0] = 0
    brh_cdf[1:-1] = mass_per_interval * np.arange(num_interior_intervals+1) + left_tail_mass
    brh_cdf[-1] = brh_cdf[-2] + right_tail_mass
    print( brh_cdf )

    # Return BRH-defining values
    return brh_pts, brh_cdf










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
    interval_density = interval_masses / interval_widths

    # Compute moment summand from each BRH interval
    interval_moment_summand = (
        np.power( brh_pts[1:], mom_ord + 1 ) - np.power( brh_pts[:-1], mom_ord + 1)
    ) * interval_density / (mom_ord + 1) 

    # Sum all interval moment summands to obtain raw moment
    return np.sum( interval_moment_summand )

        



    



    



    # # Determine interior probability mass (assuming boundaries are outside of uniq_data1d's span)
    # interior_mass = 1. - (right_tail_mass + left_tail_mass)
  
    # # Determine empirical distribution probability mass associated with each non-degenerate data point
    # mass_per_point = interior_mass / np.sum(uniq_cnts1d)

    # # Determine unique data points that lie within two boundaries
    # flag_within_bnds = ( uniq_data1d > left_bound ) * (uniq_data1d < right_bound)
    # uniq_data_within_bnds = deepcopy( uniq_data1d[flag_within_bnds] )
    # cnts_within_bnds = deepcopy( uniq_cnts1d[flag_within_bnds] )

    # # Count number of data points that are either (a) on the left boundary or (b) left of the left boundary
    # flag_outside_left = ( uniq_data1d <= left_bound )
    # cnt_outside_left = np.sum( uniq_cnts1d[flag_outside_left] )

    # # Locations at which BRH CDF values are defined
    # brh_pts = np.zeros( len(uniq_data_within_bnds)+2, dtype=np.float64 )
    # brh_pts[0] = left_bound
    # brh_pts[1:-1] = uniq_data_within_bnds[:]
    # brh_pts[-1] = right_bound

    # # Initialize BRH CDF values
    # brh_cdf = np.zeros( len(brh_pts), dtype=np.float64)
    
    # # Leftmost BRH CDF value
    # brh_cdf[0] = cnt_outside_left * mass_per_point

    # # Accumulate CDF values within interior (left to right)
    # for i, cnt in enumerate( cnts_within_bnds ):

    #     # Accumulate probability mass
    #     brh_cdf[i+1] = brh_cdf[i] + cnt * mass_per_point

    #     # Special handling for the left tail mass
    #     if i == 0:
    #         brh_cdf[i+1] += left_tail_mass
    
    # # Final CDF value
    # brh_cdf[-1] = 1.

    # return brh_pts, brh_cdf
    






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
    # np.random.seed(0)
    samples = norm.ppf( np.linspace( 0.01, 0.99, 11) )*2 + 2
    print( np.mean(samples))

    # Set up some BRH
    brh_pts, brh_cdf = setup_brh_defining_cdf_vals( samples, -10, 10, 0.1, 0.1)

    # Plot the BRH
    plt.plot( brh_pts, brh_cdf, '-r')
    plt.savefig('tmp.png')

    # Analytic mean of BRH
    analytic_integ = eval_brh_mth_raw_moment(brh_pts, brh_cdf, 0)
    analytic_avg = eval_brh_mth_raw_moment(brh_pts, brh_cdf, 1)
    analytic_var = eval_brh_mth_raw_moment(brh_pts-analytic_avg, brh_cdf, 2)

    # Draw many many samples from brh
    draws_from_brh = np.interp( np.random.uniform(size=100000), brh_cdf, brh_pts )

    print( 'sampled avg ', np.mean(draws_from_brh), ' analytic avg ', analytic_avg)
    print( 'sampled var ', np.var(draws_from_brh, ddof=1), ' analytic var ', analytic_var)
