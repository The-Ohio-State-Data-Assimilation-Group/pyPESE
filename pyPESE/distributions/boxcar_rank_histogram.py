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
from scipy.optimize import root
from numba import njit
from numba import float64 as nb_f64
from numba.types import Tuple as nb_tuple










'''
    SciPy-like class for BRH distribution.
    This is a univariate bounded rank histogram distribution!!!
'''
class boxcar_rank_histogram:

    # Initialize 
    def __init__( self, brh_pts, brh_cdf ):
        self.brh_pts = brh_pts
        self.brh_cdf = brh_cdf
        return

    # Fit BRH distirbution to 1d data
    def fit( data1d ):
        # For each variable, fit BRH distribution
        return fit_brh_dist( data1d )

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
        samples1d = eval_brh_inv_cdf( uniform_samples1d )
        return samples1d.reshape(shape)
    
# ------ End of BRH distribution SciPy-like class












'''
    FUNCTION TO FIT BOXCAR RANK HISTOGRAM TO UNIVARIATE SAMPLES
    
    This function uses a moment-matching scheme to ascertain the boxcar tail widths
    and boxcar tail probability masses.

    CURRENTLY MATCHES MEAN AND VARIANCE

    
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
def fit_brh_dist( data1d ):

    # Number of RAW data points
    num_data = len( data1d )

    # Unbiased estimation of data mean and variance
    # ---------------------------------------------
    # BRH distribution will be fitted s.t. it's mean and variance is the same as data1d.
    targetted_mean      = np.mean(data1d)
    targetted_variance  = np.var( data1d, ddof=1 ) 


    # Solve for fitting parameters
    # ----------------------------
    tail_width = np.sqrt( targetted_variance )
    tail_mass = 0.
    firstguess = np.array( [tail_width, tail_mass ])
    soln = root(
        brh_two_moment_fitting_vector_func, firstguess, 
        args = (data1d, targetted_mean, targetted_variance)
    )

    # Load the solution
    fitted_tail_width, fitted_tail_mass = soln.x

    # Use crudest solution if convergence failed or weird solutions are outputted
    if ( soln.success == False 
            or fitted_tail_width < 0 
            or fitted_tail_mass < 0 
            or fitted_tail_mass > 0.5
        ):
        fitted_tail_width = np.sqrt( targetted_variance )
        fitted_tail_mass = 0
    

    
    # Setup fitted BRH and return
    data_min = data1d.min()
    data_max = data1d.max()
    brh_pts, brh_cdf = setup_brh_defining_cdf_vals(
                            data1d, 
                            data_min-fitted_tail_width,
                            data_max+fitted_tail_width, 
                            fitted_tail_mass, 
                            fitted_tail_mass
    )

    return brh_pts, brh_cdf














'''
    VECTOR FUNCTION THAT BRH-FITTING SOLVES
'''
def brh_two_moment_fitting_vector_func( brh_properties, data1d, targ_mean, targ_vari ):

    # Extract tail width and tail mass from brh_properties vector
    tail_width, tail_mass = brh_properties

    # Setup BRH distribution
    data_min = data1d.min()
    data_max = data1d.max()
    brh_pts, brh_cdf = setup_brh_defining_cdf_vals( 
        data1d, data_min-tail_width, data_max+tail_width, 
        tail_mass, tail_mass
    )

    # Evaluate BRH mean & variance
    brh_mean = eval_brh_mth_raw_moment( brh_pts, brh_cdf, 1 )
    brh_vari = eval_brh_mth_raw_moment( brh_pts-brh_mean, brh_cdf, 2 )

    # Compute difference from desired mean and variance    
    return np.array( [ brh_mean - targ_mean, brh_vari - targ_vari] )
    
    














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
# @njit( nb_f64[:](nb_f64[:],nb_f64[:],nb_f64[:])  )
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
# @njit( nb_f64[:](nb_f64[:],nb_f64[:],nb_f64[:])  )
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
# @njit( nb_f64[:](nb_f64[:],nb_f64[:],nb_f64[:])  )
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
    
    samples = np.random.normal( size=100)

    brh_pts, brh_cdf = boxcar_rank_histogram.fit(samples)
    brh_dist = boxcar_rank_histogram( brh_pts, brh_cdf)
    
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
    gaussian_cdf = norm.cdf(  many_pts )
    axs[1].plot( many_pts, gaussian_cdf, ':k' )
    plt.savefig('visualize_brh.png')
    plt.close()



    print( 'checking boxcar rank histogram moments')
    brh_avg = eval_brh_mth_raw_moment( brh_pts, brh_cdf, 1)
    print( 'ens mean ', np.mean(samples), ' fitted BRH mean: ', brh_avg )
    brh_var = eval_brh_mth_raw_moment( brh_pts, brh_cdf, 2)
    print( 'ens var ', np.var(samples, ddof=1), ' fitted BRH var: ', brh_var )