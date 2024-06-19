'''
    BOUNDED BOXCAR RANK HISTOGRAM (BBRH) DISTRIBUTION
    =================================================================================
    This is similar to Jeffrey Anderson's Gaussian-tailed Rank Histogram, except that
    I am assigning box-car tails outside of the data range.

    The BBRH distribution differs from the boxcar rank histogram (BRH) distribution in
    terms of:
    1) Termination location of the tails are fixed in BBRH whereas those of BBRH are 
       computed. 
    2) BBRH's left and right tails can have different probability masses whereas those
       of BBRH have the same probability mass.

    When it comes to moment matching, fitting the BBRH only requires solving a system of 
    THREE linear equations (efficient). In contrast, fitting the BBRH requires solving a
    system of nonlinear polynomials (less efficient than the BBRH fitting process). These
    difference in fitting processes is due to the differences in the two distributions'
    treatment of tail probability masses and tail termination locations.
'''


import numpy as np
from copy import deepcopy
from numba import njit
from numba import float64 as nb_f64
from numba import float32 as nb_f32
from numba import int64 as nb_i64
from numba.types import Tuple as nb_tuple


























'''
    SciPy-like class for BBRH distribution.
    This is a univariate bounded rank histogram distribution!!!
'''
class bounded_boxcar_rank_histogram:

    name = 'bounded boxcar rank histogram'


    # Initialize 
    def __init__( self, cdf_locs, cdf_vals ):
        self.cdf_locs = cdf_locs
        self.cdf_vals = cdf_vals
        return

    # Fit BBRH distirbution to 1d data
    def fit( data1d, min_bound=-1e9, max_bound=1e9 ):
        # For each variable, fit BBRH distribution
        return BBRH_fit_dist_to_ens( data1d, min_bound, max_bound )

    # Function to evaluate CDF of fitted BBRH. 
    def cdf(self, eval_pts):
        return eval_bbrh_cdf( eval_pts, self.cdf_locs, self.cdf_vals )

    # Function to evaluate inverse CDF of fitted BBRH
    def ppf(self, eval_cdf):
        return eval_bbrh_inv_cdf( eval_cdf, self.cdf_locs, self.cdf_vals )
    
    # Function to evaluate PDF of fitted BBRH
    def pdf( self, eval_pts ):
        return eval_bbrh_pdf( eval_pts, self.cdf_locs, self.cdf_vals  )
        
    # Function to draw samples consistent with fitted BBRH
    def rvs( self, shape ):
        uniform_samples1d = np.random.uniform( size=np.prod(shape) )
        samples1d = self.ppf( uniform_samples1d )
        return samples1d.reshape(shape)
    
# ------ End of BBRH distribution SciPy-like class


















'''
    Function to fit BBRH distribution to an ensemble via matching the first two moments

    Mandatory arguments:
    --------------------
    1) raw_ens1d
            1D NumPy array containing an ensemble of values for a forecast model variable
    2) min_bound
            User-specified scalar value indicating the left boundary of BBRH's support
    3) max_bound
            User-specified scalar value indicating the right boundary of BBRH's support

    Output:
    -------
    1) cdf_locs
            1D NumPy array indicating locations where BBRH's CDF is defined.
            Note that cdf_locs[1:-1] contains the preprocessed ensemble.
    2) cdf_vals
            1D NumPy array of BBRH CDF values at cdf_locs

            
    Additional note:
        No Just-In-Time decorator because there are no loops inside this function
'''
def BBRH_fit_dist_to_ens( raw_ens1d, min_bound, max_bound ):

    # Ensemble size
    ens_size = raw_ens1d.shape[0]


    # Determine first two moments of the ensemble
    # -------------------------------------------
    ens_moment1 = np.mean(          raw_ens1d     )
    ens_moment2 = np.mean( np.power(raw_ens1d, 2) )


    # Preprocess the ensemble to remove duplicates & out-of-bounds
    # -------------------------------------------------------------
    ens1d = preprocess_ens( raw_ens1d, min_bound, max_bound )


    # Generate locations at which the BBRH CDF is defined
    # ---------------------------------------------------
    cdf_locs       = np.zeros( ens_size + 2, dtype='f8' )
    cdf_locs[1:-1] = np.sort(ens1d)
    cdf_locs[0]    = min_bound
    cdf_locs[-1]   = max_bound


    # Fit BBRH by solving a system of linear equations
    # ------------------------------------------------
    left_tail_mass, right_tail_mass, interior_interval_mass = (
        BBRH_solve_moment_matching_equations( cdf_locs, ens_size, ens_moment1, ens_moment2 )
    )

    # Exception case: tail masses must be positive semi-definite and the interior interval
    # mass must be positive definite
    if ( left_tail_mass < 0 or right_tail_mass < 0 or interior_interval_mass <= 0 ):
        left_tail_mass          = 1./(ens_size +1)
        right_tail_mass         = 1./(ens_size +1)
        interior_interval_mass  = 1./(ens_size +1)
    # ---- End of exception handling

    # Construct CDF values
    cdf_vals = np.zeros( ens_size+2, dtype='f8' )
    cdf_vals[0] = 0.
    cdf_vals[1:-1] = left_tail_mass + np.arange(ens_size ) * interior_interval_mass
    cdf_vals[-1] = cdf_vals[-2] + right_tail_mass


    return cdf_locs, cdf_vals


















'''
    Function to solve the system of equations to fit BBRH to a set of CDF locations.

    See Bounded_Boxcar_Rank_Histogram_distribution_theory.jpg for those equations.
    Note that this system of equations is a linear system.

    Mandatory arguments:
    --------------------
    1) cdf_locs
            1D NumPy array indicating locations where BBRH's CDF is defined.
            Note that cdf_locs[1:-1] contains the preprocessed ensemble.

    Output:
    -------
    1) alpha
            Probability mass on the left-tail of the BBRH
    2) beta
            Probability mass on the right-tail of the BBRH
    3) gamma
            Probability mass in each interval in the span of the preprocessed ensemble.

            
    Additional note:
        No Just-In-Time decorator because there are no loops inside this function
'''
# @njit( nb_tuple( nb_f64, nb_f64, nb_f64 )( nb_f64[:], nb_i64, nb_f64, nb_f64 ) )
def BBRH_solve_moment_matching_equations( cdf_locs, ens_size, ens_moment1, ens_moment2 ):

    # Construct system of linear equations for the moment matching process
    # --------------------------------------------------------------------
    # See Bounded_Boxcar_Rank_Histogram_distribution_theory.jpg for that system of equations

    # Compute f_n values defined in Bounded_Boxcar_Rank_Histogram_distribution_theory.jpg
    fn_vals = cdf_locs[1:] + cdf_locs[:-1]

    # Compute g_n values defined in Bounded_Boxcar_Rank_Histogram_distribution_theory.jpg
    gn_vals = np.power( cdf_locs[1:], 3) - np.power( cdf_locs[:-1], 3)
    gn_vals /= ( cdf_locs[1:] - cdf_locs[:-1])

    # Construct coefficient matrix for the system of equations. 
    # Note that solution vector is [alpha, gamma, beta].T
    coef_matrix = np.zeros( (3,3), dtype='f8' )

    # Coefficients for the first moment equation
    coef_matrix[0,0] = fn_vals[0] / 2
    coef_matrix[0,1] = np.sum(fn_vals[1:-1])/2
    coef_matrix[0,2] = fn_vals[-1] /2

    # Coefficients for the second moment equation
    coef_matrix[1,0] = gn_vals[0] / 3
    coef_matrix[1,1] = np.sum(gn_vals[1:-1]) / 3
    coef_matrix[1,2] = gn_vals[-1] / 3

    # Coefficients for the probability mass equation
    coef_matrix[2,0] = 1.
    coef_matrix[2,1] = ens_size - 1.
    coef_matrix[2,2] = 1.

    # Vector of values on the right-hand-side of the system of linear equations
    rhs_vec = np.matrix( np.zeros( (3,1), dtype='f8' ) )
    rhs_vec[0] = ens_moment1
    rhs_vec[1] = ens_moment2
    rhs_vec[2] = 1.

    # Compute values of alpha, beta and gamma
    inv_coef_matrix = np.matrix( np.linalg.inv( coef_matrix ) )
    soln_vec = inv_coef_matrix * rhs_vec
    alpha = soln_vec[0,0]
    gamma = soln_vec[1,0]
    beta  = soln_vec[2,0]

    return alpha, beta, gamma




    








'''
    FUNCTION TO EVALUATE BBRH CDF

    Mandatory Arguments:
    --------------------
    1) eval_pts
            Locations to evaluate the BBRH CDF
    
    2) bbrh_pts
            Locations defining the piecewise linear BBRH CDF
    
    3) bbrh_cdf
            BBRH CDF values at locations bbrh_pts

'''
# @njit( nb_f64[:](nb_f64[:],nb_f64[:],nb_f64[:])  )
def eval_bbrh_cdf( eval_pts, bbrh_pts, bbrh_cdf ):

    return np.interp( eval_pts, bbrh_pts, bbrh_cdf )





















'''
    FUNCTION TO EVALUATE BBRH QUANTILE FUNCTION
    This function applies the inverse of the BBRH CDF.

    Mandatory Arguments:
    --------------------
    1) eval_cdfs
            Quantiles to apply the BBRH inverse CDF on.
    
    2) bbrh_pts
            Locations defining the piecewise linear BBRH CDF
    
    3) bbrh_cdf
            BBRH CDF values at locations bbrh_pts

'''
# @njit( nb_f64[:](nb_f64[:],nb_f64[:],nb_f64[:])  )
def eval_bbrh_inv_cdf( eval_cdf, bbrh_pts, bbrh_cdf ):
    
    return np.interp( eval_cdf, bbrh_cdf, bbrh_pts )




















'''
    FUNCTION TO EVALUATE BBRH PROBABILITY DENSITY FUNCTION (PDF)
    This function uses the centered difference method on the BBRH CDF
    to estimate the BBRH PDF at specified locations

    Mandatory Arguments:
    --------------------
    1) eval_pts
            Locations to evaluate the BBRH PDF 
    
    2) bbrh_pts
            Locations defining the piecewise linear BBRH CDF
    
    3) bbrh_cdf
            BBRH CDF values at locations bbrh_pts
'''
# @njit( nb_f64[:](nb_f64[:],nb_f64[:],nb_f64[:])  )
def eval_bbrh_pdf( eval_pts, bbrh_pts, bbrh_cdf ):

    # Interval used to estimate BBRH PDF values
    interval = (bbrh_pts[1:] - bbrh_pts[:-1]).min() * 1e-3

    # Identify left and right points used to evaluate 
    left_pts  = eval_pts - interval/2.
    right_pts = eval_pts + interval/2.

    # Evaluate CDF at left and right points
    left_cdf  = eval_bbrh_cdf(  left_pts, bbrh_pts, bbrh_cdf )
    right_cdf = eval_bbrh_cdf( right_pts, bbrh_pts, bbrh_cdf )

    # Evaluate PDF via centered difference
    return (right_cdf - left_cdf) / interval





































'''
    Function to preprocess ensemble to handle out-of-bounds values and duplicate values

    Such values can cause issues with certain distributions, and issues with resampling.

    Mandatory arguments:
    --------------------
    1) input_ens1d
            1D NumPy array containing an ensemble of values for a forecast model variable
    2) min_bound
            User-specified scalar value indicating the left boundary of BBRH's support
    3) max_bound
            User-specified scalar value indicating the right boundary of BBRH's support

    Output:
    -------
    1) ens1d
            1D NumPy array containing preprocessed ensemble (NOT SORTED!!!)
    
'''
def preprocess_ens( input_ens1d, min_bound, max_bound ):

    # Sort ensemble values
    # --------------------
    # This procedure also guards against unintentional object-passing effects because
    # an internal copy of the ensemble of values is generated.
    ens1d = np.sort( input_ens1d )
    ens_size = ens1d.shape[0]

    
    # Determine rank statistics of the input ensemble
    # -----------------------------------------------
    ens1d_inds = np.argsort( np.argsort( input_ens1d ) )


    # Define an offset value
    # ----------------------
    # To handle value duplicate & out-of-bound values, a small offset must be defined.
    # The reason for this definition will be apparant later in this function.

    # Determine smallest interval between unique values in BBRH
    all_vals = np.zeros( ens_size+2, dtype='f' )
    all_vals[0] = min_bound
    all_vals[1] = max_bound
    all_vals[2:] = ens1d
    uniq_vals = np.unique(ens1d)
    smallest_uniq_interval = (uniq_vals[1:] - uniq_vals[:-1]).min()

    # Define offset value based on smallest interval between unique values
    offset_val = smallest_uniq_interval / (ens_size)


    # Handling duplicate values within the bounded ensemble
    # -----------------------------------------------------
    # Detect duplicates
    uniq_vals, uniq_cnts = np.unique( ens1d, return_counts=True )
    flags_duplicated_uniq_vals = uniq_cnts > 1

    # Reconstruct ensemble with all duplicates removed
    ens1d[:] = np.nan
    ind = 0
    for iuniq in range( uniq_vals.shape[0] ):
        cnt = uniq_cnts[iuniq]
        new_vals  = np.arange(cnt) * offset_val 
        new_vals -= np.mean(new_vals)
        new_vals += uniq_vals[iuniq]
        ens1d[ind:ind+cnt] = new_vals
        ind += cnt
    # ---- End of ensemble reconstruction


    # Handling out-of-bounds ensemble values
    # --------------------------------------
    # Handle overly small ensemble values by relocating them to the left boundary 
    # of the support, PLUS some offset
    flags_oversmall = (ens1d <= min_bound)
    num_oversmall   = np.sum(flags_oversmall)
    ens1d[flags_oversmall] = min_bound + ( np.arange(num_oversmall)+1 ) * offset_val

    # Handle overly large ensemble values by relocating them to the right boundary
    # of the support, MINUS some offset
    flags_overlarge = (ens1d >= max_bound)
    num_overlarge   = np.sum( flags_overlarge )
    ens1d[flags_overlarge] = max_bound - ( np.arange(num_overlarge)+1 )[::-1] * offset_val
    

    # Return de-sorted preprocessed ensemble
    return ens1d[ens1d_inds]





















if __name__ == '__main__':

    from matplotlib import use as mpl_use
    mpl_use('agg')
    import matplotlib.pyplot as plt
#     from pyPESE.utility import preprocess_ens

    # Generate a test ensemble with degenerate values
    np.random.seed(0)
    raw_ens = np.random.normal( size=30 )
    raw_ens[10:20] = 0.

    ens1d = preprocess_ens( raw_ens, -1, 2)

    # Generate BBRH distribution with nasty bounds
    cdf_locs, cdf_vals = bounded_boxcar_rank_histogram.fit( ens1d, -1., 2. )
    bbrh_obj = bounded_boxcar_rank_histogram( cdf_locs, cdf_vals )

    random_samples = bbrh_obj.rvs( (1000,1000) )

    print( np.mean( ens1d), np.var(ens1d, ddof=1))
    print( np.mean( random_samples), np.var(random_samples, ddof=1))


    plt.plot( cdf_locs, cdf_vals )
    plt.scatter( raw_ens, raw_ens*0-0.1, marker = 'x', s=20, c='r' )
    plt.scatter( cdf_locs[1:-1], cdf_locs[1:-1]*0-0.2, marker = 'x', s=20, c='k' )
    plt.savefig('test.png')