'''
    MIXTURE USER + WEIGHTED EMPIRICAL DISTRIBUTION (MUWE DISTRIBUTIOON)
    =================================================================================
    In an ensemble of simulations, there can sometimes be repeated (i.e.) degenerate values.
    For example, multiple WRF members' QCLOUD value at the same location can be zero. These
    degenerate values interfere with probit transforms. We will treat degenerate values as 
    being drawn from a dirac-delta distribution, and the non-degenerate values as being drawn
    from another distribution. In other words, we use a mixture of two distributions when 
    there are degenerate values: (1) the user-specified distribution, and (2) weighted 
    empirical distribution. This mixture distribution's pdf is 
    \begin{equation}
        p_{\text{muwe}}(x) := \sum^{N_d}_{j=1} w_j * delta(x-x_j)
                                + ( w_{\text{user}} ) * {p_{\text{user}}(x)}
    \end{equation}

    where $N_d$ is number of unique degenerate ensemble values, $x_j$ refers to those unique degenerate
    values, $w_j$ is the weight assigned to each degenerate value
    \begin{equation}
        w_j := (number of members with value x_j) / (ensemble size),
    \end{equation}
    $w_{\text{user}}$ is the weight assigned to the user distribution
    \begin{equation}
        w_{\text{user}} := (number of non degenerate members) / (ensemble size),
    \end{equation}
    and ${p_{\text{user}}(x)}$ is the pdf of the user-specified distribution. Note that
    \begin{equation}
        w_{\text{user}} + \sum^{N_d}_{j=1} w_j = 1
    \end{equation}
    and all weights are positive semidefinite.

    When there are no degenerate values, the mixture distribution turns into the user-specified 
    distribution. As such, the delta+user mixture distribution should be used by default in PESE-GC.

    ChatGPT transcript used to help with coding: https://chatgpt.com/share/68334900-9414-8008-8a92-ce219ca16c2e
'''


import numpy as np
from copy import deepcopy
from numba import njit
from numba import float64 as nb_f64




'''
    Class defining the mixture user + weighted empirical distribution

    Note: This function is initialized using SciPy-like distribution classes.
'''
class mixture_user_weighted_empirical: 

    name = 'generic mixture user + weighted empirical distribution'

    # Initialization
    def __init__( self, delta_pts, delta_weights, user_dist_inst, user_weight):
        self.delta_pts      = delta_pts
        self.delta_weights  = delta_weights
        self.user_dist      = user_dist_inst
        self.user_weight    = 1. - np.sum(delta_weights)   
        return  

    # Fit muwe distribution to 1d ensemble data
    def fit( data1d, user_dist_class ):
        
        # Identifying delta points and weights
        vals, cnt = np.unique( data1d, return_counts=True )
        delta_pts = vals[ cnt>1 ]
        delta_weights = cnt[cnt>1] * 1. / len(data1d)

        # Determine user distribution parameters based on non-degen values
        user_params = user_dist_class.fit( vals[cnt==1] )

        # Generate instance of user distribution
        user_dist_inst = user_dist_class(*user_params)

        # Return parameters needed to initialize muwe instance
        return delta_pts, delta_weights, user_dist_inst
    
    # CDF function
    def cdf( self, eval_pts ):
        return muwe_cdf( self.delta_pts, self.delta_weights, self.user_dist, self.user_weight, eval_pts )
    
    # PPF function
    def ppf( self, eval_cdf ):
        return muwe_ppf_fast( self.delta_pts, self.delta_weights, self.user_dist, self.user_weight, eval_cdf )
    
    # PDF function -- estimated via taking numerical gradient of CDF
    def pdf( self, eval_pts ):
        eval_left  = eval_pts * (1 - 1e-4) - 1e-6
        eval_right = eval_pts * (1 + 1e-4) + 1e-6
        dx = eval_right - eval_left
        return (self.cdf( eval_right ) - self.cdf( eval_left ))/dx
    
    # Random sampling
    def rvs( self, shape ):
        uniform_samples1d = np.random.uniform( size=np.prod(shape) )
        samples1d = self.ppf( uniform_samples1d )
        return samples1d.reshape(shape)

# # ------------ End of definition for mixture_user_weighted_empirical distribution SciPy-like class



























'''
    FUNCTION TO EVALUATE CDF OF MIXTURE USER AND WEIGHTED EMIRICAL DISTRIBUTION

    Inputs:
    1) delta_pts        -- Locations of delta functions (a sorted 1D NumPy array)
    2) delta_weights    -- Weights assigned to each delta function (1D NumPy array)
    3) user_dist        -- Fitted user-specified SciPy-like distribution instance
    4) user_weight      -- Scalar weight assigned to the user distribution
    5) eval_pts         -- 1D NumPy array containing locations at which to evaluate CDF

    NOTE: sum of delta_weights and user_weight must be unity!
'''
def muwe_cdf( delta_pts, delta_weights, user_dist, user_weight, eval_pts ):

    out_cdf = np.zeros_like(eval_pts)

    # Evaluate contribution of user distribution to MUWE CDF
    if user_weight > 0:
        out_cdf += user_dist.cdf( eval_pts ) * user_weight
    # --- End of user distribution contribution

    # Delta distribution CDF contribution
    if np.sum(delta_weights) > 0:

        # Normalizing weights
        delta_weights_normalized = delta_weights / np.sum(delta_weights)

        # Evaluate weighted empirical CDF
        delta_cdf = weighted_empirical_cdf( 
            delta_pts, delta_weights_normalized, eval_pts
        ) 

        # Contribute to output CDF
        out_cdf += delta_cdf * (1.-user_weight)

    # --- End of delta distribution contribution

    return out_cdf


# Sanity check for MUWE CDF function
def muwe_cdf_SANITY_CHECK():

    import matplotlib.pyplot as plt
    from scipy.stats import norm

    # Setting up weighted emipirical distribution
    delta_pts = np.arange(3) -1.
    delta_weights = np.array((0.25,0.5, 0.25), dtype='f8') * 0.5
    
    # Setting up user distribution
    user_dist = norm(0,1)
    user_weight = 0.5

    # Setting up evaluation locations
    eval_locs = np.zeros(20)
    eval_locs[:12] = np.linspace(-0.5,2.5, 12 )
    eval_locs[12:] = [-1,-1,0,0,0,0,1,1]
    eval_locs = np.sort(eval_locs)

    # Evaluate on dense locations
    dense_eval_locs = np.linspace( -3,3, 1001 )


    # Evaluate and plot CDF!
    cdf = muwe_cdf(delta_pts, delta_weights, user_dist, user_weight, eval_locs)
    cdf_dense = muwe_cdf(delta_pts, delta_weights, user_dist, user_weight, dense_eval_locs)
    plt.plot( dense_eval_locs, cdf_dense, '-r', label='Dense CDF evaluation', zorder=0)
    plt.scatter(eval_locs, cdf, s=30, label='Sparse CDF evaluation')
    plt.scatter( 
        delta_pts, np.cumsum( delta_weights) + norm.cdf(delta_pts)*user_weight, s = 30, 
        marker='x', c='r', label='True CDF at delta locations')
    plt.ylabel('CDF')
    plt.xlabel('x')
    plt.legend()
    plt.title('Sanity Checking muwe_cdf')
    plt.savefig('SANITY_CHECK_muwe_cdf.png')
    plt.close()

    return
    























'''
    FUNCTION TO EVALUATE CDF OF WEIGHTED EMPIRICAL DISTRIBUTION

    Inputs:
    1) delta_pts                -- Sorted UNIQUE locations of delta functions (1D NumPy array)
    2) delta_weights_normalized -- Weights assigned to each delta function (1D NumPy array)
    3) eval_pts                 -- 1D NumPy array containing locations at which to evaluate CDF

    NOTES: 
        ~ delta_pts must be sorted, and delta_weights corresponds to each delta point
        ~ sum of delta_weights should be less than or equal to unity
        ~ borrowed ideas from Jeffrey Anderson to counteract issue of having multiple 
          evaluation points that lie on the same delta point
        ~ assumes that repeated eval points lie on a delta point
'''
def weighted_empirical_cdf( delta_pts, delta_weights_normalized, eval_pts ):

    # Detect delta points that coincide with eval points
    vals, cnts = np.unique( eval_pts, return_counts=True)
    flag_delta = np.isin( delta_pts, vals )
    delta_cnts = np.zeros_like( delta_pts )
    for ipt, pt in enumerate(delta_pts):
        if flag_delta[ipt]:
            ind = np.where( vals == pt )[0][0]
            delta_cnts[ipt] = cnts[ind]
    # --- End of detection
    
    # Calculate CDF using an eager compilation of the calculation loop
    out_cdf = weighted_empirical_cdf_loop_njit_accel( delta_pts, delta_weights_normalized, eval_pts, delta_cnts )
    
    return out_cdf



# Accelerating loop used in weighted_empirical_cdf_evaluator
@njit( nb_f64[:]( nb_f64[:], nb_f64[:], nb_f64[:], nb_f64[:] ) )
def weighted_empirical_cdf_loop_njit_accel( delta_pts, delta_weights_normalized, eval_pts, delta_cnts ):

    # Init useful counter
    delta_counter = np.zeros_like( delta_cnts )

    # Init output array
    out_cdf = np.empty_like( eval_pts)

    # Loop!
    for ipt, pt in enumerate( eval_pts ):

        # Identify delta locations that this point is to the right of
        flag_right = pt > delta_pts

        # Evaluate CDF based on delta points to the left of pt
        out_cdf[ipt] = np.sum( delta_weights_normalized[flag_right] )

        # Special handling if point lies on delta function
        if pt in delta_pts:
            
            # Index corresponding to delta_cnts
            ind = np.where( delta_pts == pt )[0][0]

            # Increment CDF if eval point lies on delta point
            delta_counter[ind] += 1
            out_cdf[ipt] += (
                delta_weights_normalized[ind]
                * (delta_counter[ind] / delta_cnts[ind])
            )
        # --- End of special treatment
    # --- End of loop over all eval points

    return out_cdf



# Sanity checker
def weighted_empirical_cdf_SANITY_CHECK():

    import matplotlib.pyplot as plt

    # Setting up weighted distribution
    delta_pts = np.arange(3)*1.
    delta_weights_normalized = np.array((0.25,0.5, 0.25), dtype='f8')

    # Setting up evaluation locations
    eval_locs = np.zeros(20)
    eval_locs[:14] = np.linspace(-0.5,2.5, 14 )
    eval_locs[14:] = [0,0,1,1,1,2]
    eval_locs = np.sort(eval_locs)

    # Evaluate on dense locations
    dense_eval_locs = np.linspace( -0.7, 2.7, 1001 )


    # Evaluate and plot CDF!
    cdf = weighted_empirical_cdf(delta_pts, delta_weights_normalized, eval_locs)
    cdf_dense = weighted_empirical_cdf(delta_pts, delta_weights_normalized, dense_eval_locs)
    plt.plot( dense_eval_locs, cdf_dense, '-r', label='Dense CDF evaluation', zorder=0)
    plt.scatter(eval_locs, cdf, s=30, label='Sparse CDF evaluation')
    plt.scatter( delta_pts, np.cumsum( delta_weights_normalized), s = 30, marker='x', c='r', label='True CDF at delta locations')
    plt.legend()
    plt.ylabel('CDF')
    plt.xlabel('x')
    plt.title('Sanity Checking weighted_empirical_cdf')
    plt.savefig('SANITY_CHECK_weighted_empirical_cdf.png')
    plt.close()
    
    return





































'''
    FUNCTION TO EVALUATE QUANTILE FUNCTION* OF MUWE DISTRIBUTION

    *Also known as percent-point function (PPF) and inverse CDF

    Inputs:
    1) delta_pts        -- Locations of delta functions (a sorted 1D NumPy array)
    2) delta_weights    -- Weights assigned to each delta function (1D NumPy array)
    3) user_dist        -- Fitted user-specified SciPy-like distribution instance
    4) user_weight      -- Scalar weight assigned to the user distribution
    5) eval_pctls       -- 1D NumPy array containing quantiles at which to evaluate 
                           quantile function
'''
def muwe_ppf( delta_pts, delta_weights, user_dist, user_weight, eval_pctls ):

    # Evaluate CDF upper and lower bounds corresponding to delta points
    user_cdf_at_delta_pts = user_dist.cdf( delta_pts ) * user_weight
    delta_upper_sum = np.cumsum( delta_weights )
    delta_cdf_upper = user_cdf_at_delta_pts + delta_upper_sum
    delta_cdf_lower = user_cdf_at_delta_pts
    delta_cdf_lower[1:] += delta_upper_sum[:-1]

    # Init array to hold output values
    out_vals = np.zeros_like( eval_pctls )

    # Loop over every evaluation quantile   
    for ipctl, pctl in enumerate( eval_pctls ):

        # Treatment for quantiles that lie within zone of delta jumps
        mask = ( delta_cdf_lower <= pctl ) * ( pctl <= delta_cdf_upper )
        flag_within_delta_jumps = np.sum(mask) > 0
        if flag_within_delta_jumps:
            ind = np.where(mask)[0][0]
            out_vals[ipctl] = delta_pts[ind]
        # -- End of treatment for quantiles that lie within delta jumps

        # Treatment for values outside of delta jumps
        if not flag_within_delta_jumps:
            # Determine contribution of weighted empirical CDF to this 
            # quantile value
            flag_larger_than_delta_upper = (pctl > delta_cdf_upper)
            pctl_delta_contribution = np.sum( delta_weights[flag_larger_than_delta_upper] )

            # Map percentile to normalized user distribution
            pctl_user_contribution = (pctl - pctl_delta_contribution)/user_weight

            # Invert user distribution
            out_vals[ipctl] = user_dist.ppf( pctl_user_contribution)
        # --- End of treatment for values outside of delta jumps
    # --- End of loop over quantiles

    return out_vals
        
        

# Sanity checking muwe ppf
def muwe_ppf_SANITY_CHECK():

    import matplotlib.pyplot as plt
    from scipy.stats import norm

    # Setting up weighted emipirical distribution
    delta_pts = np.arange(3) -1.
    delta_weights = np.array((0.25,0.5, 0.25), dtype='f8') * 0.5
    
    # Setting up user distribution
    user_dist = norm(0,1)
    user_weight = 0.5

    # Evaluate CDF on dense locations
    dense_eval_locs = np.linspace( -3,3, 1001 )

    # Test evaluations of muwe ppf
    pctl_vals = np.arange(10)/10. + 0.05
    ppf_vals = muwe_ppf( delta_pts, delta_weights, user_dist, user_weight, pctl_vals )

    # Evaluate MUWE CDF densely!
    cdf_dense = muwe_cdf(delta_pts, delta_weights, user_dist, user_weight, dense_eval_locs)

    # Plot CDF and ppf outcomes
    plt.plot( cdf_dense, dense_eval_locs, '-r', label='Theoretical PPF', zorder=0)
    plt.scatter( pctl_vals,ppf_vals, s=30, label='MUWE PPFs')
    plt.xlabel('CDF')
    plt.ylabel('x')

    plt.legend()
    plt.title('Sanity Checking muwe_ppf')
    plt.savefig('SANITY_CHECK_muwe_ppf.png')
    plt.close()

    return
        





















'''
    ACCELERATED FUNCTION TO EVALUATE QUANTILE FUNCTION* OF MUWE DISTRIBUTION

    *Also known as percent-point function (PPF) and inverse CDF

    Inputs:
    1) delta_pts        -- Locations of delta functions (a sorted 1D NumPy array)
    2) delta_weights    -- Weights assigned to each delta function (1D NumPy array)
    3) user_dist        -- Fitted user-specified SciPy-like distribution instance
    4) user_weight      -- Scalar weight assigned to the user distribution
    5) eval_pctls       -- 1D NumPy array containing quantiles at which to evaluate 
                           quantile function
''' 
def muwe_ppf_fast( delta_pts, delta_weights, user_dist, user_weight, eval_pctls ):

    # Evaluate CDF upper and lower bounds corresponding to delta points
    user_cdf_at_delta_pts = user_dist.cdf( delta_pts ) * user_weight
    delta_upper_sum = np.cumsum( delta_weights )
    delta_cdf_upper = user_cdf_at_delta_pts + delta_upper_sum
    delta_cdf_lower = user_cdf_at_delta_pts
    delta_cdf_lower[1:] += delta_upper_sum[:-1]

    # Apply JIT-accelerated loop calculation
    user_weight_f64 = np.float64(user_weight)
    out2d = muwe_ppf_fast_loop_njit_accel( 
        delta_pts, delta_weights, user_weight_f64, eval_pctls, delta_cdf_lower, delta_cdf_upper
        )

    # Parsing outcome of the JIT-accelerated loop calculations
    out_vals = out2d[:,0]
    flag_user_ppf_eval = np.isnan(out_vals)
    out_vals[flag_user_ppf_eval] = user_dist.ppf(out2d[:,1][flag_user_ppf_eval])
    
    return out_vals




# Accelerated loop calculation
@njit(  nb_f64[:,:]( nb_f64[:], nb_f64[:], nb_f64, nb_f64[:], nb_f64[:], nb_f64[:] ) )
def muwe_ppf_fast_loop_njit_accel( delta_pts, delta_weights, user_weight, eval_pctls, delta_cdf_lower, delta_cdf_upper ):

    # Init 2D array to hold output values
    # Column 0: PPF'ed values for quantiles within delta jump zone
    # Column 1: transformed quantile that can be used for user distribution ppf
    out_arr2d = np.zeros( (len(eval_pctls), 2) ) + np.nan
    
    # Loop over every evaluation quantile   
    for ipctl, pctl in enumerate( eval_pctls ):

        # Treatment for quantiles that lie within zone of delta jumps
        mask = ( delta_cdf_lower <= pctl ) * ( pctl <= delta_cdf_upper )
        flag_within_delta_jumps = np.sum(mask) > 0
        if flag_within_delta_jumps:
            ind = np.where(mask)[0][0]
            out_arr2d[ipctl,0] = delta_pts[ind]
        # -- End of treatment for quantiles that lie within delta jumps

        # Treatment for values outside of delta jumps
        if not flag_within_delta_jumps:
            # Determine contribution of weighted empirical CDF to this 
            # quantile value
            flag_larger_than_delta_upper = (pctl > delta_cdf_upper)
            pctl_delta_contribution = np.sum( delta_weights[flag_larger_than_delta_upper] )

            # Map percentile to normalized user distribution
            out_arr2d[ipctl,1] = (pctl - pctl_delta_contribution)/user_weight

        # --- End of treatment for values outside of delta jumps
    # --- End of loop over quantiles

    return out_arr2d


# Sanity checking fast muwe ppf
def muwe_ppf_fast_SANITY_CHECK():

    import matplotlib.pyplot as plt
    from scipy.stats import norm

    # Setting up weighted emipirical distribution
    delta_pts = np.arange(3) -1.
    delta_weights = np.array((0.25,0.5, 0.25), dtype='f8') * 0.5
    
    # Setting up user distribution
    user_dist = norm(0,1)
    user_weight = 0.5

    # Evaluate CDF on dense locations
    dense_eval_locs = np.linspace( -3,3, 1001 )

    # Test evaluations of muwe ppf
    pctl_vals = np.arange(10)/10. + 0.05
    ppf_vals = muwe_ppf_fast( delta_pts, delta_weights, user_dist, user_weight, pctl_vals )

    # Evaluate MUWE CDF densely!
    cdf_dense = muwe_cdf(delta_pts, delta_weights, user_dist, user_weight, dense_eval_locs)

    # Plot CDF and ppf outcomes
    plt.plot( cdf_dense, dense_eval_locs, '-r', label='Theoretical PPF', zorder=0)
    plt.scatter( pctl_vals,ppf_vals, s=30, label='MUWE Fast PPFs')
    plt.xlabel('CDF')
    plt.ylabel('x')

    plt.legend()
    plt.title('Sanity Checking muwe_ppf_fast')
    plt.savefig('SANITY_CHECK_muwe_ppf_fast.png')
    plt.close()

    return



















'''
    SANITY CHECKS
'''
if __name__ == '__main__':
    weighted_empirical_cdf_SANITY_CHECK()
    muwe_cdf_SANITY_CHECK()
    muwe_ppf_SANITY_CHECK()
    muwe_ppf_fast_SANITY_CHECK()