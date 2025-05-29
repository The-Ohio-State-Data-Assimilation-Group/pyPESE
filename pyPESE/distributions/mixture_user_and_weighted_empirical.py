'''
    MIXTURE DIRAC DELTA DISTRIBUTION
    =================================================================================
    In an ensemble of simulations, there can sometimes be repeated (i.e.) degenerate values.
    For example, multiple WRF members' QCLOUD value at the same location can be zero. These
    degenerate values interfere with probit transforms. We will treat degenerate values as 
    being drawn from a dirac-delta distribution, and the non-degenerate values as being drawn
    from another distribution. In other words, we use a mixture of two distributions when 
    there are degenerate values: (1) delta distributions, and (2) the user-specified distribution. 
    This mixture distribution is defined below

    \begin{equation}
        p(x) = (1/N) * { \sum^{N_d}_{j=1} (delta(x-x_j)) }
                + ( 1 - N_d/N ) * {p_user(x)}
    \end{equation}

    When there are no degenerate values, the mixture distribution turns into the user-specified 
    distribution. As such, the delta+user mixture distribution should be used by default in PESE-GC.

    ChatGPT transcript used to help with coding: https://chatgpt.com/share/68334900-9414-8008-8a92-ce219ca16c2e
'''


import numpy as np
from copy import deepcopy
# from numba import njit
# from numba import float64 as nb_f64
# from numba import float32 as nb_f32
# from numba import int64 as nb_i64
# from numba.types import Tuple as nb_tuple




# '''
#     Class defining the delta+user mixture distribution

#     Note: This function is initialized using other pyPESE distribution classes.
# '''
# class mixture_user_and_delta: 

#     name = 'generic delta-user mixture distribution'


#     # Initialize 
#     # Note: user_dist_class must be a class, not an instance!!!
#     def __init__( self, ens_values1d, user_dist_class ):

#         # Error-handling variables
#         self.err_flag = False
#         self.err_msg = ''

#         # Error check: user_dist_class is an instance, not a class
#         if isinstance( user_dist_class ):
#             self.err_flag = True
#             self.err_msg += '\nERROR (pyPESE . distributions . mixture_user_and_delta . mixture_user_and_delta . __init__):\n'
#             self.err_msg += 'user_dist_class used to init mixture_user_and_delta is an instance!\n'
#             self.err_msg += 'Name of the inputted user_dist_class instance: %s\n' % user_dist_class.name
#         # --- End of class checking

#         # Sort ensemble values
#         sorted_ens_vals = np.sort( ens_values1d )
#         ens_size = len( sorted_ens_vals )
#         self.ens_vals = sorted_ens_vals

#         # Detect unique values 
#         vals, cnt = np.unique( sorted_ens_vals, return_counts = True)
#         uniq_vals = vals[cnt == 1]

#         # Detect degenerate values & their frequencies
#         degen_vals = vals[cnt > 1]
#         degen_freq = cnt[cnt>1]

#         # -----------------------------------------------------------------
#         # Fit & init user-specified distribution
#         # -----------------------------------------------------------------

#         # Fit user-specified distribution to unique values
#         params = user_dist_class.fit( uniq_vals )

#         # Store user-specified distribution instance
#         self.user_dist_instance = user_dist_class( *params )
#         self.user_dist_weight = len( uniq_vals ) / ens_size


#         # -----------------------------------------------------------------
#         # Prepare delta distribution information
#         # -----------------------------------------------------------------
#         self.delta_dist_weight = ( ens_size - len( uniq_vals ) ) / ens_size 

#         # Only prep info if delta distribution is activate
#         if self.delta_dist_weight > 0:
#             self.delta_dist_vals = degen_vals
#             self.delta_dist_cnts = degen_freq

#             # Eval CDF at ensemble values
#             self.delta_dist_ens_cdf = np.zeros_like( sorted_ens_vals )
#             self.num_degen_vals = np.sum( degen_freq )
#             counter = 0
#             for imem in range( ens_size ):
#                 # Increment CDF for degen values
#                 if ( self.ens_vals[imem] in self.delta_dist_vals ):
#                     counter += 1
#                 # Eval CDF
#                 self.delta_dist_ens_cdf[imem] = counter/self.num_degen_vals
#             # --- End of CDF evaluation loop
#         # --- End of delta distribution information prep


#         # Check: Do weights sum to unity?
#         checksum = self.delta_dist_weight + self.user_dist_weight
#         if ( np.abs( checksum - 1) > 1e-6 ) :
#             self.err_flag = True
#             self.err_msg += '\nERROR (pyPESE . distributions . mixture_user_and_delta . mixture_user_and_delta . __init__): \n'
#             self.err_msg += 'Sum of weights (%f) is not unity!\n' % checksum
#         # --- End of weight sum check


#         return
#     # --- End of definition for mixture_user_and_delta class


    

#     # Function to evaluate CDF of mixture_user_and_delta distribution
#     def cdf( self, eval_pts ):

#         # Evaluate CDF of user component
#         cdf_user = self.user_dist_instance.cdf( eval_pts )

#         # Evaluate CDF of delta component (should component exist)
#         if self.delta_dist_flag:
#             cdf_delta = mud_delta_cdf( self, eval_pts )
#         else:
#             cdf_delta = np.ones_like( eval_pts )
#         # --- End of delta distribution eval

#         # Mix the two CDFs
#         cdf_mix = ( 
#             cdf_user * self.user_dist_weight 
#             + cdf_delta * self.delta_dist_weight
#         )

#         # Return the mixed CDF
#         return cdf_mix
#     # --- End of function to evaluate the CDF



#     # Public-facing function to evaluate inverse CDF (aka, percent point function)
#     def ppf( self, eval_cdf ):
        
#         # PPF calculation is simple if delta component does not exist.
#         if self.delta_dist_weight == 0 :
#             ppf_output = self.user_dist_instance.ppf( eval_cdf )
#         # --- End of calculation for pure user distribution

#         # PPF calculation is not trivial for mixed distribution
#         if self.delta_dist_weight > 0:
#             ppf_output = mud_ppf( eval_cdf )


# # ------------ End of definition for mixture_and_user_delta distribution SciPy-like class



























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


# Sanity check for muwe CDF function























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

    # Init output array
    out_cdf = np.empty_like( eval_pts)

    # Detect delta points that coincide with eval points
    vals, cnts = np.unique( eval_pts, return_counts=True)
    flag_delta = np.isin( delta_pts, vals )
    delta_cnts = np.zeros_like( delta_pts )
    for ipt, pt in enumerate(delta_pts):
        if flag_delta[ipt]:
            ind = np.where( vals == pt )[0][0]
            delta_cnts[ipt] = cnts[ind]
    # --- End of detection
    

    # Looping over every evaluation point
    delta_counter = np.zeros_like( delta_cnts )
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
    plt.title('Sanity Checking weighted_empirical_cdf')
    plt.savefig('SANITY_CHECK_weighted_empirical_cdf.png')
    















'''
    SANITY CHECKS
'''
if __name__ == '__main__':
    weighted_empirical_cdf_SANITY_CHECK()