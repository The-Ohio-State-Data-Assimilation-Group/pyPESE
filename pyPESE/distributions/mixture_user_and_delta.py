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
from numba import njit
from numba import float64 as nb_f64
from numba import float32 as nb_f32
from numba import int64 as nb_i64
from numba.types import Tuple as nb_tuple



'''
    Class defining the delta+user mixture distribution

    Note: This function is initialized using other pyPESE distribution classes.
'''
class mixture_user_and_delta: 

    name = 'generic delta-user mixture distribution'


    # Initialize 
    # Note: dist_class must be a class, not an instance!!!
    def __init__( self, ens_values1d, dist_class ):

        # Error-handling variables
        self.err_flag = False
        self.err_msg = ''

        # Error check: dist_class is an instance, not a class
        if isinstance( dist_class ):
            self.err_flag = True
            self.err_msg += '\nERROR (pyPESE . distributions . mixture_user_and_delta . mixture_user_and_delta . __init__):\n'
            self.err_msg += 'dist_class used to init mixture_user_and_delta is an instance!\n'
            self.err_msg += 'Name of the inputted dist_class instance: %s\n' % dist_class.name
        # --- End of class checking

        # Sort ensemble values
        sorted_ens_vals = np.sort( ens_values1d )
        ens_size = len( sorted_ens_vals )

        # Detect unique values 
        vals, cnt = np.unique( sorted_ens_vals, return_counts = True)
        uniq_vals = vals[cnt == 1]

        # Detect degenerate values & their frequencies
        degen_vals = vals[cnt > 1]
        degen_freq = cnt[cnt>1]

        # Generate delta distribution information
        if len( uniq_vals ) < ens_size:
            self.delta_dist_flag = True
            self.delta_dist_vals = np.sort( np.repeat( vals[cnt>1], cnt[cnt>1] ) )
            self.delta_dist_weight = len(self.degen_dist_vals) / ens_size 
        else:
            self.delta_dist_flag = False
            self.delta_dist_weight = 0.

        # Fit user-specified distribution to unique values
        params = dist_class.fit( uniq_vals )

        # Store user-specified distribution instance
        self.user_dist_instance = dist_class( *params )
        self.user_dist_weight = len( uniq_vals ) / ens_size

        # Check: Do weights sum to unity?
        checksum = self.degen_weight + self.user_dist_weight
        if ( np.abs( checksum - 1) > 1e-6 ) :
            self.err_flag = True
            self.err_msg += '\nERROR (pyPESE . distributions . mixture_user_and_delta . mixture_user_and_delta . __init__): \n'
            self.err_msg += 'Sum of weights (%f) is not unity!\n' % checksum
        # --- End of weight sum check

        return
    # --- End of definition for mixture_user_and_delta class

    

    # Function to evaluate CDF of mixture_user_and_delta distribution
    def cdf( self, eval_pts ):

        # Evaluate CDF of user component
        cdf_user = self.user_dist_instance.cdf( eval_pts )

        # Evaluate CDF of delta component (should component exist)
        if self.delta_dist_flag:
            cdf_delta = np.searchsorted( 
                self.delta_dist_vals, eval_pts, side='right'
            ) / len(self.delta_dist_vals)
        else:
            cdf_delta = 0.

        # Mix the two CDFs
        cdf_mix = ( 
            cdf_user * self.user_dist_weight 
            + cdf_delta * self.delta_dist_weight
        )

        # Return the mixed CDF
        return cdf_mix
    # --- End of function to evaluate the CDF



    # Public-facing function to evaluate inverse CDF (aka, percent point function)
    def ppf( self, eval_cdf ):
        
        # PPF calculation is fast if delta component does not exist.
        if not self.delta_dist_flag:
            ppf_output = self.user_dist_instance.ppf( eval_cdf )
        # --- End of calculation for pure user distribution

        # PPF calculation is not trivial for mixed distribution
        if self.delta_dist_flag:
            ppf_output = self.mixed_user_delta_ppf( eval_cdf )

