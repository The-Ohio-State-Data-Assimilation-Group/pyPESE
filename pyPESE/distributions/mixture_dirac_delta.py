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
class generic_delta_and_user_mixture: 

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
            self.err_msg += '\nERROR (pyPESE.distributions.generic_delta_and_user_mixture.__init__):\n'
            self.err_msg += 'dist_class used to init generic_delta_and_user_mixture is an instance!\n'
            self.err_msg += 'Name of the inputted dist_class instance: %s\n' % dist_class.name
            return

        # Sort ensemble values
        sorted_ens_vals = np.sort( ens_values1d )
        ens_size = len( sorted_ens_vals )

        # Detect unique values 
        vals, cnt = np.unique( sorted_ens_vals, return_counts = True)
        uniq_vals = vals[cnt == 1]

        # Detect degenerate values & their frequencies
        self.degen_vals = vals[cnt > 1]
        self.degen_weights = cnt[cnt>1] / ens_size 

        # Fit user-specified distribution to unique values
        params = dist_class.fit( uniq_vals )

        # Store user-specified distribution instance
        self.user_dist_instance = dist_class( *params )
        self.user_dist_weight = len( uniq_vals ) / ens_size

        # Check: Do weights sum to unity?
        checksum = np.sum( self.degen_weights ) + self.user_dist_weight
        if ( np.abs( checksum - 1) > 1e-6 ) :
            self.err_flag = True
            self.err_msg += '\nERROR (pyPESE.distributions.generic_delta_and_user_mixture.__init__): \n'
            self.err_msg += 'Sum of weights (%f) is not unity!\n' % checksum
            return


        return

    


