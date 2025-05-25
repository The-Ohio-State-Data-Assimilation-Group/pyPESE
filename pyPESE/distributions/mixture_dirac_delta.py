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
    def __init__( self, ens_values, dist_class ):

        # Error-handling variables
        err_flag = False
        err_msg = ''

        # Error check: dist_class is an instance, not a class
        if isinstance( dist_class ):
            err_flag = True
            err_msg = 'ERROR: dist_class used to init generic_delta_and_user_mixture is an instance!'
            err_msg = '       '+string( dist_class )
            return err_flag, err_msg

        # Sort ensemble values
        sorted_ens_vals = np.sort( ens_values )

        # Detect unique values
        vals, cnt = np.unique( sorted_ens_vals, return_counts = True)
        uniq_val = vals[cnt == 1]
        degen_vals = vals[cnt > 1]

        # Detect uniqu

        self.cdf_locs = cdf_locs
        self.cdf_vals = cdf_vals
        return

    


