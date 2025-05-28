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
    # Note: user_dist_class must be a class, not an instance!!!
    def __init__( self, ens_values1d, user_dist_class ):

        # Error-handling variables
        self.err_flag = False
        self.err_msg = ''

        # Error check: user_dist_class is an instance, not a class
        if isinstance( user_dist_class ):
            self.err_flag = True
            self.err_msg += '\nERROR (pyPESE . distributions . mixture_user_and_delta . mixture_user_and_delta . __init__):\n'
            self.err_msg += 'user_dist_class used to init mixture_user_and_delta is an instance!\n'
            self.err_msg += 'Name of the inputted user_dist_class instance: %s\n' % user_dist_class.name
        # --- End of class checking

        # Sort ensemble values
        sorted_ens_vals = np.sort( ens_values1d )
        ens_size = len( sorted_ens_vals )
        self.ens_vals = sorted_ens_vals

        # Detect unique values 
        vals, cnt = np.unique( sorted_ens_vals, return_counts = True)
        uniq_vals = vals[cnt == 1]

        # Detect degenerate values & their frequencies
        degen_vals = vals[cnt > 1]
        degen_freq = cnt[cnt>1]

        # -----------------------------------------------------------------
        # Fit & init user-specified distribution
        # -----------------------------------------------------------------

        # Fit user-specified distribution to unique values
        params = user_dist_class.fit( uniq_vals )

        # Store user-specified distribution instance
        self.user_dist_instance = user_dist_class( *params )
        self.user_dist_weight = len( uniq_vals ) / ens_size


        # -----------------------------------------------------------------
        # Prepare delta distribution information
        # -----------------------------------------------------------------
        self.delta_dist_weight = ( ens_size - len( uniq_vals ) ) / ens_size 

        # Only prep info if delta distribution is activate
        if self.delta_dist_weight > 0:
            self.delta_dist_vals = degen_vals
            self.delta_dist_cnts = degen_freq

            # Eval CDF at ensemble values
            self.delta_dist_ens_cdf = np.zeros_like( sorted_ens_vals )
            self.num_degen_vals = np.sum( degen_freq )
            counter = 0
            for imem in range( ens_size ):
                # Increment CDF for degen values
                if ( self.ens_vals[imem] in self.delta_dist_vals ):
                    counter += 1
                # Eval CDF
                self.delta_dist_ens_cdf[imem] = counter/self.num_degen_vals
            # --- End of CDF evaluation loop
        # --- End of delta distribution information prep


        # Check: Do weights sum to unity?
        checksum = self.delta_dist_weight + self.user_dist_weight
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
            cdf_delta = mud_delta_cdf( self, eval_pts )
        else:
            cdf_delta = np.ones_like( eval_pts )
        # --- End of delta distribution eval

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
        
        # PPF calculation is simple if delta component does not exist.
        if self.delta_dist_weight == 0 :
            ppf_output = self.user_dist_instance.ppf( eval_cdf )
        # --- End of calculation for pure user distribution

        # PPF calculation is not trivial for mixed distribution
        if self.delta_dist_weight > 0:
            ppf_output = mud_ppf( eval_cdf )


# ------------ End of definition for mixture_and_user_delta distribution SciPy-like class






















'''
    FUNCTION TO EVALUATE DELTA DISTRIBUTION CDF OF MIXTURE USER AND DELTA DISTRIBUTION

    Inputs:
    1) mud_dist_inst -- Instance of the mixture user and delta distribution class
    2) eval_pts -- 1D NumPy array containing locations at which to evaluate CDF
'''
def mud_delta_cdf( mud_dist_inst, eval_pts ):

    # Is the eval points the same as the ensemble?
    sorted_eval_pts = np.sort( eval_pts )

    # Checking whether the ensemble values and eval values are the same
    flag_ens_eval_same = False
    if len( sorted_eval_pts ) == len(mud_dist_inst.ens_vals):
        flag_ens_eval_same = (
            int( np.sum( mud_dist_inst.ens_vals == sorted_eval_pts ) )
            == len(mud_dist_inst.ens_vals)
        )
    # --- End of checking whether ensemble values and eval values are the same

    # CDF for ensemble values
    if flag_ens_eval_same:
        sort_inds = np.argsort( eval_pts )
        out_cdf = np.empty_like( eval_pts )
        out_cdf[sort_inds] = mud_dist_inst.delta_dist_ens_cdf[:]
    # --- End of CDF evaluation at ensemble values

    # CDF for non-ensemble values
    if not flag_ens_eval_same:
        out_cdf = np.searchsorted( 
            mud_dist_inst.ens_vals,
            eval_pts, side = 'right'
        )
        out_cdf /= mud_dist_inst.num_degen_vals
    # --- End of CDF evaluation for eval_pts that are not the ensemble.

    return out_cdf









'''
    FUNCTION TO EVALUATE PPF OF MIXED USER-DELTA DISTRIBUTION
'''
def mixed_user_delta_ppf( mud_dist_inst, eval_cdf ):

    # Determine CDF jumps at locations where the delta functions exist
    cdf_jump_dict = {}
    cdf_jump_dict['val'] = []
    cdf_jump_dict['cdf st'] = []
    cdf_jump_dict['cdf ed'] = []
    delta_cdf_offset = 0.
    for ival, val in enumerate(mud_dist_inst.delta_dist_vals):
        cdf_st = (
            mud_dist_inst.user_dist_instance.cdf(val) * mud_dist_inst.user_dist_weight
            + delta_cdf_offset
        )
        cdf_ed = (
            cdf_st + mud_dist_inst.delta_dist_cnts[ival] / len( mud_dist_inst.ens_vals)
        )
        delta_cdf_offset += cdf_ed - cdf_st
        cdf_jump_dict['val'].append( deepcopy(val) )
        cdf_jump_dict['cdf st'].append( deepcopy(cdf_st) )
        cdf_jump_dict['cdf ed'].append( deepcopy(cdf_ed) )
    # --- end of CDF jump determination.

    # Numpy-rize the dictionary
    cdf_jump_dict['val']        = np.array( cdf_jump_dict['val'] )
    cdf_jump_dict['cdf span']   = np.array( cdf_jump_dict['cdf span'] )

    # Output values
    out_vals = np.empty_like( eval_cdf )

    # Evaluate PPF!
    for icdf, cdf in enumerate(eval_cdf):

        # is CDF within one of the delta ranges?
        flag_continuum = True
        for ispan in range( len(cdf_jump_dict['val']) ):

            # Is cdf value within this delta range?
            if ( cdf_jump_dict['cdf st'][ispan] < cdf 
                    and cdf <= cdf_jump_dict['cdf ed'][ispan] ):
                flag_continuum = False
                out_vals[icdf] = cdf_jump_dict['val'][ispan]

                # no point in searching further
                break
            # --- End of comparing cdf against a cdf span
        # --- end of loop over all cdf spans
    
        # If CDF is not within continuum range, move onto next CDF value
        if not flag_continuum:
            continue

        # Calculation to handle CDF within continuum ranges
        
        

            



    return