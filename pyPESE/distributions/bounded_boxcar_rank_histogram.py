'''
    BOUNDED BOXCAR RANK HISTOGRAM (BBRH) DISTRIBUTION
    =================================================================================
    This is similar to Jeffrey Anderson's Gaussian-tailed Rank Histogram, except that
    I am assigning box-car tails outside of the data range.

    The BBRH distribution differs from the boxcar rank histogram (BRH) distribution in
    terms of:
    1) Termination location of the tails are fixed in BBRH whereas those of BRH are 
       computed. 
    2) BBRH's left and right tails can have different probability masses whereas those
       of BRH have the same probability mass.

    When it comes to moment matching, fitting the BBRH only requires solving a system of 
    THREE linear equations (efficient). In contrast, fitting the BRH requires solving a
    system of nonlinear polynomials (less efficient than the BBRH fitting process). These
    difference in fitting processes is due to the differences in the two distributions'
    treatment of tail probability masses and tail termination locations.
'''


import numpy as np
from copy import deepcopy
from scipy.optimize import root
from numba import njit
from numba import float64 as nb_f64
from numba.types import Tuple as nb_tuple





'''
    Function to preprocess ensemble to handle out-of-bounds values and duplicate values

    Mandatory arguments:
    --------------------
    1) input_ens1d
            1D NumPy array containing an ensemble of values for a forecast model variable
    2) min_bound
            User-specified scalar value indicating the left boundary of BBRH's support
    2) max_bound
            User-specified scalar value indicating the right boundary of BBRH's support

    Output:
    -------
    1) ens1d
            1D NumPy array containing preprocessed ensemble
    
'''
def BBRH_preprocess_ens( input_ens1d, min_bound, max_bound ):

    # Sort ensemble values
    # --------------------
    # This procedure also guards against unintentional object-passing effects because
    # an internal copy of the ensemble of values is generated.
    ens1d = np.sort( input_ens1d )
    ens_size = ens1d.shape[0]


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
    uniq_vals, uniq_inds, uniq_inv_inds, uniq_cnts = np.unique_all( ens1d )
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
    flags_oversmall = (ens1d < min_bound)
    num_oversmall   = np.sum(flags_oversmall)
    ens1d[flags_oversmall] = min_bound + ( np.arange(num_oversmall)+1 ) * offset_val

    # Handle overly large ensemble values by relocating them to the right boundary
    # of the support, MINUS some offset
    flags_overlarge = (ens1d > max_bound)
    num_overlarge   = np.sum( flags_overlarge )
    ens1d[flags_overlarge] = max_bound - ( np.arange(num_overlarge)+1 )[::-1] * offset_val
    
    return ens1d








