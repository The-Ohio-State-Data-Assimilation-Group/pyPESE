'''
    LIBRARY OF UTILITY FUNCTIONS

'''



import numpy as np












'''
    Function to preprocess ensemble to handle out-of-bounds values and duplicate values

    Such values can cause issues with certain distributions, and issues with resampling.

    Mandatory arguments:
    --------------------
    1) input_ens1d
            1D NumPy array containing an ensemble of values for a forecast model variable

    Optional arguments
    ------------------
    1) min_bound
            User-specified scalar value indicating the left boundary of fcst pdf's support
    2) max_bound
            User-specified scalar value indicating the right boundary of fcst pdf's support

    Output:
    -------
    1) ens1d
            1D NumPy array containing preprocessed ensemble (NOT SORTED!!!)
    
'''
def preprocess_ens( input_ens1d, min_bound=-1e9, max_bound=1e9 ):

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

    # Special handling: what if the ensemble has entirely identical values?
    if ( len( uniq_vals ) == 1 ):

        if ( uniq_vals[0] != 0 ):
            smallest_uniq_interval = 1e-5
        else:
            smallest_uniq_interval = uniq_vals[0] * 1e-5

    # Serviceable case: ensemble has more than 1 unique values
    else:
        smallest_uniq_interval = (uniq_vals[1:] - uniq_vals[:-1]).min()
 
     # --- End of special handling.


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