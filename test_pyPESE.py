'''
    QUICK AND DIRTY SCRIPT TO TEST OUT PYPESE FUNCTIONS
'''

# Standard packages
import numpy as np
from matplotlib import use as mpl_use
mpl_use('agg')
import matplotlib.pyplot as plt

# Import pyPESE
import pyPESE



# Generate random samples representing original ensemble
def simple_bi_gaussian_draws( num_dims, ens_size ):

    # Hard-coded bi-gaussian settings
    weights = np.ones(2) * 0.5
    sigmas  = np.ones(2)
    means   = np.ones(2)
    means[0] = 0.
    means[1] = 2.

    # Now generate ensemble
    ensemble = np.zeros( (num_dims, ens_size) )
    for imem in range( ens_size ):
        coin_toss = np.random.uniform() < weights[0]

        # If heads, then draw from kernel 0
        if coin_toss:
            kern_id = 0 
        else:
            kern_id = 1
        
        # Draw member
        ensemble[:,imem] = np.random.normal( 
            loc = means[kern_id], scale = sigmas[kern_id],
            size = num_dims
         )

    # --- End of loop over ensemble members

    return ensemble










    



