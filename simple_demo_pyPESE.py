'''
    SCRIPT TO DEMONSTRATE PROBIT-SPACE ENSEMBLE SIZE EXPANSION PYTHON PACKAGE
    =========================================================================

    Type of pyPESE algorithm demonstrated: 
        PESE for Gaussian Copulas (PESE-GC)

'''

# Standard packages
import numpy as np
from matplotlib import use as mpl_use
mpl_use('agg')
import matplotlib.pyplot as plt
from scipy.stats import skewnorm, gamma, norm

# Import PESE-GC function from pyPESE
from pyPESE.pese_gc import pese_gc

# Import custom-made non-parametric distribution class (in pyPESE package)
from pyPESE.distributions.bounded_boxcar_rank_histogram import bounded_boxcar_rank_histogram as bbrh



'''
    Draw bivariate samples representing original ensemble
'''

# Number of original ensemble members
ens_size_original = 100

# Generate samples from Gaussian copula
probit_cov = np.matrix([ [1,0.9], [0.9,1]])
sqrt_probit_cov = np.matrix( np.linalg.cholesky( probit_cov ) )
original_probits = np.matrix( sqrt_probit_cov ) * np.matrix( np.random.normal(size=(2,ens_size_original)))
original_probits = np.array( original_probits)

# Map from probit space to quantile space
original_ens = norm.cdf( original_probits )

# Map from quantile space to native space
original_ens[0,:] = gamma(3).ppf( original_ens[0,:] )
original_ens[1,:] = skewnorm(-10).ppf( original_ens[1,:])





'''
    TWO EXAMPLES OF APPLYING PESE-GC
'''

# Number of virtual members to create
ens_size_virtual  = 1000

# Example 1: Employ user-informed distributions
# ---------------------------------------------
# Init list of distribution classes
list_dist_classes = [gamma, skewnorm]

# Init list of extra arguments needed for ensemble preprocessing
list_extra_args = ([
    {'min bound':    0, 'max bound': 1e9 },  # Arguments for the first variable
    {'min bound': -1e9, 'max bound': 1e9 }   # Arguments for the second variable
])
virtual_ensemble1 = pese_gc( original_ens, list_dist_classes, list_extra_args, ens_size_virtual, rng_seed=0 )


# Example 2: Employ non-parametric distribution (bounded rank histogram)
# ----------------------------------------------------------------------
list_dist_classes = [bbrh, bbrh]
list_extra_args = ([
    {'min bound':    0, 'max bound': original_ens[0,:].max()+1 },  # Arguments for the first variable
    {'min bound':   original_ens[1,:].min() -1, 'max bound': 0   }   # Arguments for the second variable
])
virtual_ensemble2 = pese_gc( original_ens, list_dist_classes, list_extra_args, ens_size_virtual, rng_seed=0 )

print( 'Mean and Variance of original ensemble')
print( np.mean(original_ens, axis=1), np.var(original_ens, axis=1, ddof=1))
print('Mean and Variance of virtual ensemble')
print( np.mean( virtual_ensemble2, axis=1 ), np.var(virtual_ensemble2, axis=1, ddof=1) )




# Plot both examples
# -------------------
fig, axs = plt.subplots( ncols=1, nrows=2, figsize=(6,6) )

# Plot out example 1
axs[0].scatter( original_ens[0,:], original_ens[1,:], c='dodgerblue', 
             marker='o', label = 'Original Ensemble', s=50)
axs[0].scatter( virtual_ensemble1[0,:], virtual_ensemble1[1,:], c='r',
             marker='o', label = 'Virtual Ensemble', s = 1)
axs[0].set_title('Example 1: User-informed distributions used', loc='left')
axs[0].legend()

# Plot out example 2
axs[1].scatter( original_ens[0,:], original_ens[1,:], c='dodgerblue', 
             marker='o', label = 'Original Ensemble', s=50)
axs[1].scatter( virtual_ensemble2[0,:], virtual_ensemble2[1,:], c='r',
             marker='o', label = 'Virtual Ensemble', s = 1)
axs[1].set_title('Example 2: Non-parametric distributions used', loc='left')
axs[1].legend()

# Make plots pretty
for i in range(2):
    axs[i].set_xlabel('Model variable 1')
    axs[i].set_ylabel('Model variable 2')
    axs[i].set_xlim([0,12])


fig.subplots_adjust( hspace=0.4)


plt.savefig('simple_demo.png')
