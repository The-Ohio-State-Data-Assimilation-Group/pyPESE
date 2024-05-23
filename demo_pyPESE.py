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

# Import custom-made non-parametric distribution (in pyPESE package)
from pyPESE.distributions.bounded_rank_histogram import bounded_rank_histogram



'''
    Draw bivariate samples representing original ensemble
'''

ens_size_original = 100
ens_size_virtual  = 1000

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

# Example 1: Employ user-informed distributions
# ---------------------------------------------
# Init list of distribution classes
list_dist_classes = [gamma, skewnorm]
virtual_ensemble1 = pese_gc( original_ens, list_dist_classes, ens_size_virtual, rng_seed=0 )


# Example 2: Employ non-parametric distribution (bounded rank histogram)
# ----------------------------------------------------------------------
list_dist_classes = [bounded_rank_histogram, bounded_rank_histogram]
virtual_ensemble2 = pese_gc( original_ens, list_dist_classes, ens_size_virtual, rng_seed=0 )


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


fig.subplots_adjust( hspace=0.4)


plt.savefig('demo.png')
