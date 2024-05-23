'''
    QUICK AND DIRTY SCRIPT TO TEST OUT PYPESE FUNCTIONS
'''

# Standard packages
import numpy as np
from matplotlib import use as mpl_use
mpl_use('agg')
import matplotlib.pyplot as plt
from scipy.stats import skewnorm, gamma, norm

# Import pyPESE's Gaussian resampling
from gaussian_resampling import fast_unlocalized_gaussian_resampling



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
original_ens[1,:] = skewnorm(-5).ppf( original_ens[1,:])





'''
    Now apply PESE-GC!
'''

# Step 1: Fit distributions
dist_list = [None, None]

shape1, loc1, scale1 = gamma.fit( original_ens[0,:] )
dist_list[0] = gamma( shape1, loc=loc1, scale=scale1)

shape2, loc2, scale2 = skewnorm.fit( original_ens[1,:] )
dist_list[1] = skewnorm( shape2, loc=loc2, scale=scale2 )
# print( shape2)


# Step 2: transform to probit space
fcst_probits = np.zeros( (2, ens_size_original) )
for i in range(2):
    fcst_probits[i,:] = dist_list[i].cdf( original_ens[i,:] )
    fcst_probits[i,:] = norm.ppf( fcst_probits[i,:] )
    fcst_probits[i,:] -= np.mean( fcst_probits[i] )
    fcst_probits[i,:] /= np.std( fcst_probits[i], ddof=1 )





# Step 3: fast gaussian resampling to cosntruct virtual probits
virt_probits = fast_unlocalized_gaussian_resampling(
    fcst_probits, ens_size_virtual, rng_seed = 0
)

print( np.cov( fcst_probits) )
print( np.cov( virt_probits) )


# Step 4: Invert the PPI transforms on the virtual probits
virtual_ensemble = np.zeros( (2, ens_size_virtual) )
for i in range(2):
    virtual_ensemble[i,:] = norm.cdf( virt_probits[i,:] )
    virtual_ensemble[i,:] = dist_list[i].ppf( virtual_ensemble[i,:] )




# Plot data points
plt.scatter( original_ens[0,:], original_ens[1,:], c='dodgerblue', 
             marker='o', label = 'Original', s=50)
plt.scatter( virtual_ensemble[0,:], virtual_ensemble[1,:], c='r',
             marker='o', label = 'Virtual', s = 1)
plt.legend()

plt.savefig('demo.png')
