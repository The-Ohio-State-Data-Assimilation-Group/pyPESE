from time import time

t0 = time()
from gaussian import gaussian
print('Time to compile and load gaussian functions', time()-t0)

t0 = time()
from scipy.stats import norm
print('Time to compile and load scipy.stats.norm functions', time()-t0)

import numpy as np

ntests = 10000
nsamples = 10


# Draw raw data
samples = np.random.normal( size=(ntests, nsamples))


# Init standard normal using my own distributions
t0 = time()
std_gauss = gaussian( 0.0, 1.0 )
print( 'Time to prep a standard normal gaussian distribution', time()-t0)


# Test my own system
fit_time = 0
ppi_time = 0
ippi_time = 0
for itest in range(ntests):

    # Test speed of my own fitting process
    t0 = time()
    mean, var = gaussian.fit( samples[itest,:] )
    fitted_dist = gaussian( mean, var )
    fit_time += (time()-t0)
    
    # Test speed of my PPI process
    t0 = time()
    latent_space = std_gauss.ppf( 
        fitted_dist.cdf( samples[itest,:] )
    )
    ppi_time += (time() - t0)

    if ( np.abs(np.mean(latent_space)) > 1e-9 ):
        print( 'Error in latent space mean!', np.mean(latent_space) )
        quit()
    if ( np.abs(np.var(latent_space, ddof=1) -1) > 1e-5 ):
        print( 'Error in latent space variance!', np.var(latent_space) )
        quit()

    # Test speed of my inv PPI process
    t0 = time()
    native_space = fitted_dist.ppf(
        std_gauss.cdf( latent_space )
    )
    ippi_time += (time()-t0)

    if ( np.abs( np.mean(native_space) - np.mean(samples[itest,:]) ) > 1e-5 ):
        print( 'Error in native space mean!', np.mean(native_space) - np.mean( samples[itest,:]) )
        quit()
    if ( np.abs( np.var(native_space, ddof=1) - np.var(samples[itest,:], ddof=1) ) > 1e-5 ):
            print( 'Error in native space variance!', 
                    np.var(native_space, ddof=1) - np.var(samples[itest,:], ddof=1) )
            quit()

    


print("\nCUSTOM MADE FUNCTION TIMES")
print( 'fitting times ', fit_time / ntests )
print( '    ppi times ', ppi_time / ntests )
print( '   ippi times ', ppi_time / ntests )





# Test SciPy system
fit_time = 0
ppi_time = 0
ippi_time = 0
for itest in range(ntests):

    t0 = time()
    mean, var = norm.fit( samples[itest,:] )
    fitted_dist = norm( mean, var )
    fit_time += (time()-t0)
    
    t0 = time()
    latent_space = norm.ppf( 
        fitted_dist.cdf( samples[itest,:] )
    )
    ppi_time += (time() - t0)

    t0 = time()
    native_space = norm.cdf(
        fitted_dist.ppf( latent_space )
    )
    ippi_time += (time()-t0)

print("\nSCIPY FUNCTION TIMES")
print( 'fitting times ', fit_time / ntests )
print( '    ppi times ', ppi_time / ntests )
print( '   ippi times ', ppi_time / ntests )