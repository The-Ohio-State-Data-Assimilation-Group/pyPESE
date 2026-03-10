'''
    POSITIVE SEMI-DEFINITE TRUNCATED NORMAL DISTRIBUTION FOR MIXING RATIOS
    =================================================================================
    Implementation of left-truncated normal distribution for positive semidefinite 
    quantities (e.g., mixing ratios)

    Moment matching is employed here, and is implemented with a look-up table.
'''
import numpy as np
from pyPESE.distributions.gaussian import STANDARD_NORMAL_INSTANCE as norm
# from gaussian import STANDARD_NORMAL_INSTANCE as norm # Activate during debugging






'''
    Function to evaluate variance/mean^2 over a range of mu/sigma ratios
'''
def prep_twomoment_ratios( npts = 1001 ):

    # Range of mu/sigma ratios
    alpha1d = (-MAX_MU_TO_SIGMA_RATIO * (np.linspace(0,1, npts)**2))[::-1]

    # Compute g for all alphas
    g1d = norm.pdf(alpha1d) / ( np.ones( npts ) - norm.cdf(alpha1d))

    # Compute variance/mean^2 ratios
    var_mean2_ratio1d = ( 1 - g1d*(g1d-alpha1d) ) / np.power(g1d - alpha1d,2 )

    return alpha1d, var_mean2_ratio1d






'''
    Hardcoded settings
'''
MAX_MU_TO_SIGMA_RATIO = 5
TRUNCNORM_SPECHUM_ALPHA, TRUNCNORM_VARI_AVG2_RATIOS = prep_twomoment_ratios()









'''
    SciPy-like class for positive truncated normal distribution.
'''
class truncnorm_leftbound_zero:

    name = 'positive truncated normal distribution for humidity'

    # Initialize 
    def __init__( self, mu, sig ):
        self.mu = mu
        self.sig = sig
        self.left_bnd = 0
        self.alpha = -mu/sig
        self.Z = 1 - norm.cdf(self.alpha)
        return

    # Fit distribution -- requires solving some equations
    def fit( data1d ):

        # Solve for alpha
        avg = np.mean(data1d)
        var = np.var( data1d, ddof=1 )
        ratio = var / (avg**2)

        # Situation where distribution is noticeably truncated normal
        if np.abs(avg) / np.sqrt(var) < MAX_MU_TO_SIGMA_RATIO:
            fitted_alpha = np.interp( ratio, TRUNCNORM_VARI_AVG2_RATIOS, TRUNCNORM_SPECHUM_ALPHA )

        # Situation where distribution is well approximated by a normal distribution
        else:
            fitted_alpha = - np.abs(avg) / np.sqrt(var)
        
        # Compute sigma
        fitted_g = norm.pdf( fitted_alpha ) / (1 - norm.cdf(fitted_alpha))
        fitted_sig = avg / (fitted_g - fitted_alpha)

        # Compute mu 
        fitted_mu = fitted_sig * fitted_alpha * -1

        return fitted_mu, fitted_sig

    # Function to evaluate CDF (uses CDF formula on Wikipedia)
    def cdf(self, eval_pts):
        xi = (eval_pts - self.mu)/self.sig
        return (norm.cdf( xi ) - norm.cdf( self.alpha ))/self.Z

    # Function to evaluate PPF
    def ppf( self, eval_cdf ):
        # Convert CDF to CDF of standard normal
        cdf_std_norm = eval_cdf * self.Z + norm.cdf(self.alpha)
        # Apply PPF of std normal
        xi = norm.ppf( cdf_std_norm )
        # Convert xi to x and return
        return xi*self.sig + self.mu

    # Fucntion to evaluate PDF
    def pdf( self, eval_pts ):
        xi = (eval_pts - self.mu)/self.sig
        return norm.pdf( xi ) / (self.Z * self.sig)

    # Function to draw samples consistent with fitted distribution
    def rvs( self, shape ):
        q_samples = np.random.uniform( size = shape )
        return self.ppf( q_samples )

    
# ------ End of Gaussian distribution SciPy-like class





'''
    Sanity check distribution class
'''
def SANITY_CHECK_truncnorm_leftbound_zero( nsamples = 9999 ):

    from scipy.stats import truncnorm
    import matplotlib.pyplot as plt


    # Set up true distribution
    left_bnd = 0 ;  right_bnd = 9999
    mu = 1e-4    ;  sig = 2e-4
    scipy_a = (left_bnd - mu)/sig
    scipy_b = (right_bnd - mu)/sig
    true_dist = truncnorm( scipy_a, scipy_b, loc=mu, scale=sig )
    print( true_dist.support())

    # Draw samples from true distribution and apply fitting
    samples = true_dist.rvs( size = nsamples )
    params = truncnorm_leftbound_zero.fit( samples )
    print( params)
    fit_dist = truncnorm_leftbound_zero( *params )

    dist_dict = {}
    dist_dict['true'] = true_dist
    dist_dict['fit']  = fit_dist

    # Prep fig
    fig, axs = plt.subplots( nrows=2, ncols=2, figsize=(8,8) )
    eval_pts = np.linspace(0,mu+sig*4, 101)

    # -------------- COMPARE PDF AND CDF

    linecolors = ['r', 'dodgerblue']

    # Loop over distributions
    for ikey, key in enumerate( dist_dict.keys()):
        # Make PDF
        axs[0,0].plot( eval_pts, dist_dict[key].pdf(eval_pts), label = key.upper(), 
                     color=linecolors[ikey], linewidth=5 - ikey*3 )
        # Make CDF
        axs[0,1].plot( eval_pts, dist_dict[key].cdf(eval_pts), label = key.upper(), 
                     color=linecolors[ikey], linewidth=5 - ikey*3 )
    # -- end of loop over distributions

    # Make PDF and CDF plots pretty
    axs[0,0].set_title('PDF Check')
    axs[0,0].set_ylabel('PDF')
    axs[0,0].legend()
    axs[0,1].set_title('CDF Check')
    axs[0,1].legend()
    axs[0,1].set_ylabel('CDF')

    # -------------- CHECK PPF METHOD
    test_quantiles = np.linspace(1e-3, 1-1e-3, 10)
    test_ppf_vals = fit_dist.ppf( test_quantiles )
    axs[1,0].plot( eval_pts, fit_dist.cdf(eval_pts), '-r', linewidth=2, label = 'Fit Dist CDF')
    axs[1,0].scatter( test_ppf_vals, test_quantiles, s=30, marker='o', c='k', label = 'PPF test vals', zorder=10)
    axs[1,0].legend()
    axs[1,0].set_ylabel('CDF')
    axs[1,0].set_title('PPF Check')

    # -------------- CHECK RVS METHOD
    sorted_samples = np.sort(fit_dist.rvs(9999))
    sample_quants = (np.arange(9999)+1)/9999
    axs[1,1].plot( eval_pts, fit_dist.cdf(eval_pts), '-r', linewidth=10, label = 'Fit Dist CDF')
    axs[1,1].scatter( sorted_samples, sample_quants, s=1, marker='o', c='k', label = 'Empirical CDF', zorder=10)
    axs[1,1].set_title('RVS Check (9999 samples drawn)')
    axs[1,1].legend()
    axs[1,1].set_ylabel('CDF')


    for i in range(2):
        for j in range(2):
            axs[i,j].set_xlabel('x')

    plt.tight_layout()
    plt.savefig( 'SANITY_CHECK_truncnorm_leftbound_zero.png' )
    plt.close()

    return
















# Sanity check
def SANITY_CHECK_prep_twomoment_ratios( mu_true = 0.135e-4, sig_true = 2e-4 ):

    import matplotlib.pyplot as plt

    # Settings of target distribution
    alpha_true = -mu_true / sig_true

    # Generate CDF 
    x1d = np.linspace( 0, mu_true + 5*sig_true, 1001 )
    xi_true1d = (x1d - mu_true )/sig_true
    Z = 1 - norm.cdf(alpha_true)
    cdf_true = (norm.cdf(xi_true1d) - norm.cdf(alpha_true))/Z

    # Determine true ratio
    var_true = (sig_true**2) * (
        1 + alpha_true * norm.pdf(alpha_true)/Z
        - ( norm.pdf(alpha_true)/Z )**2
    )
    avg_true = mu_true + norm.pdf(alpha_true) * sig_true / Z
    ratio_true = var_true / (avg_true**2)

    # Generate twomoment ratios
    alpha1d, var_mean2_ratio1d = prep_twomoment_ratios( npts = 1001 )

    # Numerically solve for alpha
    alpha_soln = np.interp( ratio_true, var_mean2_ratio1d, alpha1d)

    # Return error level
    err_lvl = np.abs( 1 - alpha_soln / alpha_true )

    return err_lvl



# Repeated tests of the prep_twomoment_ratio function
def RAND_SANITY_CHECK_prep_twomoment_ratios(ntests = 1000):

    err_lvls1d = np.zeros( ntests)
    
    for i in range(1000):

        # Carefully draw mu and sigma
        for j in range(100):
            mu_true = np.random.uniform()
            sig_true = np.random.uniform()
            if mu_true / sig_true < MAX_MU_TO_SIGMA_RATIO:
                break
        # --- End of mu and sigma draws

        err_lvls1d[i] = SANITY_CHECK_prep_twomoment_ratios( 
            mu_true = mu_true, 
            sig_true = sig_true)

    print( 'RAND_SANITY_CHECK_prep_twomoment_ratios outcomes...')
    print( 'Quartiles of the relative error (0th, 25th, 50th, 75th, 100th percentiles):')
    print( np.percentile( err_lvls1d, np.linspace(0,100,6)))
    print( 'Relative error values less than 1e-3 are desirable.')

    return
    















if __name__ == '__main__':
    # RAND_SANITY_CHECK_prep_twomoment_ratios()
    SANITY_CHECK_truncnorm_leftbound_zero()

    # Visualize the fitting curve
    import matplotlib.pyplot as plt
    plt.plot( np.power(TRUNCNORM_VARI_AVG2_RATIOS, -0.5), TRUNCNORM_SPECHUM_ALPHA)
    plt.plot( [0, MAX_MU_TO_SIGMA_RATIO], [0, -MAX_MU_TO_SIGMA_RATIO], linestyle='--', color='r')
    plt.xlabel( 'Avg / Std Dev')
    plt.ylabel( r'$\alpha$ parameter')
    plt.savefig( 'truncnorm_fitting_curve.png')

    






