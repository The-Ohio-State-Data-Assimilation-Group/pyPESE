'''
    TRAPEZOIDAL INTERVAL DISTRIBUTION
    =================================================================================

    Produces a distribution with piecewise linear and continuous PDF
    In other words, the CDF is smooth and 1st order differentiable.

'''


import numpy as np
# from numba import njit
# from numba import float64 as nb_f64
# from numba.types import Tuple as nb_tuple



















'''
    FUNCTION TO EVALUATE PDF OF TRAPEZOIDAL INTERVAL DISTRIBUTION
'''
def trapezoidal_interval_pdf( eval_locs, pdf_def_locs, pdf_def_vals ):

    return np.interp( eval_locs, pdf_def_locs, pdf_def_vals )






'''
    FUNCTION TO EVALUATE THE MEAN (M1) OF TRAPEZOIDAL INTERVAL DISTRIBUTION

    Within each interval, there is a 2nd order polynomial integral to evaluate
        Linear PDF function * x

     Uses 2-point Gauss-Legendre quadrature (works up to cubic polynomials, hence suitable)
'''
def trapezoidal_interval_m1( pdf_def_locs, pdf_def_vals ):

    return





'''
    FUNCTION TO EVALUATE THE VARIANCE (M2) OF TRAPEZOIDAL INTERVAL DISTRIBUTION

    Within each interval, there is a 3-th order polynomial integral to evaluate
        Linear PDF function * x^2

    Uses 2-point Gauss-Legendre quadrature (works up to cubic polynomials, hence suitable)
'''
def trapezoidal_interval_m2( pdf_def_locs, pdf_def_vals ):

    return




'''
    FUNCTION TO EVALUATE THE THIRD CENTRAL MOMENT (M3) OF TRAPEZOIDAL INTERVAL DISTRIBUTION

    Within each interval, there is a 4-th order polynomial integral to evaluate
        Linear PDF function * x^3

    Uses 3-point Gauss-Legendre quadrature (works up to 5-th order polynomials, hence suitable)
'''
def trapezoidal_interval_m3( pdf_def_locs, pdf_def_vals ):

    return







'''
THREE-MOMENT TRAPEZOIDAL INTERVAL DISTRIBUTION
------------------------------------------------

0.5 ( 0      + p1 ) * d0     = A0
0.5 ( p1     + p2 ) * d1     = (1-A0-A2)/(N-1)
0.5 ( p2     + p3 ) * d2     = (1-A0-A2)/(N-1)
                            .
                            .
                            .
0.5 ( p(N-1) + pN ) * d2     = (1-A0-A2)/(N-1)
0.5 ( pN     + 0  ) * d(N+1) = A2

'''




























































'''
    SANITY CHECKS
'''

if __name__ == '__main__':

    from matplotlib import use as mpl_use
    mpl_use('agg')
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    
    samples = np.random.normal(size=500)

    # brh_pts, brh_cdf = bounded_rank_histogram.fit(samples)
    # brh_dist = bounded_rank_histogram( brh_pts, brh_cdf)
    
    # many_pts = np.linspace( -4,4, 1000 )
    # pdf_vals = brh_dist.pdf( many_pts )
    # cdf_vals = brh_dist.cdf( many_pts )


    # # Plot PDF and CDF
    # fig, axs = plt.subplots( nrows=1, ncols=2, figsize = (6,3) )
    # axs[0].plot( many_pts, pdf_vals, '-r')
    # axs[0].set_title('BRH PDF')
    # axs[1].plot( many_pts, cdf_vals, '-r')
    # axs[1].set_title('BRH CDF')

    # # Overlay with actual CDF
    # gaussian_cdf = norm.cdf( many_pts )
    # axs[1].plot( many_pts, gaussian_cdf, ':k' )
    # plt.savefig('visualize_brh.png')
    # plt.close()