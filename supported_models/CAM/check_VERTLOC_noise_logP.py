'''
    SCRIPT TO SANITY CHECK THE VERTICALLY LOCALIZED NOISE SAMPLES
    =============================================================

    Written by Man-Yau (Joseph) Chan

    Example usage:
    --------------
        python check_VERTLOC_noise_logP.py name_of_pickle_file.pkl
'''



'''
    IMPORT PACKAGES NEEDED
'''
import numpy as np
import pickle
from math import pi as PI
from matplotlib import use as mpl_use
mpl_use('agg')
import matplotlib.pyplot as plt
from sys import argv
from pyPESE.resampling.local_gaussian_resampling import GC99


DEG2PI = PI/180


'''
    LOAD LOCALIZED NOISE SAMPLES
'''
fname = argv[1]
with open( fname, 'rb' ) as f:
    data_dict = pickle.load(f)

# ROIs
vroi_in_logP = data_dict['VROI in LogP']       # *np.sqrt(2)



'''
    COMPUTE SPATIAL CORRELATIONS AT THE CENTER OF THE NOISE SAMPLE ARRAYS
'''
# Shape of array
nsamples, nlvl = data_dict['vert loc noise samples'].shape


# Generate covariance matrix of noise samples
noise_cov2d = np.cov( data_dict['vert loc noise samples'].T )





'''
    COMPUTE THEORETICAL VERTICAL CORRELATION PATTERNS BASED ON PRESCRIBED INFORMATION

    Will compute the theoretical correlations here and compare against the localized noise
'''

# Vertical localization matrix evaluation
# -------------------------------------------
log_plvls = np.log( data_dict['base pressure'] )
true_loc_matrix = np.zeros( (nlvl, nlvl), dtype='f8' )
for i in range( nlvl ):
    dist1d = np.abs( log_plvls[i] - log_plvls )
    true_loc_matrix[i,:] = GC99( dist1d, vroi_in_logP )



'''
    PLOT OUT CORRELATIONS
'''
fig, axs = plt.subplots( nrows=1, ncols=2, layout='constrained', figsize=(8,3))

# Noise covariance matrix
ax = axs[0]
cnf = ax.contourf( 
    data_dict['base pressure'], data_dict['base pressure'], noise_cov2d,
    np.linspace(-0.99,0.99,12), cmap='RdBu_r', extend='both'
)
ax.set_title('Noise Cov Matrix')
plt.colorbar(cnf, ax=ax)


# Noise covariance matrix
ax = axs[1]
cnf = ax.contourf( 
    data_dict['base pressure'], data_dict['base pressure'], true_loc_matrix,
    np.linspace(-0.99,0.99,12), cmap='RdBu_r', extend='both'
)
ax.set_title('Localization Matrix')
plt.colorbar(cnf, ax=ax)


# Decorations
for ax in axs:
    ax.contour( 
        data_dict['base pressure'], data_dict['base pressure'], true_loc_matrix,
        np.linspace(-0.99,0.99,12), colors='k', linestyles=':'
    )

    ax.set_xlabel('Pres')
    ax.set_ylabel('Pres')
    ax.axis('equal')



plt.savefig('fig_check_VERTLOC_noise.png', dpi=200)
plt.close()