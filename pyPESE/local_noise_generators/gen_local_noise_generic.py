'''
    FUNCTIONS TO GENERATE FOR LOCAL WEIGHTS NEEDED FOR LOCALIZED PESE ON A GENERIC 3D GRID
    
    Description:
    ------------
    Locally-correlated Gaussian noise samples are needed for localized PESE.
    This package contains generic functions needed to perform this localization.
    
    Both horizontal & vertical localization are supported.
    
    The Gaspari-Cohn 1999 5th order rational function is used as the 
    localization function. 

'''

# Package imports
import numpy as np
from copy import deepcopy
from scipy.ndimage import convolve as scipy_convolve

# Loading functions relating to gaspari-cohn
from pyPESE.resampling.local_gaussian_resampling import GC99, vertical_convolve_2d_noise_GC99

'''
    ALL-PURPOSE BUT SUBOPTIMAL FUNCTION TO GENERATE LOCALIZED NOISE

    Inputs:
    1) x1d
            1D NumPy array containing x-coordinate at EVERY horizontal location of the grid
            Arr length: nx*ny -- where nx & ny are the horizontal dimensions of 3d grid
    2) y1d
            1D NumPy array containing y-coordinate at EVERY horizontal location of the grid
            Arr length: nx*ny -- where nx & ny are the horizontal dimensions of 3d grid
    3) z2d
            2D NumPy array containing z-coordinate at EVERY horizontal location of the grid
            Arr length: nx*ny -- where nx & ny are the horizontal dimensions of 3d grid

'''
def gen_3d_localized_noise( x1d, y1d, zvals2d, xy_dist_calculator, hroi_1d, vroi_2d ):

    # Loop over every location
    return