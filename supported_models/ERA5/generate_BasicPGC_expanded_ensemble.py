'''
    SCRIPT TO GENERATE EXPANDED ENSEMBLE OF ERA5 MEMBERS
    ====================================================
    Written by: Man-Yau (Joseph) Chan

    This program assumes we have enough noise samples to generate the 
    desired number of virtual members    
    
    Description:
    ------------
        This is a serial python script. No MPI is used.

        IMPORTANT: NO SPATIAL "COVARIANCE" LOCALIZATION IS APPLIED!

        The noise used here comes from generate_BasicPGC_noise.py

        See Chan (2024) for algorithm description
        
        The settings are defined in a file named config_BASIC_pyPESE.py


    Assumption about ERA5 data:
    --------------------------
        All 1D variables and 0D variables are assumed to be the same across
        all ensemble members

        Time dimension is assumed to be singleton (i.e., only 1 time entry)

        Ensemble dimension is assumed to be the leftmost dimension.
        

    Software requirements
    ---------------------
        1) Python packages: NumPy, SciPy, NetCDF4 and pyPESE


'''



'''
    IMPORT PYTHON PACKAGES
'''

# Import standard packages
import numpy as np
from sys import argv
from copy import deepcopy
from math import ceil as ceiling
from scipy.ndimage import convolve as scipy_convolve
import pickle
from os.path import isfile
from gc import collect as gc_collect
from shutil import copyfile as shutil_copyfile

# Import netCDF package
from netCDF4 import Dataset as ncopen

# Import useful parts of PyPESE package
from pyPESE.resampling.gaussian_resampling import fast_unlocalized_gaussian_resampling_with_precalculated_coeff_matrix 
from pyPESE.resampling.gaussian_resampling import compute_unlocalized_gaussian_resampling_coefficients_with_precomputed_noise
from pyPESE.distributions.distributions import all_dist_class_dict
from pyPESE.distributions.gaussian import STANDARD_NORMAL_INSTANCE as std_norm_dist
from pyPESE.balance_diagnosis.add_simple_cloud import add_cloud_to_camfile

# Import configurations
from config_pyPESE import ensemble_configuration as ens_config_dict
from config_pyPESE import pres_lvls_variable_configurations as plvl_vbl_config_dict
from config_pyPESE import single_lvl_variable_configurations as sngl_vbl_config_dict

from time import time

# CPU timer
t_start = time()

# Function to do timed printing
def timed_print( string ):

    print( '(%6.1f secs elapsed) ---- %s' % ( time()-t_start, string ) )

    return










'''
    LOAD NOISE SAMPLES FOR BASIC PESE-GC
'''
pkl_fpath = 'noise_for_BasicPGC.pkl'

# Sanity check: does file exist?
if not isfile( pkl_fpath ):
    timed_print(f'ERROR: Missing {pkl_fpath}')
    quit()

with open(pkl_fpath, 'rb') as f:
    noise_dict = pickle.load(f)
    










'''
    PARSE ENSEMBLE CONFIGURATION
'''
timed_print( 'Parsing ensemble_configuration')

# Ensemble sizes
fcst_ens_size = ens_config_dict['original ensemble size']
virt_ens_size = ens_config_dict['expanded ensemble size'] - ens_config_dict['original ensemble size']

# Sanity check
if virt_ens_size/fcst_ens_size < 2:
    timed_print('ERROR: The expanded ensemble size specified in config_pyPESE.py is too small.')
    timed_print('       The minimum expanded ensemble size is 2x the original ensemble size.')
    quit()
# --- End of sanity check.

timed_print("Finished parsing ensemble_configuration.\n")











'''
    LOAD ORIGINAL ENSEMBLE DATA
'''

# Dictionary to hold original ensemble data
orig_ens_dict = {}
orig_ens_dict['variables'] = {}

timed_print("Loading original ERA5 members.")



# Load original ensemble pressure level data data
if ens_config_dict['expand pres lvl data?']:

    timed_print('    Loading original ERA5 ensemble members pressure level data.')

    orig_ens_dict['plvl vnames'] = []
    
    f = ncopen( ens_config_dict['original era5 pres lvl file name'], 'r' )

    # Load all variables
    for vname in f.variables.keys():

        # Register variable name
        orig_ens_dict['plvl vnames'].append( vname )

        # Load variable
        orig_ens_dict['plvl variables'][vname] = {}
        orig_ens_dict['plvl variables'][vname]['attributes'] = deepcopy(
            f.variables[vname].__dict__
        )
        orig_ens_dict['plvl variables'][vname]['dimensions'] = deepcopy(
            f.variables[vname].dimensions
        )
        orig_ens_dict['plvl variables'][vname]['data'] = np.array(
            f.variables[vname]
        )

    # Load all global attributes
    orig_ens_dict['plvl attributes'] = deepcopy( f.__dict__ )

    # Close file to release handle
    f.close()

# --- End of procedure to load pressure level data







# Load original ensemble single level data data
if ens_config_dict['expand single lvl data?']:

    timed_print('    Loading original ERA5 ensemble members single level data.')

    orig_ens_dict['slvl vnames'] = []
    
    f = ncopen( ens_config_dict['original era5 single lvl file name'], 'r' )

    # Load all variables
    for vname in f.variables.keys():

        # Register variable name
        orig_ens_dict['slvl vnames'].append( vname )

        # Load variable
        orig_ens_dict['slvl variables'][vname] = {}
        orig_ens_dict['slvl variables'][vname]['attributes'] = deepcopy(
            f.variables[vname].__dict__
        )
        orig_ens_dict['slvl variables'][vname]['dimensions'] = deepcopy(
            f.variables[vname].dimensions
        )
        orig_ens_dict['slvl variables'][vname]['data'] = np.squeeze(
            f.variables[vname]
        )

    # Load all global attributes
    orig_ens_dict['slvl attributes'] = deepcopy( f.__dict__ )

    # Close file to release handle
    f.close()

# --- End of procedure to load single level data


timed_print('Finished loading original ERA5 members.\n')










'''
    APPLY BASIC PESE-GC
'''
timed_print('Applying basic PESE-GC...')


# Dictionary to hold expanded ensemble.
expd_ens_dict = {}


# Compute gaussian resampling coefficient matrix E
nsamples = fcst_ens_size * virt_ens_size
E_matrix = compute_unlocalized_gaussian_resampling_coefficients_with_precomputed_noise(
    fcst_ens_size, virt_ens_size, noise_dict['noise_samples'][:nsamples]
)


# Perform Basic PESE-GC for various types of data
for typekey in ['pres', 'single']:

    # Skip over unneeded resampling
    if ens_config_dict[f'expand {typekey} lvl data?']:

        # 














'''
    TERMINATING PROGRAM
'''
msg = 'SUCCESS -- ERA5-pyPESE ran to completion. Exiting program.'
timed_print(msg)