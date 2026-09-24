'''
    SCRIPT TO SANITY CHECK THE EXPANDED ERA5 ENSEMBLE
'''

# Import standard libraries
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from netCDF4 import Dataset as ncopen

# Import configurations
from config_pyPESE import ensemble_configuration as ens_config_dict
from config_pyPESE import pres_lvls_variable_configurations as plvl_vbl_config_dict
from config_pyPESE import single_lvl_variable_configurations as sngl_vbl_config_dict

typekey_list = ['pres','single']



'''
    Open the files
'''
fdict = {}
for typekey in typekey_list:

    tkey = typekey[0]

    # Skip if we dont want this type of data
    if not ens_config_dict[f'expand {typekey} lvl data?']:
        continue

    fdict[typekey] = {}
    fdict[typekey]['orig'] = ncopen( ens_config_dict[f'original era5 {typekey} lvl file name'], 'r' )
    fdict[typekey]['virt'] = ncopen( ens_config_dict[f'virtual era5 {typekey} lvl file name'], 'r' )

# --- End of loop over type keys




'''
    Checking mean and covariances for variables with gaussian distributions
    (Note: PESE-GC is guaranteed to conserve the means and covariances of gaussians)
'''
orig_ens_size = ens_config_dict['original ensemble size']
virt_ens_size = ens_config_dict['expanded ensemble size'] - orig_ens_size

for typekey in typekey_list:
    tkey = typekey[0]
    # Skip if we dont want this type of data
    if not ens_config_dict[f'expand {typekey} lvl data?']:
        continue

    # Obtain relevant config
    config_dict = {'p': plvl_vbl_config_dict, 's': sngl_vbl_config_dict}[tkey]

    # Check all variables with gaussian distributions specified
    for vname in config_dict:

        dist = config_dict[vname]['marginal']

        # Skip non-gaussian dists
        if dist.lower() != 'gauss':
            continue

        # Load original ensemble data
        orig_ens = np.array( fdict[typekey]['orig'].variables[vname] )
        orig_ens = orig_ens[:orig_ens_size]

        # Init to hold expanded ensemble
        orig_shp = list( orig_ens.shape )
        expd_shp = deepcopy(orig_shp)
        expd_shp[0] = ens_config_dict['expanded ensemble size']
        expd_ens = np.zeros( expd_shp, dtype='f8' ) + np.nan

        # Load expanded ensemble
        expd_ens[:orig_ens_size] = orig_ens
        expd_ens[orig_ens_size:] = np.array( fdict[typekey]['virt'].variables[vname] )

        # Compute and compare ensemble means
        orig_avg = np.mean( orig_ens, axis=0 )
        expd_avg = np.mean( expd_ens, axis=0 )
        abs_diff = np.abs(orig_avg - expd_avg)
        print(
            f'99, 99.5, 100th percentile in absdiff of ens mean of {vname}:', np.percentile( abs_diff.flatten(), [99,99.5,100])
        )
        
        # Compute and compare ensemble variances
        orig_var = np.var( orig_ens, axis=0, ddof=1 )
        expd_var = np.var( expd_ens, axis=0, ddof=1 )
        abs_diff = np.abs(orig_var - expd_var)
        print(
            f'99, 99.5, 100th percentile in absdiff of ens variances of {vname}:', np.percentile( abs_diff.flatten(), [99,99.5,100])
        )

        # Compute covariance of a small area
        orig_var = np.var( orig_ens, axis=0, ddof=1 )
        expd_var = np.var( expd_ens, axis=0, ddof=1 )

        
    # --- End of loop over variables
# --- End of quick checks.

        


'''
    Checking mean and variances for variables with gaussian distributions
    (Note: PESE-GC is guaranteed to conserve the means and covariances of gaussians)
'''
orig_ens_size = ens_config_dict['original ensemble size']
virt_ens_size = ens_config_dict['expanded ensemble size'] - orig_ens_size

for typekey in typekey_list:
    tkey = typekey[0]
    # Skip if we dont want this type of data
    if not ens_config_dict[f'expand {typekey} lvl data?']:
        continue

    # Obtain relevant config
    config_dict = {'p': plvl_vbl_config_dict, 's': sngl_vbl_config_dict}[tkey]

    # Check all variables with gaussian distributions specified
    for vname in config_dict:

        dist = config_dict[vname]['marginal']

        # Skip non-gaussian dists
        if dist.lower() != 'gauss':
            continue

        # Load original ensemble data
        orig_ens = np.array( fdict[typekey]['orig'].variables[vname] )
        orig_ens = orig_ens[:orig_ens_size]

        # Init to hold expanded ensemble
        orig_shp = list( orig_ens.shape )
        expd_shp = deepcopy(orig_shp)
        expd_shp[0] = ens_config_dict['expanded ensemble size']
        expd_ens = np.zeros( expd_shp, dtype='f8' ) + np.nan

        # Load expanded ensemble
        expd_ens[:orig_ens_size] = orig_ens
        expd_ens[orig_ens_size:] = np.array( fdict[typekey]['virt'].variables[vname] )

        # Compute and compare ensemble means
        orig_avg = np.mean( orig_ens, axis=0 )
        expd_avg = np.mean( expd_ens, axis=0 )
        abs_diff = np.abs(orig_avg - expd_avg)
        print(
            f'99, 99.5, 100th percentile in absdiff of ens mean of {vname}:', np.percentile( abs_diff.flatten(), [99,99.5,100])
        )
        
        # Compute and compare ensemble variances
        orig_var = np.var( orig_ens, axis=0, ddof=1 )
        expd_var = np.var( expd_ens, axis=0, ddof=1 )
        abs_diff = np.abs(orig_var - expd_var)
        print(
            f'99, 99.5, 100th percentile in absdiff of ens variances of {vname}:', np.percentile( abs_diff.flatten(), [99,99.5,100])
        )
        
    # --- End of loop over variables
# --- End of loop over type keys.
