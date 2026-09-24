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

from time import time

# CPU timer
t_start = time()

# Function to do timed printing
def timed_print( string ):
    print( '(%6.1f secs elapsed) ---- %s' % ( time()-t_start, string ) )
    return

# Import standard packages
import numpy as np
from sys import argv
from copy import deepcopy
import pickle
from os.path import isfile
from gc import collect as gc_collect

# Import netCDF package
from netCDF4 import Dataset as ncopen

timed_print('Finished loading standard packages.\n')


# Import useful parts of PyPESE package
from pyPESE.resampling.gaussian_resampling import compute_unlocalized_gaussian_resampling_coefficients_with_precomputed_noise
from pyPESE.distributions.distributions import all_dist_class_dict
from pyPESE.distributions.gaussian import STANDARD_NORMAL_INSTANCE as std_norm_dist
timed_print('Finished compiling and importing pyPESE package.\n')


# Import configurations
from config_pyPESE import ensemble_configuration as ens_config_dict
from config_pyPESE import pres_lvls_variable_configurations as plvl_vbl_config_dict
from config_pyPESE import single_lvl_variable_configurations as sngl_vbl_config_dict
timed_print('Finished loading configurations.\n')











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


timed_print("Loading original ERA5 members.")



# Load original ensemble data
for typekey in ['pres','single']:
    tkey = typekey[0]

    # Skip if we dont want this kind of data.
    if not ens_config_dict[f'expand {typekey} lvl data?']:
        continue

    orig_ens_dict[f'{tkey}lvl variables'] = {}

    timed_print(f'    Loading original ERA5 ensemble members {typekey} level data.')

    orig_ens_dict[f'{tkey}lvl vnames'] = []
    
    f = ncopen( ens_config_dict[f'original era5 {typekey} lvl file name'], 'r' )

    # Load all variables
    for vname in f.variables.keys():

        # Skipping failure mode
        if vname == 'expver':
            continue

        # Register variable name
        orig_ens_dict[f'{tkey}lvl vnames'].append( vname )

        # Load variable
        orig_ens_dict[f'{tkey}lvl variables'][vname] = {}
        orig_ens_dict[f'{tkey}lvl variables'][vname]['attributes'] = deepcopy(
            f.variables[vname].__dict__
        )
        orig_ens_dict[f'{tkey}lvl variables'][vname]['dimensions'] = deepcopy(
            f.variables[vname].dimensions
        )
        
        orig_ens_dict[f'{tkey}lvl variables'][vname]['data'] = np.array(
            f.variables[vname]
        )

    # Load all global attributes
    orig_ens_dict[f'{tkey}lvl attributes'] = deepcopy( f.__dict__ )

    # Close file to release handle
    f.close()

# --- End of loop over single and pressure level data.


timed_print('Finished loading original ERA5 members.\n')











'''
    APPLY BASIC PESE-GC
'''
timed_print('Applying basic PESE-GC...')


# Dictionary to hold virtual ensemble.
virt_ens_dict = {}


# Compute gaussian resampling coefficient matrix E
nsamples = fcst_ens_size * virt_ens_size
E_matrix = compute_unlocalized_gaussian_resampling_coefficients_with_precomputed_noise(
    fcst_ens_size, virt_ens_size, noise_dict['noise_samples'][:nsamples]
)


# Perform Basic PESE-GC for various types of data
for typekey in ['pres', 'single']:

    tkey = typekey[0]
    vbl_config_dict = {'p': plvl_vbl_config_dict, 's': sngl_vbl_config_dict}[tkey]

    # Skip if user doesnt want to expand the current kind of data.
    if not ens_config_dict[f'expand {typekey} lvl data?']:
        continue

    timed_print(f'Applying PESE-GC onto {typekey} variables.')

    virt_ens_dict[f'{tkey}lvl variables'] = {}

    # Loop over available variables
    for vname in orig_ens_dict[f'{tkey}lvl variables']:

        # Duplicate variable properties 
        virt_ens_dict[f'{tkey}lvl variables'][vname] = deepcopy(
            orig_ens_dict[f'{tkey}lvl variables'][vname]
        )

        # Duplicate variables that are 1D or 0D -- these are constant fields
        if orig_ens_dict[f'{tkey}lvl variables'][vname]['data'].ndim <= 2:

            # Special treatment for number variable
            if vname == 'number':
                virt_ens_dict[f'{tkey}lvl variables'][vname]['data'] = (
                    np.arange( virt_ens_size ) + fcst_ens_size
                )
            # --- End of special treatment

            continue

        # Skip over variables with no corresponding resampling configs
        if vname not in vbl_config_dict:
            _ = virt_ens_dict[f'{tkey}lvl variables'].pop(vname, None) 
            continue

        # User-specified marginal distribution family for current variable.
        user_dist_name = vbl_config_dict[vname]['marginal']

        timed_print(f'... applying PESE-GC onto {typekey} variable {vname} (dist: {user_dist_name}).')

        # Detect dimension corresponding to the ensemble member id
        dim_list = list( orig_ens_dict[f'{tkey}lvl variables'][vname]['dimensions'] )
        ens_dim_id = dim_list.index( 'number' )

        # Re-order data dimensions s.t. ensemble dimension is last.
        vble_data = np.swapaxes(
            orig_ens_dict[f'{tkey}lvl variables'][vname]['data'], ens_dim_id, -1
        )

        # Flatten spatial dimensions to simplify loop structure
        old_dims = deepcopy( vble_data.shape )
        spatial_dims = deepcopy( vble_data.shape[:-1] )
        old_ens_size = deepcopy( vble_data.shape[-1] )
        fcst_vble_data = vble_data.reshape( (np.prod(spatial_dims), old_ens_size) )

        # Eliminate ensemble members if needed
        fcst_vble_data = fcst_vble_data[:,:fcst_ens_size]

        # Init array to hold virtual memebrs
        virt_vble_data = np.zeros( (fcst_vble_data.shape[0], virt_ens_size), dtype='f8')


        # Purrform PESE-GC
        for ix in range( fcst_vble_data.shape[0] ):

            fcst_ens1d = fcst_vble_data[ix,:]

            # Only execute PESE-GC if the ensemble is not entirely degenerate
            if np.abs( fcst_ens1d.max() - fcst_ens1d.min() ) > 1e-7:

                # -----------------------------------------------------------------------------------
                #    PESE CODE HERE
                # -----------------------------------------------------------------------------------

                # Step 1: fit MUWE distribution at current location
                #     Note that the user-specified distribution in muwe is the one specified in 
                #     config_CAM_pyPESE.py
                params = all_dist_class_dict['muwe'].fit(
                    fcst_ens1d, all_dist_class_dict[user_dist_name]
                )
                fitted_dist = all_dist_class_dict['muwe']( 
                    *params
                )

                # Step 2: Transform to probit space
                fcst_probit1d = std_norm_dist.ppf(
                    fitted_dist.cdf( fcst_ens1d )
                )

                # Step 3: Resample in probit space
                fcst_probit1d -= np.mean( fcst_probit1d)
                fcst_probit1d /= np.std( fcst_probit1d, ddof=1)                   
                virt_probit1d = np.matmul( fcst_probit1d[np.newaxis,:], E_matrix )
                virt_probit1d = np.array(virt_probit1d)[0,:]

                # Step 4: Transform from probit space to native space
                virt_ens1d = fitted_dist.ppf(
                    std_norm_dist.cdf( virt_probit1d )
                )
                virt_ens1d = virt_ens1d.astype('f8')
                

                # Checking for strange values
                if np.sum( np.isnan( virt_ens1d) + np.isinf(virt_ens1d) ) > 0:
                    msg = "Strange virtual values for %s detected at (%d, %f, %f)" % (
                        vname, ilvl,
                        my_xy_ind_dict[gridtype]['y inds'][iloc], my_xy_ind_dict[gridtype]['x inds'][iloc]
                    )
                    msg += "\nfcst_ens1d: " + str( fcst_ens1d)
                    msg += "\nfcst_probit1d: " + str( fcst_probit1d )
                    msg += "\nvirt_probit1d: " + str( virt_probit1d )
                    msg += "\nvirt_ens1d:"  + str( virt_ens1d)
                    timed_print(msg)
                    timed_print( virt_probit1d )


                # -----------------------------------------------------------------------------------
                #    PESE CODE ENDS HERE
                # -----------------------------------------------------------------------------------
                
            # --- End of handling case where the ensemble is not entirely degenerate.

            # Cases where the ensemble is essentially fully degenerate
            if np.abs( fcst_ens1d.max() - fcst_ens1d.min() ) <= 1e-7:
                virt_ens1d = np.mean( fcst_ens1d )
            # --- End of handling fully degenerate ensemble

            # Put 1d data into virtual array
            virt_vble_data[ix,:] = virt_ens1d
            
        # --- End of loop over available elements

        # Undo dimension manipulations on virtual data
        virt_dims = list( spatial_dims )
        virt_dims.append( virt_ens_size )
        virt_vble_data = np.swapaxes(
            virt_vble_data.reshape( 
                virt_dims
            ), -1, ens_dim_id
        )

        # quick check
        print( np.mean( virt_vble_data[:,0,10,10,10], axis=0 ) )
        print( np.mean( np.swapaxes(vble_data, 0,-1)[:,0,10,10,10], axis=0 ) )
        quit()

        # Hold onto virtual ensemble
        virt_ens_dict[f'{tkey}lvl variables'][vname]['data'] = deepcopy(virt_vble_data)

        gc_collect()

    # --- End of loop over variables within type
    timed_print(f'Finished applying PESE-GC onto {typekey} variables.\n')
# --- End of loop over pressure and single-lvl variable types












'''
    WRITE VIRTUAL MEMBERS INTO VIRTUAL ENSEMBLE FILE
'''
# Iterate over both kinds of ensemble data
for typekey in ['pres', 'single']:

    tkey = typekey[0]
    vbl_config_dict = {'p': plvl_vbl_config_dict, 's': sngl_vbl_config_dict}[tkey]

    # Skip if user doesnt want to expand the current type of data.
    if not ens_config_dict[f'expand {typekey} lvl data?']:
        continue

    timed_print(f'Writing virtual ERA5 ensemble {typekey} lvl data to file...')

    # Init virtual ensemble file
    out_fname = ens_config_dict[f'virtual era5 {typekey} lvl file name']
    virt_ncfile = ncopen( out_fname, 'w' )

    # Open original data file
    orig_ncfile = ncopen( 
        ens_config_dict[f'original era5 {typekey} lvl file name'], 'r'
    )

    # Copy global attributes from original file to virtual file
    virt_ncfile.setncatts(
        {attr: orig_ncfile.getncattr(attr) for attr in orig_ncfile.ncattrs()}
    )

    # Copy all but the "number" dimension
    for name, dimension in orig_ncfile.dimensions.items():
        # Skip over number dimension
        if name == 'number':
            continue
        # Actual copying process
        virt_ncfile.createDimension(name, (len(dimension) if not dimension.isunlimited() else None))
    # --- End of loop over non-number dimensions

    # Generate new number dimension
    virt_ncfile.createDimension('number', virt_ens_size)

    # Save virtual members into netcdf file
    for name, variable in orig_ncfile.variables.items():

        # Skipping over variables not recorded in virt_ens_dict
        if name not in virt_ens_dict[f'{tkey}lvl variables']:
            continue

        timed_print(f'... writing {name}')

        # Init variable.
        x = virt_ncfile.createVariable(
            name, variable.datatype, variable.dimensions
        )
        x.setncatts({k: variable.getncattr(k) for k in variable.ncattrs()})

        virt_ncfile[name][:] = (
            virt_ens_dict[f'{tkey}lvl variables'][name]['data'][:] 
        )
    
    # --- End of loop over variables

    # Save virtual members
    virt_ncfile.close()

    # Release file handle
    orig_ncfile.close()

    timed_print(f'Finished writing virtual ERA5 ensemble {typekey} lvl data to file.\n')

# --- End of loop over data types




    










'''
    TERMINATING PROGRAM
'''
msg = 'SUCCESS -- ERA5-pyPESE ran to completion. Exiting program.'
timed_print(msg)