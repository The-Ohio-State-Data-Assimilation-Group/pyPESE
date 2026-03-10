'''
    SCRIPT TO GENERATE SQUARE-ROOT OF VERTICAL LOCALIZATION MATRIX
    =============================================================
    Written by: Man-Yau (Joseph) Chan

    
    Description:
    ------------
        This Python script generates a truncated square-root of 
        a vertical covariance matrix.

    
    Required Python packages: 
    -------------------------
        numpy, sys, time, pyPESE, netCDF4, numba, scipy

        
    Required user expertise:
    ------------------------
        1) Experience with using/analysing CAM netCDF files

        
    Required data files:
    --------------------
        1) data file containing the mass grid's latitudes,
            longitudes, and base pressure. 
            TIME DIMENSION MUST HAVE ONLY ONE ELEMENT.


    Required user specifications (done via command line input):
    -----------------------------------------------------------
        1) Vertical radius of influence in log Pa
        2) Path to CAM netCDF file
        3) Path to save pickle file of matrix square-root
    
        

    Example usage:
    --------------
        python -u generate_logP_vertical_loc_sqrt_matrix.py  5   \
            sample_caminput_d01.nc  localized_noise.pkl
        
        In this example, VROI is 5 logP, the caminput file path is 
        sample_caminput_d01.nc, and the output pickle file path is 
        localized_noise.pkl.

        Note: to deactivate vertical localization, set VROI to a negative number.
'''






'''
    IMPORT PYTHON PACKAGES
'''

# Import standard packages
import numpy as np
from sys import argv
from copy import deepcopy
import pickle

# Import netCDF package
from netCDF4 import Dataset as ncopen

# Import ensemble modulation stuff
from pyPESE.ensemble_modulation.ensemble_modulation import prep_localization_matrix_sq_root
from pyPESE.resampling.local_gaussian_resampling import GC99


'''
    READ COMMAND LINE ARGUMENTS
'''

# Exception statement
if len( argv ) != 4:
    print( 
        'Example usage: \n    %s\n%s'
        % ( 'srun python generate_logP_vertical_loc_sqrt_matrix.py vroi_in_logP caminput_d01 pkl_fname',
            'See generate_logP_vertical_loc_sqrt_matrix.py for more information.'
        )
    )

# Read in ROIs
vroi_in_logP = float( argv[1] )

# Read in cam file path
cam_filepath = argv[2]

# Read in output pickle file path
pkl_fpath = argv[3]


# HARDCODED SETTING: TRUNCATION LEVEL
num_trunc = 10






'''
    LOADING COORDINATE INFORMATION FROM CAM FILE'
'''

print('Loading coordinate information from CAM file')


# Shove coordinates into a dictionary for convenience
out_dict ={}

# Assume: cam file is a caminput or camout file
ncfile = ncopen( cam_filepath, 'r' )
out_dict['pres_stag'] = (
    np.squeeze( ncfile.variables['hyai'] )
    + np.squeeze( ncfile.variables['hybi'] )
).astype('f8')*1e5
out_dict['pres_mass'] = (
    np.squeeze( ncfile.variables['hyam'] )
    + np.squeeze( ncfile.variables['hybm'] )
).astype('f8')*1e5
out_dict['nz_stag'] = len( out_dict['pres_stag'] )
out_dict['nz_mass'] = len( out_dict['pres_mass'] )

out_dict['lon'] = np.squeeze( ncfile.variables['lon'] ).astype('f8')
out_dict['lat'] = np.squeeze( ncfile.variables['lat'] ).astype('f8')
out_dict['nlon'] = len( out_dict['lon'] )
out_dict['nlat'] = len( out_dict['lat'] )

ncfile.close()


print('Finished loading coordinate information from CAM file\n')







'''
    FOR EACH KIND OF VERTICAL GRID, GENERATE SQRT OF LOCALIZATION MATRIX
'''
print('Generate square-root of vertical localization matrix')
for pname in ['pres_stag', 'pres_mass']:

    nz = out_dict[pname].shape[0]

    # Generate localization matrix
    loc_matrix = np.empty( (nz,nz), dtype='f8' )
    for k0 in range(nz):
        dist_arr1d = np.abs(
            np.log( out_dict[pname][k0] ) 
            - np.log( out_dict[pname] ) 
        )
        loc_matrix[k0,:] = GC99( dist_arr1d, vroi_in_logP )
    # --- End of loop over rows of localization matrix

    # Generate truncated square-root matrix
    sqrt_matrix = prep_localization_matrix_sq_root( loc_matrix, num_trunc )

    # Save square-root matrix
    out_dict[pname+'_sqrt_loc'] = np.array( sqrt_matrix )

# ---- End of loop over pressure types

print('Finished generating square-root of vertical localization matrix\n')


# print( loc_matrix[::5,::5])





'''
    OUTPUT SQUARE-ROOT MATRICES AS DICTIONARY WITHIN PICKLE FILE
'''

print( 'Saving square-root matrices to pickle file %s' % pkl_fpath )

# Save data
with open( pkl_fpath, 'wb') as f:
    pickle.dump( out_dict, f )
# --- end of data saving

# --- End of instructions for root process

print('Finished saving square-root matrices to pickle file.\n')






'''
    SUCCESSFUL COMPLETION OF PROGRAM
'''
print( 'Script generate_logP_vertical_loc_sqrt_matrix.py ran successfully.')
print( 'Exiting.')