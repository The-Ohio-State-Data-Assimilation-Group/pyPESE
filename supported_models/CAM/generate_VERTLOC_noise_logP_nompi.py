'''
    SCRIPT TO GENERATE VERTICAL NOISE SAMPLES USED IN AL-PESE-GC
    =============================================================
    Written by: Man-Yau (Joseph) Chan

    IMPORTANT NOTES:
        *   The noise samples are 1D samples, and are therefore only 
            compatible with generate_VERTLOC_virtual_members.py

        *   No parallelization is needed for this procedure due to
            the simplicity and usage of vectorized calculations.


    
    Description:
    ------------
        This MPI-enabled Python script generates the locally-correlated
        Gaussian noise samples used to perform localized probit-space 
        resampling within AL-PESE-GC.
        
        Both horizontal & vertical localization are supported.
        
        The Gaspari-Cohn 1999 5th order rational function is used as the 
        localization function. 

        The horizontal Radius of Influence (ROI) is specified in terms of km,
        and is currently assumed to be the same at every grid box.

        The vertical ROI is to be specified in terms of Pa, and is currently
        assumed to be the same at every grid box.

        The outcome of running this script is a pickle file containing 100 
        samples of 3D locally-correlated Gaussian noise samples. This pickle 
        file will be used when running AL-PESE-GC for CAM.

        The number of noise samples generated equals the number of processes used.

    
    Required Python packages: 
    -------------------------
        numpy, mpi4py, sys, json, time, pyPESE, netCDF4, numba, scipy

        
    Required user expertise:
    ------------------------
        1) How to use mpi4py on your computer/cluster
        2) Experience with using/analysing CAM netCDF files

        
    Required data files:
    --------------------
        1) wrfinput or wrfout file containing the mass grid's latitudes,
            longitudes, and base pressure. 
            TIME DIMENSION MUST HAVE ONLY ONE ELEMENT.


    Required user specifications (done via command line input):
    -----------------------------------------------------------
        1) Vertical radius of influence in Pa
        2) Path to CAM netCDF file
        3) Path to save pickle file of localized noise 
    
        

    Example usage with SLURM:
    -------------------------
        python -u generate_localized_noise.py  40000  \
            sample_camfile.nc  localized_noise.pkl
        
        In this example, VROI is 40,000 Pa, the CAM file path is 
        sample_camfile.nc, and the output pickle file path is 
        localized_noise.pkl.

'''







'''
    IMPORT PYTHON PACKAGES
'''

# Import standard packages
import numpy as np
from sys import argv
from copy import deepcopy
from scipy.linalg import sqrtm
import pickle
from time import time


t_start = time()

'''
    USEFUL PRINT FUNCTION
'''
# Function to do timed printing
def timed_print( string ):

    print( '(%6.1f secs elapsed) ---- %s' % ( time()-t_start, string ) )

    return


timed_print( 'Compiling needed pyPESE function.')

# Import netCDF package
from netCDF4 import Dataset as ncopen

# Load noise generation function from pyPESE
from pyPESE.resampling.local_gaussian_resampling import GC99

timed_print( 'Finished compiling needed pyPESE functions.')











'''
    HARDCODED NUMBER OF NOISE SAMPLES TO GENERATE
'''
num_samples = 300*300   # Generating massive noise samples


# Seeding random number generator
np.random.seed(0)







'''
    READ COMMAND LINE ARGUMENTS
'''

# Exception statement
if len( argv ) != 4:
    timed_print( 
        'Example usage: \n    %s\n%s'
        % ( 'python generate_VERTLOC_noise_logP_nompi.py vroi_in_logP sample_camfile.nc pkl_fname',
            'See generate_VERTLOC_noise_logP_nompi.py for more information.'
        )
    )

# Read in ROIs
vroi_in_logP = float( argv[1] )

# Read in wrf file path
wrf_filepath = argv[2]

# Read in output pickle file path
pkl_fpath = argv[3]









'''
    LOADING COORDINATE INFORMATION FROM CAM FILE'
'''

timed_print( 'Loading coordinate information from CAM file' )



# Shove coordinates into a dictionary for convenience
coord_dict ={}

# Assume: wrf file is a wrfinput or wrfout file
ncfile = ncopen( wrf_filepath, 'r' )
coord_dict['lon'] = np.squeeze( ncfile.variables['lon'] ).astype('f8')
coord_dict['lat'] = np.squeeze( ncfile.variables['lat'] ).astype('f8')
coord_dict['pres'] = (
    np.squeeze( ncfile.variables['hyai'] )
    + np.squeeze( ncfile.variables['hybi'] )
).astype('f8')*1e5
coord_dict['nz_stag'] = len( coord_dict['pres'] )
coord_dict['nlon'] = len( coord_dict['lon'] )
coord_dict['nlat'] = len( coord_dict['lat'] )
ncfile.close()



timed_print( 'Finished loading coordinate information from CAM file\n' )











'''
    GENERATE LOCAL NOISE SAMPLES
'''

timed_print( 'Generating vertically-localized noise for CAM-pyPESE' )

# Generate localization matrix
loc_matrix = np.zeros( [coord_dict['nz_stag'], coord_dict['nz_stag']], dtype='f8' )
log_plvls = np.log( coord_dict['pres'] )
for i in range( coord_dict['nz_stag'] ):
    dist1d = np.abs( log_plvls[i] - log_plvls )
    loc_matrix[i,:] = GC99( dist1d, vroi_in_logP )

# Generate symmetric square-root of localization matrix 
sqrt_loc_matrix = sqrtm( loc_matrix)

# Generate vertically-localized noise
noise2d = np.matmul( 
    sqrt_loc_matrix, 
    np.random.normal( size=(coord_dict['nz_stag'], num_samples ) )
)

# Transpose dimensions and standardize noise
noise2d = noise2d.T
noise2d -= np.mean( noise2d, axis=0)
noise2d /= np.std( noise2d, axis=0, ddof=1)



timed_print( 'Finished generating correlated noise for CAM-pyPESE\n' )





'''
    OUTPUT CORRELATED NOISE AS DICTIONARY WITHIN PICKLE FILE
'''

timed_print( 'Saving correlated noise to pickle file %s' % pkl_fpath )


# Prep dictionary
output_dict = {}
output_dict['vert loc noise samples'] = noise2d.astype('f4')
output_dict['dimensions'] = 'num_samples, nz_stag'
output_dict['nz_stag'] = coord_dict['nz_stag']
output_dict['nlat'] = coord_dict['nlat']
output_dict['nlon'] = coord_dict['nlon']
output_dict['VROI in LogP'] = vroi_in_logP
output_dict['lat'] = coord_dict['lat']
output_dict['lon'] = coord_dict['lon']
output_dict['base pressure'] = coord_dict['pres']
output_dict['description'] = 'This pickle file contains vertical noise samples for use in AL-PESE-GC.'
output_dict['source code'] = 'pyPESE/models/CAM/generate_VERTLOC_noise_logP_nompi.py'
output_dict['github repo'] = 'https://github.com/The-Ohio-State-Data-Assimilation-Group/pyPESE.git'
output_dict['contact point'] = 'Man-Yau (Joseph) Chan, Department of Geography, The Ohio State University'
output_dict['contact email address'] = 'chan.1063@osu.edu'

# Save data
with open( pkl_fpath, 'wb') as f:
    pickle.dump( output_dict, f )
# --- end of data saving


timed_print( 'Finished saving correlated noise to pickle file.\n' )






'''
    SUCCESSFUL COMPLETION OF PROGRAM
'''
timed_print( 'Script generate_VERTLOC_noise_logP_nompi.py ran successfully.' )
timed_print( 'Exiting.' )