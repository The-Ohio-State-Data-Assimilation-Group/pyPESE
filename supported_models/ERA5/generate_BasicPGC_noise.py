'''
    SCRIPT TO GENERATE NOISE SAMPLES USED BY BASIC PESE-GC
    ======================================================
    Written by: Man-Yau (Joseph) Chan

    This script generates noise samples that are used by basic PESE-GC.
    We need to generate the noise ahead of time to ensure that the same
    noise samples are used consistently. This is just a precautionary 
    measure against the bad scenario where every compute process generates
    different noise samples (which kills all physical relationships).
    
    Example usage:
    -------------------------
        python generate_pyPESE_noise.py
'''




'''
    IMPORT PYTHON PACKAGES
'''

# Import standard packages
import numpy as np
from sys import argv
from copy import deepcopy
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








'''
    GENERATE NOISE SAMPLES
'''
timed_print('Generating noise samples for basic (i.e., unlocalized) PESE-GC')

num_samples = 300*300

np.random.seed(0)

noise = np.random.normal( size = num_samples )







'''
    OUTPUT CORRELATED NOISE AS DICTIONARY WITHIN PICKLE FILE
'''

timed_print('Saving noise samples for basic (i.e., unlocalized) PESE-GC.')

# Prep dictionary
output_dict = {}
output_dict['noise_samples'] = noise.astype('f4')
output_dict['description'] = 'This pickle file contains noise samples for use in Basic PESE-GC.'
output_dict['source code'] = 'pyPESE/supported_models/ERA5/generate_BasicPGC_noise.py'
output_dict['github repo'] = 'https://github.com/The-Ohio-State-Data-Assimilation-Group/pyPESE.git'
output_dict['contact point'] = 'Man-Yau (Joseph) Chan, Department of Geography, The Ohio State University'
output_dict['contact email address'] = 'chan.1063@osu.edu'

# Save data
pkl_fpath = 'noise_for_BasicPGC.pkl'
with open( pkl_fpath, 'wb') as f:
    pickle.dump( output_dict, f )
# --- end of data saving


timed_print( 'Script generate_BasicPGC_noise.py ran successfully.' )
timed_print( 'Exiting.' )