'''
    PYTHON SCRIPT TO SETUP PYPESE PACKAGE
'''

from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Python package for PESE methods'
LONG_DESCRIPTION = 'This package contains the functions needed to run the Probit-space Ensemble Size Expansion (PESE) methods.\n\n'
LONG_DESCRIPTION += 'Reference:\n'
LONG_DESCRIPTION += '    Man-Yau Chan (2024): Improving Ensemble Data Assimilation with Probit-space Ensemble Size Expansion'
LONG_DESCRIPTION += '    for Gaussian Copulas (PESE-GC). Submitted to Nonlinear Processes in Geophysics.'
LONG_DESCRIPTION += '    doi: 10.5194/egusphere-2023-2699'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="pyPESE", 
        version=VERSION,
        author="Man-Yau (Joseph) Chan",
        author_email="<chan.1063@osu.edu>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=['numpy'], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'PESE'],
        classifiers= [
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
        
)