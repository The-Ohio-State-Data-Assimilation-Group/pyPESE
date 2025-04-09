# Probit-space Ensemble Size Expansion (PESE) Methods

This repository contains Python codes for PESE methods, and accompanies the following research journal manuscript

> Chan, M-Y., Rubin, J., Satterfield, E. and Hyer, E. J.: A Dime a Hundred: Cost-Effective Large Weather Forecast Ensembles through Probit-space Ensemble Size Expansion for Gaussian Copulas (PESE-GC). _Submitting to the Journal of Advances in Modeling Earth Systems._

&nbsp; &nbsp; 

## Usage
1) Install the packages listed under "Python-related requirements" below.
2) Download/Clone this repository (`pyPESE`) into your work directory.
3) Determine the directory in which you will create a Python script (`SCRIPT_DIR`) to perform PESE-GC.
4) Within `SCRIPT_DIR`, generate a symbolic link this repository's pyPESE sub-directory (i.e., `pyPESE/pyPESE`).
5) Within `SCRIPT_DIR`, write the Python script to call pyPESE on your data. See `simple_demo_pyPESE.py` for a toy example.

&nbsp; &nbsp; 

## Coming soon!
1) Parallelized Python script to apply PESE-GC on WRF ensembles, MPAS ensembles & ERA5 ensembles.
2) A website documenting the pyPESE (might be hosted on Github or an Ohio State University page).

&nbsp; &nbsp; 

## Python-related requirements
1) Python v3
2) NumPy package
3) Numba package
4) Scipy package


&nbsp; &nbsp; 

## Relevant Python Packages used by pyPESE' creator
1) Python v3.10.13
2) NumPy v1.26.2
3) Numba v0.58.1
4) SciPy v1.11.4

