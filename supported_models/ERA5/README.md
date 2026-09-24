# PESE-GC for ERA5 Ensemble
> Written by Man-Yau (Joseph) Chan

This directory contains model-facing scripts for applying pyPESE to ERA5
netCDF ensemble files. The scripts are intended to be run from this directory
or from a CAM working directory that contains the same scripts, configuration,
and input data layout.

To use the scripts here, you will need the following python packages:
`pyPESE`, `numpy`, `netCDF4`, `scipy`, `cdsapi`, `numba`



## What The ERA5 Interface Currently Provides

- `download_era5_ensemble_pres_lvls.py`: Download ERA5 ensemble members on pressure
    levels from Climate Data Store.
- `download_era5_ensemble_single_lvl.py`: Download ERA5 ensemble members on single
    levels from Climate Data Store (e.g., 2m temperature)
- `generate_BasicPGC_noise.py`: Generates noise samples used by PESE-GC without 
    localization.
- `generate_BasicPGC_expanded_ensemble.py`: Generate virtual ERA5 members with
    the basic form of PESE-GC (i.e., no localization)
- `config_BPGC_pyPESE.py`: example configuration template.
- `example_data/`: small ERA5 member files for local testing and examples.



## Workflow
1) Run `generate_BasicPGC_noise.py` to generate noise samples needed by the 
    resampling scheme. This only needs to be done once in your life.
```
    python generate_BasicPGC_noise.py
```

2) Download ERA5 data that you desire

3) Copy the template configuration file
```
    cp config_BASIC_pyPESE.py config_pyPESE.py
```

4) Edit `config_pyPESE.py`. 

5) Run `generate_BasicPGC_expanded_ensemble.py` to generate the desired virtual
    members. This will take some time.
```
    python -u generate_BasicPGC_expanded_ensemble.py
```
