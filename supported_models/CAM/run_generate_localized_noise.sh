#!/bin/bash

# Example SLURM script to run localized noise generation
# ------------------------------------------------------


# Example 1
srun python -u generate_simple_localized_noise_logP.py       \
    -1     0.75   example_data/cam_member_00001.nc    \
    hroi_-1km_vroi_0.75lnP.pkl

# Example 2
srun python -u generate_simple_localized_noise_logP.py       \
    -1     1.50   example_data/cam_member_00001.nc    \
    hroi_-1km_vroi_1.50lnP.pkl