#!/bin/bash

ens_size=20

for imem in `seq -f "%04g" 1 $ens_size`; do

    # Remove old file
    if [[ -e cam_member_$imem.nc ]]; then
        rm cam_member_$imem.nc
    fi

    # Generate file
    ncks    -v lon,slon,lat,slat,lev,ilev,hyam,hybm,hyai,hybi,T,US,VS,Q,PS                  \
            -d lon,0,-1,3       -d slon,2,-1,3      -d lat,1,-1,3    -d slat,2,-1,3         \
            -d lev,-17,-1,1     -d ilev,-18,-1,1                                            \
            $SCRATCH/CAM_DART/example_data_202006010000/f.e21.FHIST_BGC.f09_025.CAM6assim.011.cam_$imem.e.forecast.2020-06-01-00000.nc \
            cam_member_0$imem.nc
done
