''' Script to request for ERA5 single level ensemble within a limited area for a limited time '''

import cdsapi
import numpy as np
import datetime
import sys

# Date to request
date = datetime.datetime.strptime( sys.argv[1], '%Y%m%d%H%M')

# Read in lat-lon range of data to request
max_lat=float( sys.argv[2] )
min_lat=float( sys.argv[3] )
max_lon=float( sys.argv[4] )
min_lon=float( sys.argv[5] )

outname = 'example_data/era5_single_lvl_' + date.strftime('%Y-%m-%d_%H')+'UTC.nc'

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type':'ensemble_members',
        'variable':[   
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_dewpoint_temperature",
            "2m_temperature",
            "mean_sea_level_pressure",
            "surface_pressure",
            "skin_temperature"
        ],

        'year': date.strftime('%Y'),
        'month': date.strftime( '%m' ),
        'day': date.strftime( '%d' ),
        'area': [max_lat, min_lon, min_lat, max_lon],  # N, W, S, E
        'time': date.strftime( '%H' ),
        'data_format': 'netcdf',
        "download_format": "unarchived"
        },
   outname)
