''' Script to request for ERA5 pressure level ensemble within a limited area for a limited time '''

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

outname = 'example_data/era5_ensemble_plvls_' + date.strftime('%Y-%m-%d_%H')+'UTC.nc'

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type':'ensemble_members',
        'format':'netcdf',
        'variable':[   'specific_cloud_ice_water_content','specific_cloud_liquid_water_content',\
                       'specific_humidity','temperature','u_component_of_wind','v_component_of_wind'\

        ],
        'pressure_level':[ '1','2','3', '5','7','10','20','30','50', '70','100','125','150','175','200',
                           '225','250','300','350','400','450','500','550','600','650','700','750',
                           '775','800','825','850','875','900','925','950','975','1000' ],
        'year': date.strftime('%Y'),
        'month': date.strftime( '%m' ),
        'day': date.strftime( '%d' ),
        'area': [max_lat, min_lon, min_lat, max_lon],  # N, W, S, E
        'time': date.strftime( '%H' )
        },
   outname)
