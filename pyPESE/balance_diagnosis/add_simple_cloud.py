import numpy as np


'''
    Useful function to compute saturation ratio
'''
def calc_saturation_ratio(qvapor, tk, pressure ):

    # compute vapor pressure
    e = ( qvapor/ (287/461 +qvapor) ) * pressure

    # compute saturation vapor pressures
    e_sat = np.empty_like(e)
    flag_liq = (tk >= 273.16)
    flag_ice = (tk <  273.16)
    e_sat[flag_liq] = (
        611 
        * np.exp( 
            6808 * (1/273.16 - 1/tk[flag_liq])
            - 5.09 * np.log( tk[flag_liq]/273.16 )
        )
    )
    e_sat[flag_ice] = (
        611 
        * np.exp( 
            6293 * (1/273.16 - 1/tk[flag_ice])
            - 0.555 * np.log( tk[flag_ice]/273.16 )
        )
    )

    # Return saturation ratios
    return e/e_sat





'''
    Function to inject cloud into a CAM file
'''
def add_cloud_to_camfile( ncfile ):
    
    # Load CAM data
    qvap    = np.squeeze( ncfile.variables['Q']     )
    tk      = np.squeeze( ncfile.variables['T']     )
    psurf   = np.squeeze( ncfile.variables['PS']    )
    hyam    = np.squeeze( ncfile.variables['hyam']  )
    hybm    = np.squeeze( ncfile.variables['hybm']  )
    P0      = 1e5

    # Useful information about dimensions
    qvap_dims = ncfile.variables['Q'].dimensions
    qvap_shp = qvap.shape

    # Compute model pressure
    pres3d = np.zeros_like( qvap )
    for kk in range(qvap_shp[0]):
        pres3d[kk] = psurf * hybm[kk] + P0 * hyam[kk]


    # Compute sat ratio
    s3d = calc_saturation_ratio( qvap, tk, pres3d )

    # Generate cloud liquid and save to file
    qcld = np.zeros_like(qvap)
    qcld[ (s3d > 0.99) * (tk >= 273.16) * ( pres3d > 10000 ) ] = 1./1000
    if 'QCLDLIQ' not in ncfile.variables.keys():
        vble = ncfile.createVariable( 
                'QCLDLIQ', 'f8', ('time', 'lev', 'lat', 'lon') 
        )
        vble.long_name = 'Theoretical cloud water mixing ratio'
        vble.units = 'kg/kg'
    ncfile.variables['QCLDLIQ'][0,:,:,:] = qcld

    # Generate cloud ice
    qice = np.zeros_like(qvap)
    qice[ (s3d > 1.05) * (tk < 273.16) * ( pres3d > 10000 ) ] = 1./1000
    if 'QCLDICE' not in ncfile.variables.keys():
        vble = ncfile.createVariable( 
                'QCLDICE', 'f8', ('time', 'lev', 'lat', 'lon') 
        )
        vble.long_name = 'Theoretical cloud ice mixing ratio'
        vble.units = 'kg/kg'
    ncfile.variables['QCLDICE'][0,:,:,:] = qice

    return