'''
    CONFIGURATIONS CONTROLLING RUNTIME BEHAVIOR OF PYPESE-ERA5
    ==========================================================
    Written by Man-Yau (Joseph) Chan

    These configurations are specified through dictionaries.
    Note: no localization!!
'''



'''
    Information regarding the ERA5 ensemble being operated on
    ---------------------------------------------------------
'''
ensemble_configuration = {

    # Number of original members
    'original ensemble size': 10, # ERA5 has 10-member ensemble

    # Number of ensemble members after applying ERA5-pyPESE
    #       Must be at least 3x the original ensemble size
    'expanded ensemble size': 40,


    # Apply PESE-GC on pressure level data?
    'expand pres lvl data?': True,      # True for yes, False for no.

    # NetCDF file containing original ERA5 ensemble on pressure levels
    'original era5 pres lvl file name': 'example_data/era5_ensemble_plvls_2026-09-01_00UTC.nc',

    # NetCDF file to hold VIRTUAL ERA5 ensemble on pressure levels
    'virtual era5 pres lvl file name': 'example_data/era5_virtual_ens_plvls_2026-09-01_00UTC.nc',


    # Apply PESE-GC on single level data?
    'expand single lvl data?': True,      # True for yes, False for no.

    # NetCDF file containing original ERA5 ensemble on single levels
    'original era5 single lvl file name': 'example_data/era5_ensemble_single_lvl_2026-09-01_00UTC.nc',

    # NetCDF file containing VIRTUAL ERA5 ensemble on single levels
    'virtual era5 single lvl file name': 'example_data/era5_virtual_ens_single_lvl_2026-09-01_00UTC.nc',
    
}




'''
    Configurations controlling PESE-GC resampling of ERA5 variables
    --------------------------------------------------------------
    Virtual ERA5 files created by pyPESE will contains two kinds of variables:
    1) Member-invariant variables 
    2) Variables specified here in the variable_configurations Python dictionary.
    
    IMPORTANT:  ERA5 model variables that are neither specified in variable_configurations nor member-
                invariant will be ignored. Ignored variables will not be present in the virtual
                member files.

    Member-invariant variables are identified by max-minus-min (i.e., range) values of zero. 
    Member-invariant variables in virtual members are literal copies of the original members' 
    member-invariant variables. To be clear, PESE methods are not applied on member-invariant 
    variables.

    All variables specified in variable_configurations will be resampled by PESE methods. Currently available
    setting options are choice of marginal distributions

    See pyPESE/pyPESE/distributions/distributions.py for a list of available distributions.

    Here's an example variable_setting:
    ```
        variable_configurations = {
            't'         :   {'marginal': 'gauss'},
            'q'         :   {'marginal': 'gamma'}
        }
    ```
    The pyPESE resampling process for T and PS will use Gaussian marginals & local noise samples contained in
    hroi_0200km_vroi_400hPa.pkl. All other member-varying variables (e.g., QVAPOR) will not be resampled.

'''

# The following variable_configurations applies PESE-GC to ERA5 variables on pres lvls
pres_lvls_variable_configurations = {
    'u'         :   {'marginal': 'gauss' },
    'v'         :   {'marginal': 'gauss' },
    't'         :   {'marginal': 'gauss' },
    'q'         :   {'marginal': 'truncnorm_leftbound_zero' },
    'ciwc'      :   {'marginal': 'truncnorm_leftbound_zero' },
    'clwc'      :   {'marginal': 'truncnorm_leftbound_zero' },
}




# The following variable_configurations applies PESE-GC to ERA5 variables on single lvls
single_lvl_variable_configurations = {
    'u10'       :   {'marginal': 'gauss' },
    'v10'       :   {'marginal': 'gauss' },
    't2m'       :   {'marginal': 'gauss' },
    'd2m'       :   {'marginal': 'gauss' },
    'sp'        :   {'marginal': 'gauss' },
    'skt'       :   {'marginal': 'gauss' },
    'msl'       :   {'marginal': 'gauss' },
}