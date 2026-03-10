'''
    CONFIGURATIONS CONTROLLING RUNTIME BEHAVIOR OF PYPESE-CAM
    =========================================================
    Written by Man-Yau (Joseph) Chan

    These configurations are specified through dictionaries.
'''



'''
    Information regarding the CAM ensemble being operated on
    ---------------------------------------------------------
'''
ensemble_configuration = {

    # Number of original members
    'original ensemble size': 20,

    # Original members' template file name
    #       MemberID will be replaced with a 5-digit zeropadded value
    #       indicating the member identification number.
    'member file name template': 'example_data/cam_member_MemberID.nc',

    # Number of ensemble members after applying CAM-pyPESE
    #       Must be at least 3x the original ensemble size
    'expanded ensemble size': 220,

    # Ensemble modulation localization matrix eigendecomposition truncation
    # ONLY USED FOR ENSEMBLE MODULATION!!! Not used by PESE itself
    'truncated localization dimension': 10,

    # Mode to use for resampling: Gaussian Copula ("GC") or Kernel ("KERN")
    'resampling mode': 'GC',
    
}




'''
    Configurations controlling PESE-GC resampling of CAM variables
    --------------------------------------------------------------
    Virtual CAM member files created by pyPESE will contains two kinds of variables:
    1) Member-invariant variables (e.g., XLAT, XLONG, PB & PHB)
    2) Variables specified here in the variable_configurations Python dictionary.
    
    IMPORTANT:  CAM model variables that are neither specified in variable_configurations nor member-
                invariant will be ignored. Ignored variables will not be present in the virtual
                member files.

    Member-invariant variables are identified by max-minus-min (i.e., range) values of zero. 
    Member-invariant variables in virtual members are literal copies of the original members' 
    member-invariant variables. To be clear, PESE methods are not applied on member-invariant 
    variables.

    All variables specified in variable_configurations will be resampled by PESE methods. Currently available
    setting options are (1) choice of marginal distributions, and (2) name of Python pickle file containing 
    localized noise samples needed for AL-PESE-GC.

    Currently supported marginal distributions are Gaussian (gauss) and bounded boxcar rank histogram (bbrh).
    TODO: automatic mixture distribution generation when degenerate ensemble values are detected.
    TODO: support for gamma distribution

    Here's an example variable_setting:
    ```
        variable_configurations = {
            'T'         :   {'marginal': 'gauss',        'noise pkl file': 'hroi_0200km_vroi_400hPa.pkl' },
            'PS'        :   {'marginal': 'gauss',        'noise pkl file': 'hroi_0200km_vroi_400hPa.pkl' }
        }
    ```
    The pyPESE resampling process for T and PS will use Gaussian marginals & local noise samples contained in
    hroi_0200km_vroi_400hPa.pkl. All other member-varying variables (e.g., QVAPOR) will not be resampled.

'''

# The following variable_configurations applies PESE-GC to majority of the important prognostic variables in CAM
variable_configuration = {
    'PS'        :   {'marginal': 'gauss',        'noise pkl file': 'test_0p75lnP.pkl'  },
    'US'        :   {'marginal': 'gauss',        'noise pkl file': 'test_0p75lnP.pkl'  },
    'VS'        :   {'marginal': 'gauss',        'noise pkl file': 'test_0p75lnP.pkl'  },
    'Q'         :   {'marginal': 'gauss',        'noise pkl file': 'test_0p75lnP.pkl'  },
    'T'         :   {'marginal': 'gauss',        'noise pkl file': 'test_0p75lnP.pkl'  },
}






# The following configurations are specific to ensemble modulation
# --- These are not called for regular pyPESE
modulation_configuration = {
    'PS'        :   {'modulation matrix file': 'vroi_0p20lnP.pkl'  },
    'US'        :   {'modulation matrix file': 'vroi_0p20lnP.pkl'  },
    'VS'        :   {'modulation matrix file': 'vroi_0p20lnP.pkl'  },
    'Q'         :   {'modulation matrix file': 'vroi_0p20lnP.pkl'  },
    'T'         :   {'modulation matrix file': 'vroi_0p20lnP.pkl'  },
}