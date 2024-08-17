'''
    FUNCTIONS TO EVOKE PROBIT-SPACE ENSEMBLE SIZE EXPANSION FOR GAUSSIAN COPULAS
    ============================================================================

    
    List of functions:
    ------------------
    1) pese_gc
            Function to call PESE-GC.
'''

# Load standard Python packages
import numpy as np
# from scipy.stats import norm

# Using Joseph's ~20x faster STANDARD_NORMAL_INSTANCE for calculations
from pyPESE.distributions.gaussian import STANDARD_NORMAL_INSTANCE as norm

# Load pyPESE's custom packages
from pyPESE.resampling.gaussian_resampling import compute_unlocalized_gaussian_resampling_coefficients

# Load pyPESE's ensemble preprocessor to deal with duplicate/out-of-bounds values
from pyPESE.utilities.preprocess_ens import preprocess_ens













'''
    FUNCTION TO EXECUTE UNIVARIATE DISTRIBUTION FITTING (PESE-GC STEP 1)

    This function can handle fitting scipy.stats distributions or pyPESE-defined distributions


    Mandatory Inputs:
    -----------------
    1) fcst_ens1d
            1D NumPy array containing an ensemble of values for a forecast model variable

    2) dist_class
            Scipy/pyPESE statistical distribution class

    3) extra_args
            A Python dictionary containing entries needed to fit the distribution in question
'''
def univariate_dist_fit( fcst_ens1d, dist_class, extra_args ):

    # Special fitting procedure for BBRH distribution
    if dist_class.name == 'bounded boxcar rank histogram':

        params = dist_class.fit( 
            fcst_ens1d, 
            extra_args['raw moment 1'], 
            extra_args['raw moment 2'],
            min_bound = extra_args['min bound'], 
            max_bound = extra_args['max bound']
        )
    
    # For all other distributions, the fitting procedure is easier
    else:
        params = dist_class.fit( fcst_ens1d )

    # --- End of distribution fitting procedure

    # Apply fitted parameters into distribution class to initialize an instance of that 
    # distribution class, and then return the instance
    fitted_dist = dist_class( *params )

    return fitted_dist

    














'''
    MAIN FUNCTION TO EVOKE PESE-GC WITHOUT LOCALIZATION

    
    Mandatory Inputs:
    ------------------
    1) fcst_ens_2d (dimensions: num_variables x num_fcst_ens)
            Two-dimensional NumPy float array containing forecast ensemble data
            The leftmost dimension corresponds to model variables
            The rightmost dimension corresponds to the ensemble members.

    2) list_dist_classes (number of elements: num_variables)
            A Python List object containing the distribution classes for every ensemble 
            variable.
            There must be num_variables entries in the list.
            Two kinds of distribution classes are currently supported:
                a) scipy.stats distribution classes (e.g., skewnorm, gamma, norm)
                b) pyPESE-defined distributions 
    
    3) list_extra_args (number of elements: num_variables)
            A Python List of dictionaries. 
            Each dictionary contains information needed for preprocessing the ensemble
            and to evoke the distribution fitting process.

    4) num_virt_ens (scalar integer)
            Number of virtual members to create.
            Must be greater than num_fcst_ens. This condition is required to ensure that the 
            resampling coefficient generation algorithm works.

    5) rng_seed (scalar integer)
            Seed for NumPy's in-built random number generator.
            This seed is useful for parallelized asynchronous fast
            Gaussian resampling

'''
def pese_gc( fcst_ens_2d, list_of_dist_classes, list_extra_args, num_virt_ens, rng_seed=0 ): 

    # Determine all dimension sizes
    num_variables, num_fcst_ens = fcst_ens_2d.shape


    # Check if number of virtual members is appropriate.
    if ( num_virt_ens <= num_fcst_ens ):
        print( 'ERROR: pese_gc')
        print( '    Number of virtual members must be more than number of original members')
        quit()


    # Generate Gaussian resampling coefficient matrix
    gauss_resamp_matrix = compute_unlocalized_gaussian_resampling_coefficients( 
        num_fcst_ens, num_virt_ens, rng_seed=rng_seed 
    )


    # Init array to hold the virtual ensemble
    virt_ens_2d = np.zeros( [num_variables, num_virt_ens] )

    # Init array to hold fcst and virtual probits
    fcst_probit = np.zeros( [1, num_fcst_ens] )


    # For memory efficiency, only applying PESE-GC on one variable at a time.
    for ivar in range( num_variables ):

        # Preamble: Compute first two raw moments of the ensemble 
        #           These moments are useful for fitting the bounded boxcar rank histogram
        #           distribution.
        list_extra_args[ivar]['raw moment 1'] = np.mean( fcst_ens_2d[ivar,:] )
        list_extra_args[ivar]['raw moment 2'] = np.mean( np.power(fcst_ens_2d[ivar,:], 2) )


        # Preamble: If min and max bounds are not provided in list_extra_args, then 
        #           supplement with the max and min bounds of the forecast ensemble
        offsetting_interval = np.std( fcst_ens_2d[ivar,:] ) * 3./num_fcst_ens
        if ( 'min bound' not in list_extra_args[ivar] ):
            list_extra_args[ivar]['min bound'] = fcst_ens_2d[ivar,:].min() - offsetting_interval
        if ( 'max bound' not in list_extra_args[ivar] ):
            list_extra_args[ivar]['max bound'] = fcst_ens_2d[ivar,:].max() + offsetting_interval


        # Preamble: Handling situation where the min bound and the max bound are the same
        support = list_extra_args[ivar]['max bound'] - list_extra_args[ivar]['min bound']
        if ( np.abs(support) < 4e-5 ):    
            if ( np.abs(list_extra_args[ivar]['min bound']) > 1e-3 ):
                list_extra_args[ivar]['min bound'] -= list_extra_args[ivar]['min bound'] * 2e-5
                list_extra_args[ivar]['max bound'] += list_extra_args[ivar]['min bound'] * 2e-5
            else:
                list_extra_args[ivar]['min bound'] = -2e-5
                list_extra_args[ivar]['max bound'] = 2e-5
        # --- End of handling degenerate ensemble bounds            


        # Preamble: Remove duplicates and/or out-of-bounds values from ensemble
        extra_args = list_extra_args[ivar]
        fcst_ens_2d[ivar,:] = preprocess_ens( 
            fcst_ens_2d[ivar,:], min_bound = extra_args['min bound'],
            max_bound = extra_args['max bound'] 
        )

        # Step 1: Fit distribution to fcst ensembel for selected variable
        extra_args = list_extra_args[ivar]
        fitted_dist = univariate_dist_fit(
            fcst_ens_2d[ivar,:], list_of_dist_classes[ivar], extra_args
        )

        # Step 2: Transform forecast ensemble into probit space
        fcst_probit[0,:] = norm.ppf( 
            fitted_dist.cdf( fcst_ens_2d[ivar,:] )
        )
        if ( np.sum( np.isnan(fcst_probit) + np.isinf(fcst_probit) ) > 0 ):
            print( 
                'ERROR: Invalid value detected\n',  
                fcst_probit[0,:] ,'\n',  
                fcst_ens_2d[ivar,:], '\n', 
                extra_args['min bound'], extra_args['max bound'],'\n'
                'MEOW' )
        
        fcst_probit -= np.mean(fcst_probit)
        fcst_probit /= np.std( fcst_probit, ddof=1)

        # Step 3: Apply fast Gaussian resampling
        virt_probit = ( np.matmul( fcst_probit, gauss_resamp_matrix ) )

        # Step 4: Invert PPI transforms on virtual probits
        virt_ens_2d[ivar,:] = (
            fitted_dist.ppf(
                norm.cdf( virt_probit[0,:] )
            )
        )

    # --- End of loop over variables

    
    return virt_ens_2d, gauss_resamp_matrix




