'''
    Python package for Bishop & Hodyss' ensemble modulation (EM) technique

    Implemented by Man-Yau Chan on Nov 23, 2025
'''
import numpy as np




'''
    Function to pre-compute square-root of the localization matrix. Per convention, 
    we use eigenvectors for our square-root.
    *   Strictly speaking, the cholesky decomposition of the localization matrix
        is kosher. However, that might pose difficulty with the truncation process


    Inputs:
    -------
    1) L_matrix
            2D NumPy array or matrix containing localization coefficients.
            Must be an invertible matrix. 
    2) n_trunc
            Integer specifying the number of singular vectors to truncate the 
            square-root at
'''
def full_spectrum_matrix_sqrt( L_matrix ):

    # Check symmetricity
    test = np.sum( np.abs( L_matrix - L_matrix.T ) > 1e-5 )
    if test > 1:
        print( 'ERROR from pyPESE/ensemble_modulation/ensemble_modulation.py')
        print( '    Function name:')
        print( '        full_spectrum_matrix_sqrt')
        print( '    Error message:')
        print( '        Inputted localization matrix is not symmetric!')
        print(f'        Number of asymmetric elements: {test}')
        print( 'EXITING.')
        quit()

    # Compute eigen-decomposition
    eigval, eigvec = np.linalg.eigh( L_matrix )
    eigval = eigval.real
    eigval[ eigval < 1e-8] = 0.
    eigvec = eigvec.real


    # Sort eigen decomposition
    sort_ind = eigval.argsort()[::-1]
    
    # Generate square-root matrix
    eigval_sorted = eigval[sort_ind]
    eigvec_sorted = eigvec[:,sort_ind]
    sqrt_matrix = np.matmul(
        eigvec_sorted, np.diag( np.sqrt(eigval_sorted) )
    )

    return sqrt_matrix








'''
    Function to pre-compute square-root of the localization matrix. Per convention, 
    we use eigenvectors for our square-root.
    *   Strictly speaking, the cholesky decomposition of the localization matrix
        is kosher. However, that might pose difficulty with the truncation process


    Inputs:
    -------
    1) L_matrix
            2D NumPy array or matrix containing localization coefficients.
            Must be an invertible matrix. 
    2) n_trunc
            Integer specifying the number of singular vectors to truncate the 
            square-root at
'''
def prep_localization_matrix_sq_root( L_matrix, n_trunc ):

    # Check symmetricity
    test = np.sum( np.abs( L_matrix - L_matrix.T ) > 1e-5 )
    if test > 1:
        print( 'ERROR from pyPESE/ensemble_modulation/ensemble_modulation.py')
        print( '    Function name:')
        print( '        prep_localization_matrix_sq_root')
        print( '    Error message:')
        print( '        Inputted localization matrix is not symmetric!')
        print(f'        Number of asymmetric elements: {test}')
        print( 'EXITING.')
        quit()

    # Compute eigen-decomposition
    eigval, eigvec = np.linalg.eigh( L_matrix )
    eigval = eigval.real
    eigval[ eigval < 1e-8] = 0.
    eigvec = eigvec.real


    # Sort eigen decomposition
    sort_ind = eigval.argsort()[::-1]
    
    # Truncate!
    eigval_trunc = eigval[sort_ind][:n_trunc]
    eigvec_trunc = eigvec[:,sort_ind][:,:n_trunc]

    # Generate truncated localization matrix
    tmp = np.matmul(eigvec_trunc, np.diag(eigval_trunc))
    loc_mat_trunc = np.matmul( 
        tmp,
        eigvec_trunc.T
    )

    # Determine diagonal matrix factor to obtain unit values 
    # in localization matrix diagonal
    # print( np.diag(loc_mat_trunc))
    rescale_matrix = np.diag(np.power(np.diag( loc_mat_trunc ), -0.5))

    # Construct square-root matrix
    sqrt_matrix = (
        np.matrix( rescale_matrix)
        * np.matrix( eigvec_trunc ) 
        * np.matrix( np.diag( np.sqrt(eigval_trunc) ) )
    )
    
    return sqrt_matrix









'''
    Function to apply ensemble modulation onto ensemble.

    Inputs:
    -------
    1) sqrt_matrix 
            2D square NumPy array/matrix containing the sqrt matrix of 
            the localization matrix
            Dims: Nx x Nt   ( Nx is number of state elements,
                              Nt is truncation number. )
    2) ens_states2d
            2D NumPy array holding ensemble of states
            Dims: Nx x Ne   (Ne is number of ensemble members)

'''
def apply_ensemble_modulation( sqrt_matrix, ens_states2d ):
    
    # Check dimensions of inputted matrix and array
    # ---------------------------------------------
    nx, n_trunc = sqrt_matrix.shape
    nx1, ne = ens_states2d.shape
    if (nx != nx1):
        print( 'ERROR from pyPESE/ensemble_modulation/ensemble_modulation.py')
        print( '    Function name:')
        print( '        apply_ensemble_modulation')
        print( '    Error message:')
        print( '        Check dimensions of inputted sqrt_matrix and ens_states2d')
        print(f'        sqrt_matrix.shape: {sqrt_matrix.shape}')
        print(f'        ens_states2d.shape: {ens_states2d.shape}')
        print( 'EXITING.')
        quit()
    
    # Modulation product!
    # ------------------
    # Init array to hold modulated ensemble
    expanded_ens_states2d = np.empty( (nx, n_trunc * ne), dtype='f8')

    # Ensemble moulation is meant to be applied onto the perturbations.
    # Separating ens mean and pert from ens_states2d
    ens_mean = np.mean( ens_states2d, axis = 1)
    raw_perts = np.array( (ens_states2d.T - ens_mean).T )
    sqrt_array = np.array( sqrt_matrix )


    # Apply modulation product onto raw ensemble perturbations
    for itrunc in range( n_trunc ):
        # Determine index of the current modulated member
        imem_st = itrunc * ne
        imem_ed = (itrunc+1) * ne
        expanded_ens_states2d[:,imem_st:imem_ed] = (
            sqrt_array[:,itrunc] * raw_perts.T
        ).T
        # --- End of loop over raw memebrs
    # --- End of loop over sqrt matrix columns

    # Re-scale perturbations to preserve trace
    expanded_ens_states2d *= np.sqrt( n_trunc*ne-1 ) / np.sqrt( ne-1 )

    # Add back ensemble mean
    expanded_ens_states2d = ( expanded_ens_states2d.T + ens_mean ).T
    
    return expanded_ens_states2d
    





















'''
    Function to sanity check prep_localization_matrix_sq_root function.
'''
def SANITY_CHECK_prep_localization_matrix_sq_root(num_elements = 30):

    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

    # Generate localization matrix
    loc_matrix = generate_positive_semidefinite_matrix( num_elements )


    # Square-root matrix without truncation
    # ----------------------------------------

    # Prep full square-root matrix
    full_sqrt_matrix = prep_localization_matrix_sq_root(loc_matrix, num_elements )

    # Reconstruct localization matrix from square-root matrix
    full_loc_matrix = np.matmul( full_sqrt_matrix, full_sqrt_matrix.T )

    
    # Square-root matrix with truncation
    # ----------------------------------

    # Prep truncated square-root matrix
    trunc_sqrt_matrix = prep_localization_matrix_sq_root(loc_matrix, 10 )

    # Reconstruct localization matrix from squarte-root matrix
    trunc_loc_matrix = np.matmul( trunc_sqrt_matrix, trunc_sqrt_matrix.T )
    


    # Plot outcomes
    # -------------
    fig, axs = plt.subplots( nrows=3, ncols=2, figsize=(6,9))

    # Visualize original localization matrix
    ax = axs[0,0]
    cnf = ax.pcolormesh( 
        np.arange(num_elements)+1, np.arange(num_elements)+1, 
        loc_matrix, vmin=0, vmax=1,  cmap='inferno'
    )
    plt.colorbar( cnf, ax=ax )
    ax.set_title('Original Loc Matrix')
    ax.set_aspect('equal')


    # Visualize reconstructed matrix 
    ax = axs[1,0]
    cnf = ax.pcolormesh( 
        # np.arange(num_elements)+1, np.arange(num_elements)+1, 
        full_loc_matrix, vmin=0, vmax=1,  cmap='inferno'
    )
    plt.colorbar( cnf, ax=ax )
    ax.set_title('Full Loc Matrix')
    ax.set_aspect('equal')

    # Visualize errors in reconstructed matrix 
    ax = axs[2,0]
    cnf = ax.pcolormesh( 
        np.arange(num_elements)+1, np.arange(num_elements)+1, 
        np.abs(full_loc_matrix - loc_matrix), 
        norm=colors.LogNorm(vmin=1e-6, vmax=1),
        cmap='Reds'
    )
    plt.colorbar( cnf, ax=ax )
    ax.set_title('Sqrt Process Error')
    ax.set_aspect('equal')


    # Visualize sqrt of localization matrix
    ax = axs[0,1]
    cnf = ax.pcolormesh( 
        np.arange(num_elements)+1, np.arange(num_elements)+1, 
        full_sqrt_matrix, vmin=-1, vmax=1,  cmap='RdBu_r'
    )
    plt.colorbar( cnf, ax=ax )
    ax.set_title('Full Sqrt Matrix')
    ax.set_aspect('equal')


    # Visualize truncated localization matrix
    ax = axs[1,1]
    cnf = ax.pcolormesh( 
        np.arange(num_elements)+1, np.arange(num_elements)+1, 
        trunc_loc_matrix, vmin=0, vmax=1,  cmap='inferno'
    )
    plt.colorbar( cnf, ax=ax )
    ax.set_title('Trunc Loc Matrix')
    ax.set_aspect('equal')

    # Visualize errors in truncated reconstructed matrix 
    ax = axs[2,1]
    cnf = ax.pcolormesh( 
        np.arange(num_elements)+1, np.arange(num_elements)+1, 
        np.abs(trunc_loc_matrix - loc_matrix), 
        norm=colors.LogNorm(vmin=1e-6, vmax=1),
        cmap='Reds'
    )
    plt.colorbar( cnf, ax=ax )
    ax.set_title('Truc Process Error')
    ax.set_aspect('equal')
  

    plt.savefig('SANITY_CHECK_sqrt_process.png')

    return










'''
    Function to generate a symmetric positive semi-definite matrix
'''
def generate_positive_semidefinite_matrix( num_elements, num_filter = 30 ):

    from scipy.ndimage import convolve1d

    # Generate localization matrix via repeated applications of 121 filter
    # --------------------------------------------------------------------
    # Generate un-normalized localization matrix
    out_matrix = np.eye(num_elements)
    conv_weights = [1,1,1]
    for iapply in range( num_filter ):
        out_matrix = convolve1d( out_matrix, conv_weights, mode='wrap' )

    # Normalize localization matrix
    normalizer = np.diag( np.power( np.diag( out_matrix ), -0.5 ) )
    out_matrix = np.matmul(
        np.matmul( normalizer, out_matrix ),
        normalizer
    )

    # Symmetricity checker
    test = np.sum( np.abs( out_matrix - out_matrix.T ) > 1e-5 )
    if test > 1:
        print( 'ERROR from pyPESE/ensemble_modulation/ensemble_modulation.py')
        print( "    Function name:")
        print( '        SANITY_CHECK_prep_localization_matrix_sq_root')
        print( '    Error message:')
        print( '        Constructed localization matrix is not symmetric!')
        print(f'        Number of asymmetric elements: {test}')
        print( 'EXITING.')
        quit()

    return out_matrix













'''
    Function to sanity check application of ensemble modulation
'''
def SANITY_CHECK_apply_ensemble_modulation(num_elements = 30, ens_size=11, num_trunc=10):

    import matplotlib.pyplot as plt
    from scipy.ndimage import convolve1d
    import matplotlib.colors as colors

    # Generate white noise ensemble
    # -----------------------------
    ens_states2d = np.random.normal( size = (num_elements, ens_size ) )

    # Generate localization matrix
    # ----------------------------
    loc_matrix = generate_positive_semidefinite_matrix( num_elements, num_filter = 5 )

    # Generate truncated sqrt of localization matrix
    # ----------------------------------------------
    trunc_sqrt_matrix = np.array( prep_localization_matrix_sq_root( loc_matrix, num_trunc ) )
    trunc_loc_matrix = np.array(np.matmul( trunc_sqrt_matrix, trunc_sqrt_matrix.T ))

    # Apply ensemble modulation
    # -------------------------
    expanded_ens_states2d = apply_ensemble_modulation( trunc_sqrt_matrix, ens_states2d )

    # Generate covariance matrices
    # ----------------------------
    true_cov = np.eye( num_elements)
    raw_ens_cov = np.cov( ens_states2d, ddof=1)
    target_cov = raw_ens_cov * trunc_loc_matrix
    expanded_ens_cov = np.cov( expanded_ens_states2d, ddof=1)
    print( np.diag( raw_ens_cov)/np.diag(expanded_ens_cov))


    # Plot outcomes
    # -------------
    fig, axs = plt.subplots( nrows=3, ncols=2, figsize=(6,9))

    # Visualize true covariance matrix
    ax = axs[0,0]
    cnf = ax.pcolormesh( 
        np.arange(num_elements)+1, np.arange(num_elements)+1, 
        true_cov, vmin=0, vmax=1,  cmap='inferno'
    )
    plt.colorbar( cnf, ax=ax )
    ax.set_title('True Cov Matrix')
    ax.set_aspect('equal')


    # Visualize ensemble matrix 
    ax = axs[1,0]
    cnf = ax.pcolormesh( 
        # np.arange(num_elements)+1, np.arange(num_elements)+1, 
        raw_ens_cov, vmin=0, vmax=1,  cmap='inferno'
    )
    plt.colorbar( cnf, ax=ax )
    ax.set_title('Ens Cov Matrix')
    ax.set_aspect('equal')

    # Visualize localized covariance matrix
    ax = axs[0,1]
    cnf = ax.pcolormesh( 
        np.arange(num_elements)+1, np.arange(num_elements)+1, 
        target_cov, vmin=0, vmax=1,  cmap='inferno'
    )
    plt.colorbar( cnf, ax=ax )
    ax.set_title('Loc Ens Cov Matrix')
    ax.set_aspect('equal')


    # Visualize modulated ensemble covariance matrix
    ax = axs[1,1]
    cnf = ax.pcolormesh( 
        np.arange(num_elements)+1, np.arange(num_elements)+1, 
        expanded_ens_cov, vmin=0, vmax=1,  cmap='inferno'
    )
    plt.colorbar( cnf, ax=ax )
    ax.set_title('Mod Cov Matrix')
    ax.set_aspect('equal')


    # Visualize modulated ensemble covariance matrix error 
    ax = axs[2,1]
    cnf = ax.pcolormesh( 
        np.arange(num_elements)+1, np.arange(num_elements)+1, 
        np.abs( expanded_ens_cov - target_cov), cmap='Reds',
        norm=colors.LogNorm(vmin=1e-6, vmax=1),
    )
    plt.colorbar( cnf, ax=ax )
    ax.set_title('Mod Cov VS Loc Cov')
    ax.set_aspect('equal')

    plt.delaxes( axs[2,0])


    plt.savefig('SANITY_CHECK_ensemble_modulation.png')

    return









'''
    Main program
'''
if __name__ == '__main__':
    SANITY_CHECK_prep_localization_matrix_sq_root()
    SANITY_CHECK_apply_ensemble_modulation()
    
    



