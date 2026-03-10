'''
    PYTHON CODE TO COLLATE ALL AVAILABLE DISTRIBUTIONS
'''

# Load all available distribution classes
from pyPESE.distributions.mixture_user_and_weighted_empirical import mixture_user_weighted_empirical as muwe
from pyPESE.distributions.gaussian import gaussian as gauss
from pyPESE.distributions.bounded_boxcar_rank_histogram import bounded_boxcar_rank_histogram as bbrh
from pyPESE.distributions.exponential import exponential as expo
from pyPESE.distributions.gamma import gamma 
from pyPESE.distributions.pchip import pchip 
from pyPESE.distributions.beta import beta 
from pyPESE.distributions.gamma_leftbound_zero import gamma_leftbound_zero
from pyPESE.distributions.truncnorm_leftbound_zero import truncnorm_leftbound_zero

# Generate dictionary holding all available distribution classes
all_dist_class_dict = {}
all_dist_class_dict['gauss'] = gauss
all_dist_class_dict[ 'bbrh'] = bbrh
all_dist_class_dict[ 'muwe'] = muwe 
all_dist_class_dict[ 'expo'] = expo
all_dist_class_dict['gamma'] = gamma
all_dist_class_dict['pchip'] = pchip
all_dist_class_dict[ 'beta'] = beta
all_dist_class_dict['gamma_leftbound_zero'] = gamma_leftbound_zero
all_dist_class_dict['truncnorm_leftbound_zero'] = truncnorm_leftbound_zero
