# pyPESE Package

> This documentation was drafted by Codex and has been reviewed by Man-Yau
> (Joseph) Chan. Note that this README file is still incomplete.
>
> If you have any questions or spot mistakes, please email Joseph at
> chan.1063@osu.edu.

This directory is the importable Python package for PESE methods. It is used by
the model-specific workflows under `supported_models/` and can also be imported
directly from custom scripts.

## Core PESE-GC API (NOT USED FOR HIGH-ORDER MODELS LIKE CAM)

### `pese_gc`

```python
from pyPESE.pese_gc import pese_gc

virtual_ens_2d, coeff_matrix = pese_gc(
    fcst_ens_2d,
    list_of_dist_classes,
    list_extra_args,
    num_virt_ens,
    rng_seed=0,
)
```

Inputs:

- `fcst_ens_2d`: 2D NumPy array shaped
  `(num_variables, num_original_members)`.
- `list_of_dist_classes`: one marginal distribution class per variable.
  Classes may come from `scipy.stats` or from `pyPESE.distributions`.
- `list_extra_args`: one dictionary per variable. These dictionaries provide
  preprocessing bounds and any distribution-specific fitting arguments.
- `num_virt_ens`: number of virtual members to generate. This must be greater
  than the number of original members.
- `rng_seed`: seed used to build the Gaussian resampling coefficients.

Returns:

- `virtual_ens_2d`: generated virtual members, shaped
  `(num_variables, num_virt_ens)`.
- `coeff_matrix`: Gaussian resampling coefficient matrix, shaped
  `(num_original_members, num_virt_ens)`.

### `pese_gc_univariate`

```python
from pyPESE.pese_gc import pese_gc_univariate

virtual_ens_1d = pese_gc_univariate(
    fcst_ens_1d,
    dist_class,
    extra_args,
    gauss_resamp_matrix,
)
```

Use this lower-level interface when each variable or grid point needs a
different resampling matrix, as in localized PESE-GC.

## Distribution Classes

Bundled distribution classes are collected in:

```python
from pyPESE.distributions.distributions import all_dist_class_dict
```

Available keys are:

- `gauss`
- `bbrh`
- `muwe`
- `expo`
- `gamma`
- `pchip`
- `beta`
- `gamma_leftbound_zero`
- `truncnorm_leftbound_zero`

Distribution classes used with `pese_gc` must provide:

- a class attribute `name`
- a `fit(...)` method returning initialization parameters
- instance methods `cdf(...)` and `ppf(...)`

The bundled bounded boxcar rank histogram distribution (`bbrh`) uses the
forecast ensemble's first two raw moments and explicit `min bound` and
`max bound` values. `pese_gc` fills the raw moments automatically.

## Resampling Helpers

`pyPESE.resampling.gaussian_resampling` provides unlocalized Gaussian
resampling:

- `compute_unlocalized_gaussian_resampling_coefficients(...)`
- `compute_unlocalized_gaussian_resampling_coefficients_with_precomputed_noise(...)`
- `fast_unlocalized_gaussian_resampling(...)`
- `fast_unlocalized_gaussian_resampling_with_precalculated_coeff_matrix(...)`

`pyPESE.resampling.local_gaussian_resampling` provides localization utilities:

- `GC99(...)`: Gaspari-Cohn 1999 fifth-order localization function
- great-circle distance helpers
- localized Gaussian resampling coefficient generation
- vertical convolution of Gaussian noise

## Ensemble Modulation

`pyPESE.ensemble_modulation.ensemble_modulation` provides:

- `prep_localization_matrix_sq_root(...)`
- `apply_ensemble_modulation(...)`
- supporting matrix square-root utilities

The CAM ensemble modulation workflow uses these functions to apply vertical
ensemble modulation to CAM state variables.

## Utilities And Diagnostics

- `pyPESE.utilities.preprocess_ens`: handles duplicate and out-of-bound values
  before probit transforms.
- `pyPESE.utilities.vertical_interp`: basic pressure and eta-level interpolation.
- `pyPESE.utilities.global_latlon_grid`: global latitude-longitude derivative
  and spherical padding utilities.
- `pyPESE.balance_diagnosis`: geostrophic, frictionless, and CAM cloud helper
  routines.
- `pyPESE.local_noise_generators`: lower-level localized noise generators for
  regular latitude-longitude-pressure and Cartesian grids.

## Minimal Example

```python
import numpy as np
from scipy.stats import norm

from pyPESE.pese_gc import pese_gc
from pyPESE.distributions.bounded_boxcar_rank_histogram import (
    bounded_boxcar_rank_histogram as bbrh,
)

fcst = np.random.normal(size=(2, 50))
dist_classes = [norm, bbrh]
extra_args = [
    {"min bound": -1.0e9, "max bound": 1.0e9},
    {"min bound": fcst[1].min() - 1.0, "max bound": fcst[1].max() + 1.0},
]

virtual, coeffs = pese_gc(fcst, dist_classes, extra_args, 200, rng_seed=0)
```
