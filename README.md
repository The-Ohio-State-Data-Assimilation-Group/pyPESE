# pyPESE

> Draft notice: this documentation was drafted by Codex and has not yet been
> vetted by Man-Yau (Joseph) Chan.

Python implementations of Probit-space Ensemble Size Expansion (PESE) methods
for generating virtual ensemble members. The repository contains two layers:

- `pyPESE/`: the importable Python package with PESE-GC, resampling,
  distribution, ensemble-modulation, and utility routines.
- `supported_models/`: model-facing workflows that apply the package to
  specific model output formats. The current supported model directory is CAM.

Most users should start with the model interface under
`supported_models/CAM/` when working with CAM ensembles. Use the package modules
directly when building a custom workflow for another model or diagnostic.

## Repository Layout

```text
pyPESE/
  pese_gc.py                       Core PESE-GC interfaces
  distributions/                   Marginal distribution classes
  resampling/                      Gaussian and localized resampling helpers
  ensemble_modulation/             Ensemble modulation utilities
  balance_diagnosis/               CAM/balance-related helper routines
  local_noise_generators/          Localized noise generation routines
  utilities/                       Grid, preprocessing, and interpolation tools

supported_models/
  CAM/                             CAM-FV3 driver scripts and example data

simple_demo_pyPESE.py              Minimal package-level PESE-GC example
```

## Requirements

The core package expects Python 3 plus:

- `numpy`
- `scipy`
- `numba`

The examples and model workflows may also require:

- `matplotlib` for `simple_demo_pyPESE.py`
- `netCDF4` for CAM netCDF input/output
- `mpi4py` and an MPI launcher such as `srun` or `mpiexec` for the CAM member
  generation scripts

The code has been used with Python 3.10.13, NumPy 1.26.2, Numba 0.58.1, and
SciPy 1.11.4. There is currently no `pyproject.toml` or `setup.py`, so the
package is normally used directly from a source checkout.

## Using The Package From Source

From the repository root, Python can import `pyPESE` directly:

```bash
python simple_demo_pyPESE.py
```

From another working directory, add the repository root to `PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/pyPESE:${PYTHONPATH}
python your_script.py
```

An older workflow is to create a symbolic link to the `pyPESE/` package
directory from the directory that contains your script.

## Quick Package Example

```python
import numpy as np
from scipy.stats import gamma, skewnorm

from pyPESE.pese_gc import pese_gc

original = np.random.normal(size=(2, 100))
dist_classes = [gamma, skewnorm]
extra_args = [
    {"min bound": 0.0, "max bound": 1.0e9},
    {"min bound": -1.0e9, "max bound": 1.0e9},
]

virtual, coeff_matrix = pese_gc(
    original,
    dist_classes,
    extra_args,
    num_virt_ens=1000,
    rng_seed=0,
)
```

For a fuller working example, see `simple_demo_pyPESE.py`.

## CAM Workflows

CAM users should read `supported_models/CAM/README.md`. That directory contains
scripts to:

- generate vertically localized Gaussian noise samples for PESE-GC
- generate vertically localized CAM virtual members
- generate square-root vertical localization matrices
- generate vertically modulated CAM members

The CAM scripts expect a `config_pyPESE.py` file in the CAM working directory.
Create it by copying/adapting `supported_models/CAM/config_VERTLOC_pyPESE.py`.

## Main Python Interfaces

- `pyPESE.pese_gc.pese_gc(...)`: apply unlocalized PESE-GC to a 2D ensemble
  array shaped `(num_variables, num_original_members)`.
- `pyPESE.pese_gc.pese_gc_univariate(...)`: apply PESE-GC to one variable with
  a precomputed Gaussian resampling matrix, useful for localized workflows.
- `pyPESE.resampling.gaussian_resampling`: build and apply unlocalized Gaussian
  resampling coefficient matrices.
- `pyPESE.resampling.local_gaussian_resampling`: Gaspari-Cohn localization and
  localized Gaussian resampling helpers.
- `pyPESE.distributions.distributions.all_dist_class_dict`: short names for
  bundled distribution classes, including `gauss`, `bbrh`, `muwe`, `expo`,
  `gamma`, `pchip`, `beta`, `gamma_leftbound_zero`, and
  `truncnorm_leftbound_zero`.
- `pyPESE.ensemble_modulation.ensemble_modulation`: utilities for localization
  matrix square roots and ensemble modulation.

## Notes And Limitations

- CAM driver scripts assume CAM netCDF files with a singleton time dimension.
- CAM member-invariant variables are copied from original members; configured
  member-varying variables are resampled or modulated.
- Variables that are neither member-invariant nor listed in the CAM variable
  configuration are omitted from generated virtual member files.
- The CAM scripts expect original member file names to follow a template using
  the token `MemberID`, for example `example_data/cam_member_MemberID.nc`.
