# pyPESE

> This documentation was drafted by Codex and has been reviewed by Man-Yau
> (Joseph) Chan. Note that this README file is still incomplete.
>
> If you have any questions or spot mistakes, please email Joseph at
> chan.1063@osu.edu.

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
  pese_gc.py                       DEPRECATED FILE.
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
- `netCDF4` for netCDF input/output
- `mpi4py` and an MPI launcher such as `srun` or `mpiexec` for the CAM member
  generation scripts

The code has been used with Python 3.10.13, NumPy 1.26.2, Numba 0.58.1, and
SciPy 1.11.4. There is currently no `pyproject.toml` or `setup.py`, so the
package is normally used directly from a source checkout.

## Using The Package From Source

From the repository root, Python can import `pyPESE` directly. For example

```python
import pyPESE as pese
```

From another working directory, add the repository root to `PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/pyPESE:${PYTHONPATH}
python your_script.py
```

An older workflow is to create a symbolic link to the `pyPESE/` package
directory from the directory that contains your script.



## CAM Workflows

CAM users should read `supported_models/CAM/README.md`. That directory contains
scripts to:

- generate vertically localized Gaussian noise samples for PESE-GC
- generate vertically localized CAM virtual members
- generate square-root vertical localization matrices
- generate vertically modulated CAM members

The CAM scripts expect a `config_pyPESE.py` file in the CAM working directory.
Create it by copying/adapting `supported_models/CAM/config_VERTLOC_pyPESE.py`.


## Notes And Limitations

- CAM driver scripts assume CAM netCDF files with a singleton time dimension.
- CAM member-invariant variables are copied from original members; configured
  member-varying variables are resampled or modulated.
- Variables that are neither member-invariant nor listed in the CAM variable
  configuration are omitted from generated virtual member files.
- The CAM scripts expect original member file names to follow a template using
  the token `MemberID`, for example `example_data/cam_member_MemberID.nc`.
