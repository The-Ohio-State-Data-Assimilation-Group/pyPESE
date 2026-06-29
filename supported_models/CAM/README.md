# CAM-FV3 Workflows

> Draft notice: this documentation was drafted by Codex and has not yet been
> vetted by Man-Yau (Joseph) Chan.

This directory contains model-facing scripts for applying pyPESE to CAM-FV3
netCDF ensemble files. The scripts are intended to be run from this directory
or from a CAM working directory that contains the same scripts, configuration,
and input data layout.

## What The CAM Interface Provides

- `generate_VERTLOC_noise_logP_nompi.py`: generate vertically localized Gaussian
  noise samples in log-pressure coordinates for localized PESE-GC.
- `generate_VERTLOC_virtual_members.py`: generate vertically localized CAM
  virtual members with PESE-GC.
- `generate_logP_vertical_loc_sqrt_matrix.py`: generate a truncated square-root
  vertical localization matrix for ensemble modulation.
- `generate_vertically_modulated_members_Q_CLAMP.py`: generate vertically
  modulated CAM members. The script also imports the CAM cloud helper and
  contains Q-specific clamping behavior.
- `check_VERTLOC_noise_logP.py`: inspect generated vertical localization noise.
- `config_VERTLOC_pyPESE.py`: example configuration template.
- `example_data/`: small CAM member files for local testing and examples.

## Configuration File

The generation scripts import a module named `config_pyPESE`. To use the
checked-in template, copy it before running the scripts:

```bash
cp config_VERTLOC_pyPESE.py config_pyPESE.py
```

Then edit `config_pyPESE.py` for your ensemble.

### `ensemble_configuration`

```python
ensemble_configuration = {
    "original ensemble size": 20,
    "member file name template": "example_data/cam_member_MemberID.nc",
    "expanded ensemble size": 220,
    "truncated localization dimension": 10,
    "resampling mode": "GC",
}
```

Important fields:

- `original ensemble size`: number of real CAM members.
- `member file name template`: path template for all member files. The literal
  token `MemberID` is replaced with zero-padded IDs such as `00001`.
- `expanded ensemble size`: total size after adding virtual members. For
  `generate_VERTLOC_virtual_members.py`, this must be at least three times the
  original ensemble size.
- `truncated localization dimension`: number of vertical localization modes
  retained for ensemble modulation.
- `resampling mode`: currently `GC` for Gaussian copula PESE-GC.

Generated file names use the same template:

- PESE-GC virtual members replace `MemberID` with values like `00021_virt`.
- Vertically modulated members replace `MemberID` with values like
  `00021_modulated`.

### `variable_configuration`

`generate_VERTLOC_virtual_members.py` reads `variable_configuration`.

```python
variable_configuration = {
    "PS": {"marginal": "gauss", "noise pkl file": "test_0p75lnP.pkl"},
    "Q":  {"marginal": "gauss", "noise pkl file": "test_0p75lnP.pkl"},
    "T":  {"marginal": "gauss", "noise pkl file": "test_0p75lnP.pkl"},
}
```

Each key is a CAM variable to resample. Each value selects:

- `marginal`: short distribution name from
  `pyPESE.distributions.distributions.all_dist_class_dict`.
- `noise pkl file`: localized Gaussian noise pickle created by
  `generate_VERTLOC_noise_logP_nompi.py`.

Member-varying CAM variables that are not listed here are not written to the
virtual member files. Member-invariant variables are copied from original
members.

### `modulation_configuration`

`generate_vertically_modulated_members_Q_CLAMP.py` reads
`modulation_configuration`.

```python
modulation_configuration = {
    "PS": {"modulation matrix file": "vroi_0p20lnP.pkl"},
    "Q":  {"modulation matrix file": "vroi_0p20lnP.pkl"},
    "T":  {"modulation matrix file": "vroi_0p20lnP.pkl"},
}
```

Each configured variable points to a square-root localization matrix pickle
created by `generate_logP_vertical_loc_sqrt_matrix.py`.

## Data Assumptions

The CAM scripts assume:

- input files are CAM netCDF files
- the time dimension contains exactly one time entry
- all 0D and 1D variables are member-invariant
- CAM hybrid-coordinate variables such as `hyai`, `hybi`, `hyam`, `hybm`,
  `lon`, and `lat` are present
- enough memory is available to hold the required localized noise or modulation
  data

The MPI member-generation scripts use two internal data layouts:

- state-complete mode: each MPI rank holds the whole model state for assigned
  ensemble members
- ensemble-complete mode: each MPI rank holds a subset of the model state for
  all original and generated members

## Workflow: Vertically Localized PESE-GC CAM Members

1. Prepare the configuration:

```bash
cp config_VERTLOC_pyPESE.py config_pyPESE.py
```

Edit `config_pyPESE.py` so the member template, ensemble sizes, variables, and
noise pickle paths match your CAM ensemble.

2. Generate vertically localized Gaussian noise:

```bash
python -u generate_VERTLOC_noise_logP_nompi.py \
    0.75 \
    example_data/cam_member_00001.nc \
    test_0p75lnP.pkl
```

Arguments:

- vertical radius of influence in log-pressure units
- CAM file used to read the vertical grid and horizontal dimensions
- output pickle path

3. Run PESE-GC virtual member generation with MPI:

```bash
srun python -u generate_VERTLOC_virtual_members.py
```

or, outside SLURM:

```bash
mpiexec -n 4 python -u generate_VERTLOC_virtual_members.py
```

The script reads `config_pyPESE.py`, opens the original member files, and writes
virtual member files using the configured member file template.

## Workflow: Vertically Modulated CAM Members

1. Prepare the configuration:

```bash
cp config_VERTLOC_pyPESE.py config_pyPESE.py
```

Edit `modulation_configuration` and `ensemble_configuration`.

2. Generate a square-root vertical localization matrix:

```bash
python -u generate_logP_vertical_loc_sqrt_matrix.py \
    0.20 \
    example_data/cam_member_00001.nc \
    vroi_0p20lnP.pkl
```

Arguments:

- vertical radius of influence in log-pressure units
- CAM file used to read `hyai`, `hybi`, `hyam`, `hybm`, `lon`, and `lat`
- output pickle path

Set the radius of influence to a negative value to deactivate vertical
localization.

3. Run vertical ensemble modulation with MPI:

```bash
srun python -u generate_vertically_modulated_members_Q_CLAMP.py
```

or:

```bash
mpiexec -n 4 python -u generate_vertically_modulated_members_Q_CLAMP.py
```

## Example Data

The `example_data/` directory contains 20 sample CAM member files named
`cam_member_00001.nc` through `cam_member_00020.nc`. The default configuration
uses these files:

```python
"member file name template": "example_data/cam_member_MemberID.nc"
```

That makes the directory suitable for a smoke test after creating
`config_pyPESE.py`.

## Troubleshooting

- `ModuleNotFoundError: No module named 'config_pyPESE'`: copy or rename the
  template configuration to `config_pyPESE.py`.
- `ModuleNotFoundError: No module named 'pyPESE'`: run from the repository root
  or add the repository root to `PYTHONPATH`.
- Missing original member file errors: check that `member file name template`
  contains `MemberID` and expands to existing files.
- Empty or incomplete generated member files: ensure every member-varying
  variable you need is listed in `variable_configuration` or
  `modulation_configuration`.
- MPI launch failures: confirm `mpi4py` is installed against the MPI
  implementation used by `srun` or `mpiexec`.
