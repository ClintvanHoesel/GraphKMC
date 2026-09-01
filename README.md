# GraphKMC
Kinetic Monte Carlo algorith in Python to work on graph-based systems.
Written during my PhD to verify orientation factor effects in triplet-triplet annihilation and triplet-polaron quenching in phopsphorescent OLEDs.

## Installation

GraphKMC is now finally a Python package. Install it as

```bash
python -m pip install -e .
```

HDF5-backed multi-verse tracking additionally requires the optional
`hdfdict` dependency (`python -m pip install -e '.[hdf5]'`).

The package API is available as `graphkmc`:

```python
from graphkmc import Lattice, Point, ProcessTable, Run
```

The configurable steady-state simulation can be launched with
`graphkmc-steady-state --help`.  The original parameterised simulations are
kept as command-line programs in [`scripts/`](scripts/README.md); for example,
`python scripts/main_pure_radiative_decay.py`.

Simulation programs create their output in `data/`.  Production scripts use
`SLURM_JOB_ID` in output names, so set it when running outside SLURM.
