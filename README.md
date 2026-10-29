# MultiGTF

**MultiGTF** is a modification and extension of the [**pygtfcode** package](https://github.com/yaronetokayer/pygtfcode).  It is a modern Python implementation of a 1D Lagrangian gravothermal fluid code that can handle mutiple species and simulate mass segregation due to conduction driven by two-body relaxation.

See the [demo notebook](examples/demo.ipynb) for a fuller walkthrough of the API, configuration options, and example simulations. [Saved demo outputs](examples/demo_output/) are also provided for analysis without rerunning the long simulations.

Contact yarone.tokayer-at-yale.edu with questions or comments.

---

## Overview

The code is organized around **two user-facing classes**:

### 1. `Config`

Stores the input parameters and species definitions:

- `io`: Output paths, logging, and snapshot cadence.
- `grid`: Initial radial domain and number of shells.
- `spec`: A dictionary of species, each with a particle mass (`m_part`), mass-fraction weight (`frac`), and initial profile (`init`).
- `sim`: Physical options, transport coefficients, binary heating, background potential, and stopping limits.
- `prec`: Numerical tolerances and iteration limits.
- `mtot`: The model's physical mass normalization.

Profiles are configured separately for each species through `config.spec[name].init`. Available profile families include NFW, truncated NFW, alpha-beta-gamma, exponential, King, and exponentially truncated NFW. `Config()` initially contains no species; add them with `add_species()` before initializing a simulation.

### 2. `State`

Holds the evolving arrays and simulation state:

- Radial edges and shell midpoints: `r`, `rmid`.
- Enclosed species mass, density, and one-dimensional velocity dispersion squared: `m`, `rho`, `v2`.
- Diagnostics, including innermost total density (`rho0`) and the most recent transport/heating fractional energy change (`du_max`).
- Time and output counters: `t`, `dt`, `step_count`, `snapshot_index`.
- Characteristic physical scales in `char`, and the originating `Config` in `config`.
- A `run()` method to advance the system until a requested or configured stopping condition is reached.

Initialize a simulation with `State.from_config(config)`. Each State retains its Config by reference; use separate configurations for separate experiments. Loading a State from disk with `State.from_dir()` is **not implemented**. Saved results can be analyzed with the readers and plotting functions without constructing a State.

Array rows follow `state.labels`, ordered by decreasing particle mass. For `s` species and `N` shells, `r` and `m` have shape `(s, N+1)`, while `rmid`, `rho`, and `v2` have shape `(s, N)`. The evolving quantities are dimensionless; `state.char` holds their physical conversion scales. Pressure and specific internal energy are calculated as `rho * v2` and `1.5 * v2`, respectively.

Three plotting functions are exported alongside these classes and are also available as State methods:

- **`plot_time_evolution()`** plots a global diagnostic over time and can compare multiple simulations.
- **`plot_snapshots()`** compares saved snapshots of one simulation, with up to three profiles in separate panels.
- **`make_movie()`** animates saved profiles with fixed axis limits and optional insets. Rendering is parallel by default and requires `ffmpeg`.

Internally, the profile routines initialize the model, and the hydrostatic routines establish equilibrium. During evolution, the integrator coordinates energy transport, optional binary heating, and hydrostatic readjustment, then calls the output routines. The package layout below shows where these components live.

Use `help()` for documentation on individual classes, methods, and functions.

---

## Getting started

### Installation

Use Python 3.12 or later.

```bash
git clone https://github.com/yaronetokayer/multigtf.git
cd multigtf
python -m pip install -e .
```

The dependencies `numpy`, `scipy`, `numba`, `matplotlib`, and `tqdm` are installed automatically. Install Jupyter separately to run the demo notebook. Movie generation additionally requires an `ffmpeg` executable on PATH.

### Example usage

```python
from pathlib import Path
import multigtf as gtf

output = Path('simulation_output')
output.mkdir(parents=True, exist_ok=True)

config = gtf.Config()
config.io.base_dir = str(output)

profile = ('abg', {'alpha': 2.0, 'beta': 5.0, 'gamma': 0.0})
config.add_species('heavy', m_part=4.0, frac=0.2, init=profile)
config.add_species('light', m_part=1.0, frac=0.8, init=profile)

state = gtf.State.from_config(config)
state.run(steps=1000)
```

This is a short example to check the installation; the demo follows longer physical evolution. `m_part` is the individual particle mass in solar masses. The `frac` values are mass-fraction weights and must sum to one; the mass represented inside the finite grid also depends on the profiles and domain. For a single-species model, add just one species with `frac=1.0`.

Set profile and numerical options **before** calling `State.from_config()`, for example:

```python
config.spec['heavy'].init = ('abg', {'alpha': 2.0, 'beta': 5.0, 'gamma': 0.0})
config.grid.ngrid = 200
config.prec.eps_du = 1.0e-4
config.io.nlog = 25_000
```

Other stopping conditions can be used instead of a step count:

```python
state.run(stoptime=0.1)  # Advance by an additional 0.1 simulation time units.
state.run(rho0=1.0e4)    # Run until the innermost total density reaches this threshold.
```

Successive `run()` calls continue from the current state. Configured halting limits also apply. `state.reset()` restores the initial arrays and resets the counters using the existing model setup; create a fresh Config and State when changing species or structural parameters.

Outputs are written under `config.io.base_dir` in a directory named `ModelXXXXX`. The base directory must exist. If `model_no` is unset, the lowest unused number is selected on first access. Set `config.io.model_no` explicitly if needed, but reusing a model directory can overwrite previous results; the current `io.overwrite` flag does not prevent this.

### Plotting

```python
# Compare initial and final saved profiles.
state.plot_snapshots(snapshots=[0, -1], profiles=['rho', 'v2'])

# Plot the innermost total-density history and save the figure.
state.plot_time_evolution(quantity='rho0', filepath='rho0_vs_time.png')

# Optional: animate saved profiles without insets.
state.make_movie(profiles=['rho', 'v2'], insets=False, parallel=False)
```

Standalone functions accept a State, Config, or model number. Supply `base_dir` when using model numbers. After creating two simulations named `state1` and `state2`, compare them with:

```python
gtf.plot_time_evolution([state1, state2], quantity='rho0')
```

Available snapshot profiles are `rho`, `m`, `v2`, and `eta`. The standalone `plot_snapshots()` defaults to the initial snapshot; the State method defaults to the latest **saved** snapshot. The plotting functions read output files rather than unsaved live arrays. See the demo for additional quantities, movie options, and analysis of saved simulations.

---

## Output files

Each model directory contains:

| File | Contents |
| --- | --- |
| `model_metadata.txt` | Configuration and species definitions. |
| `char_params.txt` | Characteristic physical scales and normalized Coulomb logarithms. |
| `logfile.txt` | Progress and timestep statistics, including entries every `nlog` steps. |
| `profile_x.dat` | Total and per-species radial profiles; `x` is the snapshot index. |
| `snapshot_conversion.txt` | Snapshot index, simulation time, physical time in Gyr, and step count. |
| `time_evolution.txt` | Global and per-species diagnostics over time. |

Profile snapshots are triggered by changes in innermost density (`io.drho_prof`). Time-history output is triggered by density changes (`io.drho_tevol`) or changes in the species half-mass-radius spread (`io.dr50_tevol`). Initial and final output is also written. File headers identify the columns; radius columns in profile files contain base-10 logarithms.

---

## Package Layout

The main components inside the `multigtf/` package are shown below; `__init__.py` files are omitted.

```text
multigtf/
├── config.py                   # Config class and species management
├── state.py                    # State initialization, evolution, and user methods
│
├── parameters/                 # Input containers and characteristic scales
│   ├── char_params.py
│   ├── constants.py
│   ├── grid_params.py
│   ├── init_params.py
│   ├── io_params.py
│   ├── prec_params.py
│   ├── sim_params.py
│   └── spec_params.py
│
├── profiles/                   # Initial profiles and background potential
│   ├── abg.py
│   ├── bkg_pot.py
│   ├── etnfw.py
│   ├── exp.py
│   ├── king.py
│   ├── nfw.py
│   ├── truncated_nfw.py
│   └── profile_init_routines.py
│
├── evolve/                     # Time integration and physical updates
│   ├── integrator.py
│   ├── transport.py
│   ├── hydrostatic.py
│   ├── binaries.py
│   └── evaporate.py            # Prototype; not used by State.run
│
├── io/                         # Reading and writing simulation output
│   ├── read.py
│   └── write.py
│
├── plot/                       # Static figures and movies
│   ├── snapshot.py
│   └── time_evolution.py
│
└── util/                       # Shared numerical and interpolation helpers
    ├── calc.py
    ├── interpolate.py
    └── little_helpers.py
```

The repository's `examples/` directory contains the demo notebook and saved example outputs. `tests/` contains the automated tests.

## Next steps

The API described here is on `main`; multiple species and binary heating are already implemented. The following features remain unfinished:

- Restarting a saved simulation through `State.from_dir()`.
- Evaporation through `sim.evap` and the decaying Hernquist background `hernq_decay`.
- Central black holes, inner cavities, and thermal reservoirs, under development on the separate `central_bh` branch.

Use the demo and docstrings for the supported configuration options. Numerical settings and convergence should be checked for each scientific application; record the exact commit or release used for published results.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
