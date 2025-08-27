# pygtf2

**pygtf2** is a modification and extension of the [**pygtfcode** package](https://github.com/yaronetokayer/pygtfcode).  It is a modern Python implementation of a 1D Lagrangian gravothermal fluid code that can handle mutiple species and simulate mass segregation due to conduction.

This implementation extends the formalism outlined in [Zhong & Shapiro (2025)](https://journals.aps.org/prd/abstract/10.1103/z1fy-2p8s), with modular components for initialization, evolution, and output.

See the `./examples/demo.ipynb` for more comprehensive guidance on using this module.

Contact yarone.tokayer-at-yale.edu with any questions/comments.

NOTE: THIS IS UNDER CONSTRUCTION AND NOT READY FOR USE

---

## Overview

The code is organized around **two user-facing classes**:

### 1. `Config`

Stores static input parameters, grouped into modules:

* `io`: Output paths and snapshot cadence
* `grid`: Grid resolution and domain
* `init`: Initial profile (NFW, truncated NFW, or $\alpha$-$\beta$-$\gamma$)
* `sim`: Physical parameters and simulation option (e.g., self-interaction cross section, calibrated parameters)
* `prec`: Precision tolerances and iteration limits

### 2. `State`

Holds the dynamically evolving quantities:

* Radial grid and shell midpoints: `r`, `rmid`
* Physical variables: `m`, `rho`, `p`, `u`, `v2`, `kn`, `trelax`
* Diagnostic quantities: `mintrel`, `maxvel`, `minkn`, `du_max`, `dr_max`
* Time tracking: `t`, `dt`, `step_count`, `snapshot_index`
* Characteristic scales (derived from `Config`)
* A `run()` method to evolve the system until collapse or a stopping condition is reached

A `State` object can be constructed in one of two ways:
1. From a `Config` object.  The `Config` object then becomes an attribute of `State`.  In this way, multiple `State`s can be instantiated with different `Config` objects.
2. From the directory of a simulation that has already been initialized.  This is useful for either analyzing or resuming a simulation from a different Python session.  No `Config` object is needed.  By default the state will initialize to the latest snapshot of the imported simulation.

Note that `r` and `m` define bin edges, while `rho`, `p`, `u`, `v2`, `kn`, and `trelax` are all defined at `rmid`.  Therefore, `r` and `m` are longer than the other arrays by one entry.

In addition to these two classes, there are three plotting functions that are automatically imported, each of which also exists as a method of `State`:

### 1. `plot_time_evolution()`

Plot the time evolution of any one of a number of system-wide quantities.  Multiple simulations can be passed to compare their evolutions on the same plot.

### 2. `plot_snapshots()`

Plot up to three profiles of a simulation or multiple simulations at specified points in time.

### 3. `make_movie()`

Animate the full evolution of up to three profiles.  This required ffmpeg to be installed and callable with `ffmpeg` from the working directory.

Use `help()` for documentation on any of these classes, methods, and functions.

---

## Getting started

### Installation

```bash
git clone https://github.com/yaronetokayer/pygtfcode.git
cd pygtfcode
pip install -e .
```

Dependencies: Python 3.8+, `numpy`, `scipy`, `numba`, `matplotlib`, `tqdm`

### Example usage

```python
import pygtfcode as gtf

config = gtf.Config()
state = gtf.State.from_config(config)
state.run()
```

To import an existing simulation:

```python
import pygtfcode as gtf

state = gtf.State.from_dir(model_dir='/path/to/Model002')

# The latest snapshot will be loaded by default.  To load a particular snapshot:
state = gtf.State.from_dir(model_dir='/path/to/Model002', snapshot=59)
```

Alternatively, you can call `from pygtfcode import Config, State`.  In that case, plotting functions will not be automatically imported.

We can also run for a specified duration:
```python
state.run(steps=100) # Run for 100 simulation steps
state.run(time=55.0) # Run for a duration of 55.0 simulation time units
state.run(rho_c=500.0) # Run until the central density exceeds 500.0
```

Halting criteria in `config` override these conditions.  multiple `run()` commands can be executed in succession, and each will continue from the current state.  Call `state.reset()` to reset the state to its initial condition and reset the set counter.

To customize defaults:

```python
# Customize initial profile
config.init = "abg"                                 # Use ABG with default params
config.init = ("abg", {"alpha": 3.5, "beta": 4.5})  # Custom ABG

# Customize other configuration parameters
config.grid.ngrid = 200
config.io.model_no = 42
config.io.base_dir = "/tmp/sims"                    # Default is the current working directory
config.sim.sigma_m = 1.0

# Switch to a truncated NFW
config.init = ("truncated_nfw", {"Zt": 0.05, "deltaP": 1e-4})

# Turn off chatter
config.io.chatter = False
```

If you don’t explicitly assign `config.io.model_no`, it is automatically set to the next available model number in `config.io.base_dir` (e.g., if `Model000`, `Model001`, and `Model002` directories exist, it will assign `model_no = 3`) **when a `State` is instantiated**. You can explicity assign a `model_no` with `config = gtf.Config(model_no=5)` or with `config.io.model_no = 5` once config is instantiated.  Note that in that case, outputs from a previous simulations with the same `model_no` will be overridden.

### Plotting

There are three plotting functions that are imported with the `pygtfcode` package:

The `plot_time_evolution()` function plots the evolution of system-wide parameters over time.  It can plot any of the columns in the `time_evolution.txt` output.

```python
import pygtfcode as gtf

# Compare the central density evolution of two different simulations
# The default plot is `rho_c` when no quantity is specified
gtf.plot_time_evolution(state1, state2)

# Alternatively, the simulations can be called by their Config objects or by their model numbers
# v_max_phys returns the max velocity in physical units (km/s)
gtf.plot_time_evolution(config1, config2, quantity="v_max_phys", ylabel=r"Custom ylabel")

# base_dir needs to be specified if simulations are called by model number:
gtf.plot_time_evolution(5, 6, quantity="kn_min", base_dir='./') # This is useful for simulations run in a different session

# The plot can be saved to a file
# Use 'show' to show the figure in standard output as well
gtf.plot_time_evolution(state1, filepath='./rho_c_vs_time.png', show=True)
```

The `plot_snapshots()` function plots up to three profiles in separate panels for one or multiple snapshots of the simulation.  Snapshots are specified by the index of the `snapshot_x.dat` file.  Like `plot_time_evolution()`, the State object, Config object, or model number can used to specify the simulation you wish to plot.

```python
import pygtfcode as gtf

# Plot the initial density profile
gtf.plot_snapshots(state)

# Plot the mass profile at a specified snapshot
gtf.plot_snapshots(config, snapshots=50, profiles='m')

# Plot the density, v^2, and Knudsen number profiles, comparing several snapshots
gtf.plot_snapshots(4, snapshots=[0, 50, 100], profiles=['rho', 'v2', 'kn'], base_dir='./')

# The plot can be saved to a file
# Use 'show=True' to show the figure in standard output as well
gtf.plot_snapshots(state, filepath='./initial_rho.png')
```

The `make_movie()` function generates animations of up to three profiles in separate panels for a simulation.  Like `plot_time_evolution()`, the State object, Config object, or model number can used to specify the simulation you wish to plot.  Only the snapshots of the most recent run for the simulation model_no will be included, even if profiles with higher indices are in the directory from previous simulation runs.  You can check the current version of the `snapshot_conversion.txt` file for all snapshots that will be included in the animation

```python
import pygtfcode as gtf

# Plot the density profile
gtf.make_movie(state)

# All keyword arguments available in plot_snapshots function, other than 'snapshots', can be used here
gtf.make_movie(2, base_dir='./', profiles=['v2', 'p'], grid=True)
```

The plotting functions also exist as methods to the `State` object:

```python
import pygtfcode as gtf

config = gtf.Config()
state = gtf.State.from_config(config)
state.run()

state.plot_time_evolution()         # Accepts all keyword arguments, but cannot compare between simulations when used this way

state.plot_snapshots()              # Defaults to the latest state
state.plot_snapshots(snapshots=0, filename="./initial_profs.png")   # Plot and save initial profiles
# Note that while the standalone function defaults to the initial profile, the `State` method defaults to the current state.

state.make_movie(profiles=['rho', 'kn', 'v2'])
```

---

## Output files

All outputs are written to the directory specified by `config.io.base_dir` and `model_no`.

### 1. `model_metadata.txt`

Stores all information about the simulation model for reference.  Unpacks all attributes of the `Config` object that instantiated the `State`.

### 2. `logfile.txt`

Logs relevant quantities every `nlog` steps (set in `config.io`).  If `chatter` is set to `True`, then these are also output to the console.

### 3. `profile_x.dat`

Radial profiles of all fluid variables, written each time the central density changes by a fractional amount `drho_prof` (set in `config.io`). The suffix `x` is the snapshot index.  `snapshot_conversion.txt` stores the conversion between the snapshot index `x` and simulation time.

Each row contains:

```
i   log(r_i)   log(rmid_i)   m_i   rho_i   v2_i   trelax_i   kn_i
```

### 4. `time_evolution.txt`

Records the time evolution of relevant quantites, written each time the central density changes by a fractional amount `drho_tevol` (set in `config.io`).

---

## Package Layout

```
pygtfcode/
├── config.py               # Defines 'Config' class
├── state.py                # Defines 'State' class
│
├── parameters/             # Parameter subclasses for 'Config' attributes
│   ├── char_params.py
│   ├── constants.py
│   ├── grid_params.py
│   ├── init_params.py
│   ├── io_params.py
│   ├── prec_params.py
│   └── sim_params.py
│
├── profiles/               # Profile specific tools to set initial conditions of 'State'
│   ├── abg.py
│   ├── nfw.py
│   ├── truncated_nfw.py
│   └── profile_routines.py
│
├── evolve/                 # Integration and solver, used by 'State' methods
│   ├── integrator.py
│   ├── transport.py
│   └── hydrostatic.py
│
├── io/                     # I/O routines
│   └── write.py
│   └── read.py
│
├── plot/                   # Plotting routines
│   ├── time_evolution.py
│   └── snapshot.py
```

---

## Next steps

* v1.0 is complete, v2.0 in development
* v2.0 will be able to accomodate multiple species

---

## License

MIT License. See [LICENSE](./LICENSE) for details.
