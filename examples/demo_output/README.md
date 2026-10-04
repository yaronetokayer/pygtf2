# Saved demo simulations

These five models are the outputs of the executed `../demo.ipynb`. They replace the obsolete single-fluid examples formerly stored in `examples/Model000`–`Model002`.

| Model | Experiment | Final steps | Final simulation time | Final central density |
| --- | --- | ---: | ---: | ---: |
| Model00017 | Two-species Plummer model, particle mass ratio 2 | 70,731 | 9.101485 | 10,000.46 |
| Model00018 | Mass ratio 1.5, matched physical duration | 27,728 | 6.944759 | 30.24516 |
| Model00019 | Binary heating, 30,000 particles, post-collapse oscillations | 1,840,297 | 9.200002 | 46,114.43 |
| Model00020 | Static Hernquist background | 14,105 | 1.000081 | 5.042355 |
| Model00021 | Refined main model: 240 shells and half the energy-change tolerance | 141,552 | 9.101428 | 10,000.49 |

Times and densities in the table are dimensionless code quantities. Characteristic scales are in each model's `char_params.txt`; time conversion to Gyr is also recorded in `snapshot_conversion.txt`. The binary-heating model has a different total mass from the main model, consistently chosen for its particle count. It is a separate oscillation demonstration, not a matched heating-only comparison.

Each model contains:

- `model_metadata.txt`: configuration used for the run.
- `char_params.txt`: characteristic scales.
- `time_evolution.txt`: global and species diagnostics.
- `snapshot_conversion.txt`: snapshot index, simulation time, time in Gyr, and integration step.
- `profile_*.dat`: saved radial profiles.
- `logfile.txt`: numerical integration log.

The main model also includes `initial.png` and `compare.png`. Movie encoding is opt-in in the notebook, so these examples contain no MP4 files.

## Plot the examples without rerunning them

From the repository root, with `pygtf2` installed:

```python
from pathlib import Path
import pygtf2 as gtf

outputs = Path('examples/demo_output')
gtf.plot_time_evolution(19, base_dir=str(outputs), quantity='rho0', grid=True)
gtf.plot_snapshots(19, base_dir=str(outputs), snapshots=[0, -1],
                   profiles=['rho', 'v2', 'eta'], grid=True)
```

Use the notebook's zoomed post-collapse plot to resolve the binary-heating oscillations. The metadata retains the original machine's output path as run provenance; pass your own `base_dir` when reading or plotting. If reconstructing a Config from metadata, replace its `io` settings before creating a new State. `State.from_dir()` is currently disabled.

Rerunning the notebook chooses unused model numbers. Other generated model directories are ignored by Git; the five examples above are explicitly retained. The full output set is approximately 71 MiB on disk.
