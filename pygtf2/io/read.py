import os
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, Union

def _coerce_scalar(s: str) -> Union[bool, int, float, str]:
    """
    Convert a string token into bool/int/float if possible; otherwise return original string.
    """
    t = s.strip()
    if t == "":
        return ""  # allow blank values if they ever appear
    # Booleans
    if t == "True":
        return True
    if t == "False":
        return False
    # Int
    try:
        # Ensure ints like "004" are preserved as int 4 (expected)
        i = int(t)
        return i
    except ValueError:
        pass
    # Float (incl. scientific notation)
    try:
        f = float(t)
        return f
    except ValueError:
        pass
    # String fallback (e.g., paths)
    return t

def extract_time_evolution_data(filepath):
    """
    Extract time-evolution data from a pygtf2 time_evolution.txt file.

    Parameters
    ----------
    filepath : str
        Path to the time_evolution.txt file.

    Returns
    -------
    dict
        {
          'step'     : array,
          'time'     : array,
          'rho_c_tot': array,
          'v_max'    : array,
          'mintrel'  : array,
          'species'  : {
              'dm': {
                  'rho_c' : array,
                  'r01' : array,
                  'r05' : array,
                  'r10': array,
                  'r20': array,
                  'r50': array,
                  'r90': array,
              },
              'stars': { ... },
              ...
          },
          'model_id' : int
        }
    """
    # Parse header to get column names
    with open(filepath, "r") as f:
        header_line = f.readline().strip()
    colnames = header_line.split()

    # Load numeric data
    data = np.loadtxt(filepath, skiprows=1)

    # Handle case where data is 1D (only one row)
    if data.ndim == 1: 
        data = data[np.newaxis, :]

    # Map name -> index
    idx = {name: j for j, name in enumerate(colnames)}

    # Global quantities
    out = {
        'step'     : data[:, idx['step']].astype(int),
        'time'     : data[:, idx['time']],
        'rho_c_tot': data[:, idx['rho_c_tot']],
        'v_max'    : data[:, idx['v_max']],
        'mintrel'  : data[:, idx['mintrel']],
        'species'  : {},
    }

    # Per-species blocks
    for name in colnames:
        if name.startswith('rho_c['):
            label = name.split('[',1)[1].rstrip(']')
            out['species'][label] = {
                'rho_c' : data[:, idx[f'rho_c[{label}]']],
                'r01' : data[:, idx[f'r01[{label}]']],
                'r05' : data[:, idx[f'r05[{label}]']],
                'r10': data[:, idx[f'r10[{label}]']],
                'r20': data[:, idx[f'r20[{label}]']],
                'r50': data[:, idx[f'r50[{label}]']],
                'r90': data[:, idx[f'r90[{label}]']],
            }

    # Model ID from directory name
    model_dir = os.path.basename(os.path.dirname(filepath))
    if model_dir.lower().startswith("model"):
        try:
            out['model_id'] = int(model_dir.replace("Model", "").replace("model",""))
        except ValueError:
            out['model_id'] = None
    else:
        out['model_id'] = None

    return out

def extract_snapshot_indices(model_dir):
    """
    Extract snapshot indices and times from snapshot_conversion.txt.

    Expected columns (with header):
    index, time, time_Gyr, step

    Parameters
    ----------
    model_dir : str
        Path to the model directory.

    Returns
    -------
    dict with:
      'snapshot_index' : ndarray[int]
      't'              : ndarray[float]  # time in code units
      't_Gyr'          : ndarray[float]  # time in Gyr
      'step_count'     : ndarray[int]
    """
    path = os.path.join(model_dir, "snapshot_conversion.txt")

    data = np.loadtxt(path, skiprows=1)
    if data.ndim == 1:
        data = data[np.newaxis, :]

    # Expect exactly 4 columns: index, time, time_Gyr, step
    if data.shape[1] != 4:
        raise ValueError(
            f"snapshot_conversion.txt should have 4 columns; found {data.shape[1]}"
        )

    snapshot_index = data[:, 0].astype(int)
    t              = data[:, 1]
    t_Gyr          = data[:, 2]
    step_count     = data[:, 3].astype(int)

    return {
        'snapshot_index': snapshot_index,
        't_t0': t,
        't_Gyr': t_Gyr,
        'step_count': step_count,
    }

def get_time_conversion(filepath, index, phys=False):
    """
    Get conversion from index to time from snapshot_conversion.txt.

    Parameters
    ----------
    filepath : str
        Path to the profile_x.dat file.
    index : int
        Snapshot index at which to get time
    phys : bool
        If True, get value in Gyr, otherwise in simulation units

    Returns
    -------
    float
        Time value.
    """
    # Find corresponding timestep.log in the same ModelXXX directory
    model_dir = os.path.dirname(filepath)

    data = extract_snapshot_indices(model_dir)

    # Lookup time
    idx = np.where(data['snapshot_index'] == index)[0][0]
    if not phys:
        t = data['t_t0'][idx]
    else:
        t = data['t_Gyr'][idx]

    return t

def extract_snapshot_data(filename):
    """
    Extract data from a snapshot timestep file.

    Parameters
    ----------
    filename : str
        Path to the timestep_*.dat file.

    Returns
    -------
    dict
        {
          'log_r'    : array,
          'log_rmid' : array,
          'm_tot'    : array,
          'rho_tot'  : array,
          'p_tot'    : array,
          'species'  : {
              'dm': {
                  'lgr'   : array,
                  'lgrm'  : array,
                  'm'     : array,
                  'rho'   : array,
                  'v2'    : array,
                  'p'     : array,
                  'trelax': array,
              },
              'stars': { ... }
          },
          'time' : float (from snapshot_conversion.txt via get_time_conversion)
        }
    """
    # Read header line
    with open(filename, "r") as f:
        header_line = f.readline().strip()
    colnames = header_line.split()

    # Load numeric data
    data = np.loadtxt(filename, skiprows=1)
    if data.ndim == 1:
        data = data[np.newaxis, :]

    # Map name -> column
    idx = {name: j for j, name in enumerate(colnames)}

    out = {
        'log_r'   : data[:, idx['log_r']],
        'log_rmid': data[:, idx['log_rmid']],
        'm_tot'   : data[:, idx['m_tot']],
        'rho_tot' : data[:, idx['rho_tot']],
        'p_tot'   : data[:, idx['p_tot']],
        'species' : {},
    }

    # Detect species blocks
    for name in colnames:
        if name.startswith('lgr['):
            label = name.split('[',1)[1].rstrip(']')
            out['species'][label] = {
                'lgr'   : data[:, idx[f'lgr[{label}]']],
                'lgrm'  : data[:, idx[f'lgrm[{label}]']],
                'm'     : data[:, idx[f'm[{label}]']],
                'rho'   : data[:, idx[f'rho[{label}]']],
                'v2'    : data[:, idx[f'v2[{label}]']],
                'p'     : data[:, idx[f'p[{label}]']],
                'trelax': data[:, idx[f'trelax[{label}]']],
            }

    # Extract timestep number from filename and get time
    basename = os.path.basename(filename)
    step = int(basename.replace("profile_", "").replace(".dat", ""))
    t = get_time_conversion(filename, step)
    out['time'] = t

    return out

def import_metadata(model_dir: Union[Path, str]) -> Dict[str, Dict[str, Any]]:
    """
    Read `<model_dir>/model_metadata.txt` into a nested dict by indentation.
    Supports arbitrary nesting like:

        _mtot: 3e9
        grid:
            _ngrid: 200
        spec:
            dm:
                _frac: 1.0
                _init:
                    profile: abg
                    _r_s: 1.0
            stars:
                _frac: 0.8
                ...

    Returns
    -------
    dict
    """
    pdir = Path(model_dir)
    if not pdir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {pdir}")

    meta_path = pdir / "model_metadata.txt"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")

    root: Dict[str, Any] = {}
    # Stack holds tuples of (indent_level, current_dict)
    stack: list[tuple[int, Dict[str, Any]]] = [(0, root)]

    def current_dict_for(indent: int) -> Dict[str, Any]:
        # Pop until top of stack has indent <= requested
        while stack and stack[-1][0] >= 0 and stack[-1][0] >= indent + 1:
            stack.pop()
        # Now ensure the parent dict on stack has indent <= requested
        while len(stack) > 1 and stack[-1][0] > indent:
            stack.pop()
        return stack[-1][1]

    with meta_path.open("r", encoding="utf-8") as f:
        for raw in f:
            # normalize tabs to 4 spaces
            line = raw.expandtabs(4).rstrip("\n")

            # Skip blanks and separators
            if not line.strip():
                continue
            if set(line.strip()) == {"="}:
                continue
            # Skip header like "Model 000 Metadata"
            if line.strip().startswith("Model ") and line.strip().endswith("Metadata"):
                continue

            # Determine indentation (count leading spaces)
            indent = len(line) - len(line.lstrip(" "))
            stripped = line.strip()

            # Sanity: require indentation in non-negative multiples
            if indent < 0:
                raise ValueError(f"Negative indentation? line: {line!r}")

            # SECTION (ends with ":" and only one colon)
            if stripped.endswith(":") and (":" not in stripped[:-1]):
                key = stripped[:-1].strip()
                parent = current_dict_for(indent)
                # Create or reuse a dict at this key
                new_container = parent.get(key)
                if not isinstance(new_container, dict):
                    new_container = {}
                    parent[key] = new_container
                stack.append((indent + 1, new_container))
                continue

            # KEY: VALUE line (split on first colon)
            if ":" not in stripped:
                raise ValueError(f"Malformed metadata line (no colon): {line!r}")

            key_part, value_part = stripped.split(":", 1)
            key = key_part.strip()
            value_str = value_part.strip()
            if value_str == "None":
                value = None
            else:
                value = _coerce_scalar(value_part)

            target = current_dict_for(indent)
            target[key] = value

    return root

def load_snapshot_bundle(model_dir: Union[str, Path], snapshot: Optional[int] = None) -> Dict[str, Any]:
    """
    Load one snapshot's arrays (via extract_snapshot_data) and add *current* run info
    from the last row of snapshot_conversion.txt.

    Parameters
    ----------
    model_dir : str | Path
        Path to the model directory.
    snapshot : int or None
        Snapshot index to load. If None, loads the latest snapshot in snapshot_conversion.txt.

    Returns
    -------
    dict
        Includes everything from extract_snapshot_data(profile_<idx>.dat) plus:
          - 'snapshot_index'       : int  (the index that was loaded)
          - 'current_step_count'   : int  (last row of snapshot_conversion.txt)
          - 'current_time'         : float (simulation units, last row)
    """
    # Basic checks
    pdir = Path(model_dir)
    if not pdir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {pdir}")

    # Load the conversion table
    conv = extract_snapshot_indices(str(pdir))

    # Choose which snapshot to load
    if snapshot is None:
        # Latest snapshot is the last entry in the table
        snap_idx = int(conv["snapshot_index"][-1])
    else:
        snap_idx = int(snapshot)  # ensure int
        # sanity: ensure it exists in the table
        if snap_idx not in set(conv["snapshot_index"].tolist()):
            # not fatal strictly, but helpful to warn early
            raise ValueError(
                f"Snapshot index {snap_idx} not present in snapshot_conversion.txt "
                f"(available: {conv['snapshot_index'].tolist()})"
            )
    # Find the row in conv corresponding to snap_idx
    row_idx = int(np.where(conv["snapshot_index"] == snap_idx)[0][0])
    step_val = int(conv["step_count"][row_idx])
    t_val    = float(conv["t_t0"][row_idx])

    # Resolve the profile file path and load its arrays
    profile_path = pdir / f"profile_{snap_idx}.dat"
    if not profile_path.is_file():
        raise FileNotFoundError(f"Snapshot file not found: {profile_path}")
    
    # Load arrays/time for the chosen snapshot
    snap_payload = extract_snapshot_data(str(profile_path))

    out: Dict[str, Any] = dict(snap_payload)
    out.update(
        {
            "snapshot_index": snap_idx,
            "step_count": step_val,
            "time": t_val
        }
    )
    return out
