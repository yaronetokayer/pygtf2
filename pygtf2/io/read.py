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
    Extract time-evolution data from a pygtfcode time_evolution.txt file.

    Parameters
    ----------
    filepath : str
        Path to the time_evolution.txt file.

    Returns
    -------
    dict
        Dictionary with keys:
            't',                # (array) Time in code units
            't_phys',           # (array) Time in physical units (Gyr)
            'rho_c',            # (array) Central density in code units
            'rho_c_phys',       # (array) Central density in physical units (Msun/pc^3)
            'v_max',            # (array) Maximum circular velocity in code units
            'v_max_phys'        # (array) Maximum circular velocity in physical units (km/s)
            'kn_min',           # (array) Minimum Knudsen number
            'mintrel',          # (array) Minimum relaxation time in code units
            'mintrel_phys',     # (array) Minimum relaxation time in physical units (Gyr)
            'model_id'          # (int) Model number
    """
    data = np.loadtxt(filepath, usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9), skiprows=1)
    model_dir = os.path.basename(os.path.dirname(filepath))
    model_id = int(model_dir.replace("Model", ""))
    # Handle case where data is 1D (only one row)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    return {
        't': data[:, 0],
        't_phys': data[:, 1],
        'rho_c': data[:, 2],
        'rho_c_phys': data[:, 3],
        'v_max': data[:, 4],
        'v_max_phys': data[:, 5],
        'kn_min': data[:, 6],
        'mintrel': data[:, 7],
        'mintrel_phys': data[:, 8],
        'model_id': model_id
    }

def extract_snapshot_indices(model_dir):
    """
    Extract snapshot indices and times from snapshot_conversion.txt.

    Parameters
    ----------
    model_dir : str
        Path to the model directory.

    Returns
    -------
    dict
        Dictionary with keys 'step', 't_t0', and 't_Gyr' containing numpy arrays.
    """
    path = os.path.join(model_dir, "snapshot_conversion.txt")

    data = np.loadtxt(path, usecols=(0, 1, 2, 3), skiprows=1)
    if data.ndim == 1:
        # Only one row of data
        snapshot_index = np.array([int(data[0])])
        t_t0 = np.array([data[1]])
        t_Gyr = np.array([data[2]])
        step = np.array([int(data[3])])
    else:
        snapshot_index = data[:, 0].astype(int)
        t_t0 = data[:, 1]
        t_Gyr = data[:, 2]
        step = data[:, 3].astype(int)
    return {
        'snapshot_index': snapshot_index,
        't_t0': t_t0,
        't_Gyr': t_Gyr,
        'step_count': step
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
        Dictionary of numpy arrays with keys:
        'log_r', 'log_rmid', 'm', 'rho', 'v2', 'p', 'trel', 'kn', 'time'
    """
    data = np.loadtxt(filename, usecols=range(1, 9), skiprows=1)

    # Extract timestep number from filename and get time
    basename = os.path.basename(filename)
    step = int(basename.replace("profile_", "").replace(".dat", ""))
    t = get_time_conversion(filename, step)

    return {
        'log_r': data[:, 0],
        'log_rmid': data[:, 1],
        'm': data[:, 2],
        'rho': data[:, 3],
        'v2': data[:, 4],
        'p': data[:, 5],
        'trelax': data[:, 6],
        'kn': data[:, 7],
        'time': t
    }

def import_metadata(model_dir: Union[Path, str]) -> Dict[str, Dict[str, Any]]:
    """
    Read `<model_dir>/model_metadata.txt` and return a dict-of-dicts.

    Expected file structure (YAML-like):
        <header line>
        ========================================
        section_name:
            key: value
            key: value
        other_section:
            key: value
            ...

    Returns
    -------
    dict
        e.g. {
          "_init": {...},
          "grid": {...},
          "io": {...},
          "prec": {...},
          "sim": {...},
        }

    Raises
    ------
    FileNotFoundError
        If model_dir or model_metadata.txt is missing.
    ValueError
        If the file is malformed (e.g., a key-value outside a section).
    """
    pdir = Path(model_dir)
    if not pdir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {pdir}")

    meta_path = pdir / "model_metadata.txt"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")
    
    # Parse
    out: Dict[str, Dict[str, Any]] = {}
    current_section: Optional[str] = None

    with meta_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")

            # Skip blank lines and separators
            if not line.strip():
                continue
            if set(line.strip()) == {"="}:  # e.g., "====="
                continue

            # Skip the header line like "Model 004 Metadata"
            if line.startswith("Model ") and line.endswith(" Metadata"):
                continue

            # Section header? (e.g., "grid:" / "_init:")
            stripped = line.strip()
            if stripped.endswith(":") and (":" not in stripped[:-1]):
                # New section
                current_section = stripped[:-1]  # drop trailing colon
                out[current_section] = {}
                continue

            # Expect key-value inside a section, like "    _ngrid: 200"
            if current_section is None:
                # If someone placed a key: value outside any section
                raise ValueError(
                    f"Malformed metadata: found key-value outside a section near line: {line!r}"
                )

            # Must contain a colon to split key and value
            if ":" not in line:
                # Allow odd lines to be ignored? Safer to error out:
                raise ValueError(
                    f"Malformed metadata: expected 'key: value' in section '{current_section}', got: {line!r}"
                )

            # Split on the first colon only, to allow paths with colons (rare) later
            key_part, value_part = line.split(":", 1)
            key = key_part.strip()
            value = _coerce_scalar(value_part)

            out[current_section][key] = value

    return out

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
          - 'current_time_Gyr'     : float (Gyr, last row)
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
