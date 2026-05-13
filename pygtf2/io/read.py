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
        Dictionary containing global columns, species sub-dictionaries,
        and 'model_id'.
    """
    # Read header
    with open(filepath, "r") as f:
        header = f.readline().strip().split()

    # Load data
    data = np.loadtxt(filepath, skiprows=1)

    # Handle single-row case
    if data.ndim == 1:
        data = data[np.newaxis, :]

    # Build dictionary dynamically
    result = {col: data[:, i] for i, col in enumerate(header)}

    # Preserve original structured species behavior
    result["species"] = {}

    for col in header:
        if col.startswith("rho_c["):
            label = col.split("[", 1)[1].rstrip("]")

            result["species"][label] = {
                "rho_c" : result[f"rho_c[{label}]"],
                "r01"   : result[f"r01[{label}]"],
                "r05"   : result[f"r05[{label}]"],
                "r10"   : result[f"r10[{label}]"],
                "r20"   : result[f"r20[{label}]"],
                "r50"   : result[f"r50[{label}]"],
                "r90"   : result[f"r90[{label}]"],
                "r50evo": result[f"r50evo[{label}]"],
            }

    # Preserve original dtype behavior for step
    if "step" in result:
        result["step"] = result["step"].astype(int)

    # Extract model_id from directory name
    model_dir = os.path.basename(os.path.dirname(filepath))
    if model_dir.lower().startswith("model"):
        try:
            result["model_id"] = int(
                model_dir.replace("Model", "").replace("model", "")
            )
        except ValueError:
            result["model_id"] = None
    else:
        result["model_id"] = None

    return result

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
        Dictionary mapping snapshot_conversion.txt column names to numpy arrays.
    """
    path = os.path.join(model_dir, "snapshot_conversion.txt")

    # Read file with column names from header
    data = np.genfromtxt(path, names=True, dtype=None, encoding=None)

    # Handle single-row case (genfromtxt returns 0-d structured array)
    if data.shape == ():
        data = np.array([data], dtype=data.dtype)

    # Convert structured array to dictionary
    result = {}
    for name in data.dtype.names:
        col = data[name]

        # Optional: cast integer-like columns
        if np.issubdtype(col.dtype, np.integer):
            result[name] = col.astype(int)
        else:
            result[name] = col

    return result

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
    idx = np.where(data['index'] == index)[0][0]
    if not phys:
        t = data['time'][idx]
    else:
        t = data['time_Gyr'][idx]

    return t

def extract_snapshot_data(filepath, add_time=True):
    """
    Extract data from a multi-species snapshot timestep file.
    """

    # Read first line
    with open(filepath, "r") as f:
        header = f.readline().strip().split()

    # Load data
    data = np.loadtxt(filepath, skiprows=1)

    # Handle single-row case
    if data.ndim == 1:
        data = data[np.newaxis, :]

    # Build dictionary dynamically
    raw = {col: data[:, i] for i, col in enumerate(header)}

    # Start output with non-species columns
    result = {
        col: values
        for col, values in raw.items()
        if "[" not in col and "]" not in col
    }

    # Add species dictionary
    result["species"] = {}

    # Detect and organize species columns
    for col, values in raw.items():
        if "[" in col and col.endswith("]"):
            quantity, label = col.split("[", 1)
            label = label.rstrip("]")

            if label not in result["species"]:
                result["species"][label] = {}

            result["species"][label][quantity] = values

    # Extract timestep number from filepath and get time
    if add_time:
        basename = os.path.basename(filepath)
        step = int(basename.replace("profile_", "").replace(".dat", ""))
        result["time"] = get_time_conversion(filepath, step)

    return result

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
        snap_idx = int(conv["index"][-1])
    else:
        snap_idx = int(snapshot)  # ensure int
        # sanity: ensure it exists in the table
        if snap_idx not in set(conv["index"].tolist()):
            # not fatal strictly, but helpful to warn early
            raise ValueError(
                f"Snapshot index {snap_idx} not present in snapshot_conversion.txt "
                f"(available: {conv['index'].tolist()})"
            )
    # Find the row in conv corresponding to snap_idx
    row_idx = int(np.where(conv["index"] == snap_idx)[0][0])
    step_val = int(conv["step"][row_idx])
    t_val    = float(conv["time"][row_idx])

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
