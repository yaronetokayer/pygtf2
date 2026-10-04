import numpy as np
import os
from pygtf2.util.calc import mass_fraction_radii, calc_r_c, calc_v2_c, calc_core_species_quantities, CORE_HALF_RHO0, CORE_SPITZER87
from pygtf2.util.interpolate import sum_extensive_loglog_single, sum_intensive_loglog_single

def make_dir(state):
    """
    Create the model directory if it doesn't exist.

    Arguments
    ---------
    state : State
        The current simulation state.
    """
    model_dir = state.config.io.model_dir
    base_dir = state.config.io.base_dir

    full_path = os.path.join(base_dir, model_dir)
    
    if not os.path.exists(full_path):
        os.makedirs(full_path)
        if state.config.io.chatter:
            print(f"Created directory: {full_path}")
    else:
        if state.config.io.chatter:
            print(f"Directory already exists: {full_path}")

def write_metadata(state):
    """
    Write model metadata to disk for reference.

    Arguments
    ---------
    state : State
        The current simulation state.
    """
    io = state.config.io
    filename = os.path.join(io.base_dir, io.model_dir, f"model_metadata.txt")

    def dump_container(val, indent=0, key_name=None):
        pad = " " * indent
        lines = []

        def line(s): lines.append(pad + s)

        # Dicts: print a header, then sorted keys
        if isinstance(val, dict):
            if key_name is not None:
                line(f"{key_name}:")
            for k in sorted(val.keys()):
                lines.extend(dump_container(val[k], indent + 4, key_name=str(k)))
            return lines

        # Lists / tuples: index each item
        if isinstance(val, (list, tuple)):
            if key_name is not None:
                line(f"{key_name}:")
            for i, item in enumerate(val):
                lines.extend(dump_container(item, indent + 4, key_name=f"[{i}]"))
            return lines

        # Objects with attributes: recurse into their __dict__
        if hasattr(val, "__dict__"):
            if key_name is not None:
                line(f"{key_name}:")
            # Sort attribute names for determinism
            for attr in sorted(vars(val).keys()):
                attr_val = getattr(val, attr)
                lines.extend(dump_container(attr_val, indent + 4, key_name=attr))
            return lines

        # Scalars / everything else
        if key_name is not None:
            line(f"{key_name}: {val}")
        else:
            line(str(val))
        return lines

    with open(filename, "w") as f:
        f.write(f"Model {io.model_no:05d} Metadata\n")
        f.write("=" * 40 + "\n\n")
        for line in dump_container(state.config):
            f.write(line + "\n")

    if io.chatter:
        print(f"Model information written to model_metadata.txt")

def write_char_params(state):
    """
    Write characteristic parameters to a file.
    """
    char = state.char
    io = state.config.io
    filename = os.path.join(io.base_dir, io.model_dir, "char_params.txt")

    names = []
    values = []

    for key, value in char.__dict__.items():

        if key == "lnL" and value is not None:
            lnL = np.asarray(value, dtype=float)

            if lnL.ndim != 2 or lnL.shape[0] != lnL.shape[1]:
                raise ValueError("char.lnL must be a square 2D matrix.")

            # Optional sanity check
            if not np.allclose(lnL, lnL.T, equal_nan=True):
                raise ValueError("char.lnL is expected to be symmetric.")

            n = lnL.shape[0]

            # Upper triangle including diagonal
            for i in range(n):
                for j in range(i, n):
                    names.append(f"lnL_{i}_{j}")
                    values.append(float(lnL[i, j]))

        else:
            names.append(key)
            values.append(np.nan if value is None else float(value))

    col_width = 18
    header = "".join(f"{name:>{col_width}}" for name in names)
    fmt = f"%{col_width}.8e"

    np.savetxt(
        filename,
        [values],
        header=header,
        fmt=fmt,
        delimiter="",
        comments=""
    )

    if state.config.io.chatter:
        print("Characteristic parameters written to char_params.txt")

def write_log_entry(state, start_step):
    """ 
    Append a line to the simulation log file.
    Overwrites any lines with step_count >= current step_count.

    Arguments
    ---------
    state : State
        The current simulation state.
    start_step : int
        The starting value of the current simulation run
    """
    config = state.config
    io = config.io
    prec = config.prec
    filepath = os.path.join(io.base_dir, io.model_dir, f"logfile.txt")
    chatter = io.chatter
    step = state.step_count
    nlog = io.nlog
    if ( step - start_step ) % nlog != 0:
        nlog = ( step - start_step ) % nlog

    # Build header (base columns up to <du lim>)
    header_cols = [
        f"{'step':>10}",
        f"{'time':>12}",
        f"{'<dt>':>12}",
        f"{'rho0':>12}",
        f"{'r50_spread':>10}",
        f"{'<du lim>':>8}",
        f"{'<n_iter_du>':>11}",
    ]

    header = "  ".join(header_cols) + "\n"

    # Build data row
    if step == start_step:
        # On the first step, some averaged/limiting quantities are N/A.
        row = [
            f"{step:10d}",
            f"{state.t:12.6e}",
            f"{'N/A':>12}",                 # <dt>
            f"{state.rho0:12.6e}",
            f"{state.r50_spread:10.4e}",
            f"{'N/A':>8}",                  # <du lim>
            f"{'N/A':>11}",                  # <n_iter_du>
        ]

    else:
        # Normal numeric row
        row = [
            f"{step:10d}",
            f"{state.t:12.6e}",
            f"{state.dt_cum / nlog:12.6e}",
            f"{state.rho0:12.6e}",
            f"{state.r50_spread:10.4e}",
            f"{state.du_max_cum / prec.eps_du / nlog:8.2e}",
            f"{state.n_iter_du / nlog:11.5e}"
        ]

    new_line = "  ".join(row) + "\n"

    _update_file(filepath, header, new_line, step)

    state.n_iter_du = 0
    state.dt_cum = 0.0
    state.du_max_cum = 0.0

    if chatter:
        if step == 0:
            print("Log file initialized:")
        if step == start_step:
            print(header[:-1])
        print(new_line[:-1])

def write_profile_snapshot(state, initialize=False):
    """
    Write dimensionless radial profiles and update the snapshot table.

    Totals are interpolated onto a shared radial grid; each species retains
    its own grid. Columns are i, log_r, log_rmid, m_tot, rho_tot, v2_tot,
    eta, followed by lgr[label], lgrm[label], m[label], rho[label], and
    v2[label] for each species. Radius columns are base-10 logarithms.
    eta is masked outside the species' common radial overlap.

    Parameters
    ----------
    state : State
        State to save at its current snapshot_index. The corresponding file
        is replaced; when initialize=False, higher-index profiles are removed.
    initialize : bool, optional
        If True, keep snapshot_index unchanged. Otherwise increment it after
        writing. The conversion table is updated in either case.
    """
    io = state.config.io
    filename = os.path.join(io.base_dir, io.model_dir, f"profile_{state.snapshot_index}.dat")

    # If not initializing, remove any higher-index snapshot files
    if not initialize:
        snapshot_dir = os.path.join(io.base_dir, io.model_dir)

        for fname in os.listdir(snapshot_dir):
            if not fname.startswith("profile_") or not fname.endswith(".dat"):
                continue

            try:
                idx = int(fname[len("profile_"):-len(".dat")])
            except ValueError:
                continue  # ignore unexpected files

            if idx > state.snapshot_index:
                os.remove(os.path.join(snapshot_dir, fname))

    # Use species 0 for the r columns.
    r = state.r
    rmid = state.rmid
    s, Np1 = r.shape
    N = Np1 - 1
    labels = list(state.labels)
    s = len(labels)
    m_part = state.m_part

    # Compute totals on the fly
    from pygtf2.util.interpolate import sum_intensive_loglog, sum_extensive_loglog, interp_intensive_loglog
    from pygtf2.util.calc import compute_eta_multi
    r_tot = np.zeros((Np1,))
    r_tot_min = np.min(r[:,1:])
    r_tot_max = np.max(r[:,1:])
    r_tot[1:] = np.geomspace(r_tot_min, r_tot_max, num=N, endpoint=True)
    r_totmid = 0.5 * (r_tot[1:] + r_tot[:-1])

    m_tot = sum_extensive_loglog(r_tot, r, state.m)
    rho_tot = sum_intensive_loglog(r_totmid, rmid, state.rho)
    p_tot = sum_intensive_loglog(r_totmid, rmid, state.rho * state.v2,)
    v2_tot = p_tot / rho_tot
    eta = np.zeros(N, dtype=np.float64)
    if s > 1 and np.max(m_part) > np.min(m_part):
        v2_interp = interp_intensive_loglog(r_totmid, rmid, state.v2)
        err = np.zeros(N, dtype=np.float64)
        compute_eta_multi(m_part, np.sqrt(v2_interp), eta, err)
    
    # apply eta overlap mask
    r_min_k = np.min(r[:, 1:], axis=1)
    r_max_k = np.max(r[:, 1:], axis=1)

    r_min_overlap = np.max(r_min_k)
    r_max_overlap = np.min(r_max_k)

    valid = (r_totmid >= r_min_overlap) & (r_totmid <= r_max_overlap)
    eta[~valid] = np.nan

    # Build header
    header_cols = [
        f"{'i':>6}",
        f"{'log_r':>13}",
        f"{'log_rmid':>13}",
        f"{'m_tot':>13}",
        f"{'rho_tot':>13}",
        f"{'v2_tot':>13}",
        f"{'eta':>13}",
    ]
    # Per-species blocks
    for name in labels:
        header_cols.extend([
            f"{'lgr['+name+']':>13}",
            f"{'lgrm['+name+']':>13}",
            f"{'m['+name+']':>13}",
            f"{'rho['+name+']':>13}",
            f"{'v2['+name+']':>13}",
        ])
    header_line = "  ".join(header_cols) + "\n"

    with open(filename, "w") as f:
        f.write(header_line)

        # Row writer (edge quantities use i+1)
        for i in range(N):
            row = [
                f"{i:6d}",
                f"{np.log10(r_tot[i+1]): 13.6e}",
                f"{np.log10(r_totmid[i]): 13.6e}",
                f"{m_tot[i+1]: 13.6e}",
                f"{rho_tot[i]: 13.6e}",
                f"{v2_tot[i]: 13.6e}",
                f"{eta[i]: 13.6e}",
            ]
            # Per-species fields
            for k in range(s):
                row.extend([
                    f"{np.log10(state.r[k, i+1]): 13.6e}",
                    f"{np.log10(state.rmid[k, i]): 13.6e}",
                    f"{state.m[k, i+1]: 13.6e}",
                    f"{state.rho[k, i]: 13.6e}",
                    f"{state.v2[k, i]: 13.6e}",
                ])
            f.write("  ".join(row) + "\n")
    
    append_snapshot_conversion(state)

    if io.chatter and state.step_count == 0:
            print("Initial profiles written to disk.")

    if not initialize: # Do not increment if this is part of intializing the grid
        state.snapshot_index += 1

def append_snapshot_conversion(state):
    """
    Append conversion between snapshot_index and time

    Arguments
    ---------
    state : State
        The current simulation state.
    """
    filepath = os.path.join(
        state.config.io.base_dir, 
        state.config.io.model_dir, 
        f"snapshot_conversion.txt"
        )
    index = state.snapshot_index
    
    header = (f"{'index':>6}  {'time':>12}  {'time_Gyr':>12}  {'step':>10}\n")

    new_line = (
        f"{index:6d}  "
        f"{state.t:12.6e}  "
        f"{state.t * state.char.t0:12.6e}  "
        f"{state.step_count:10d}\n"
    )

    _update_file(filepath, header, new_line, index)

# def write_time_evolution(state):
#     """
#     Append time evolution data to time_evolution.txt

#         The output contains global time-evolution quantities followed by
#         per-species quantities for each label in ``state.labels``. The exact
#         columns and their order are defined by the file header.

#     Arguments
#     ---------
#     state : State
#         The current simulation state.
#     """
#     filepath = os.path.join(
#         state.config.io.base_dir,
#         state.config.io.model_dir,
#         "time_evolution.txt"
#     )

#     step = state.step_count
#     t = state.t

#     labels = list(state.labels)
#     s = len(labels)

#     # --- Compute core quantities
#     rho     = state.rho
#     r_c     = calc_r_c(state.rmid, rho, state.v2, CORE_HALF_RHO0,)
#     v2_c    = calc_v2_c(r_c, r, m, state.v2)
#     m_c     = sum_extensive_loglog_single(r_c, r, m)
#     rho_c   = 3.0 * m_c / r_c**3

#     # --- Compute eta_core on the fly
#     from pygtf2.util.calc import compute_eta_multi
    
#     m_part = state.m_part

#     # Mass-weighted velocity dispersion within r_core for each species
#     sigma_core = np.empty(s, dtype=np.float64)
#     M_core_species = np.empty(s, dtype=np.float64)
#     for k in range(s):
#         M_core = 0.0
#         Mv2_core = 0.0

#         for i in range(state.r.shape[1] - 1):
#             rin = state.r[k, i]
#             rout = state.r[k, i + 1]

#             if rin >= r_c:
#                 break

#             # Include only the portion of the shell lying inside r_core
#             rout_eff = min(rout, r_c)

#             dV = rout_eff**3 - rin**3
#             dM = state.rho[k, i] * dV

#             M_core += dM
#             Mv2_core += dM * state.v2[k, i]

#         M_core_species[k] = M_core
#         sigma_core[k] = np.sqrt(Mv2_core / M_core)

#     # Fit sigma_core \propto m^(-eta_core)
#     eta_core_arr = np.zeros(1, dtype=np.float64)
#     eta_core_err_arr = np.zeros(1, dtype=np.float64)

#     if s > 1 and np.max(m_part) > np.min(m_part):
#         compute_eta_multi(
#             m_part,
#             sigma_core[:, None],
#             eta_core_arr,
#             eta_core_err_arr,
#         )

#     eta_core = eta_core_arr[0]
#     eta_core_err = eta_core_err_arr[0]

#     # Optional useful diagnostic: species mass fractions within the core
#     f_core_species = M_core_species / np.sum(M_core_species)

#     v20_tot = sum_intensive_loglog_single(np.min(state.rmid[:,0]), state.rmid, rho*state.v2) / state.rho0

#     columns = [
#         ("step", step),
#         ("time", t),
#         ("rho0_tot", state.rho0),
#         ("v20_tot", v20_tot),
#         ("r_c", r_c),
#         ("v2_c", v2_c),
#         ("m_c", m_c),
#         ("rho_c_tot", rho_c),
#         ("eta_c", eta_core),
#     ]

#     percents = np.array([0.01, 0.05, 0.10, 0.20, 0.50, 0.90], dtype=np.float64)

#     r = state.r
#     m = state.m
#     r50evo = state.r50evo[:, 1]

#     for k, name in enumerate(labels):
#         rho0_k = float(rho[k, 0])
#         r50evok = r50evo[k]
#         m_ck = M_core_species[k]

#         radii = np.asarray(
#             mass_fraction_radii(r[k], m[k], percents),
#             dtype=np.float64
#         )

#         columns.extend([
#             (f"rho0[{name}]", rho0_k),
#             (f"r01[{name}]", radii[0]),
#             (f"r05[{name}]", radii[1]),
#             (f"r10[{name}]", radii[2]),
#             (f"r20[{name}]", radii[3]),
#             (f"r50[{name}]", radii[4]),
#             (f"r90[{name}]", radii[5]),
#             (f"r50evo[{name}]", r50evok),
#             (f"m_c[{name}]", m_ck),
#         ])

#     # Build header
#     header = "  ".join(f"{name:>13}" for name, _ in columns) + "\n"

#     # Build row
#     formatted_values = []
#     for name, value in columns:
#         if isinstance(value, int):
#             formatted_values.append(f"{value:13d}")
#         else:
#             formatted_values.append(f"{value:13.6e}")

#     new_line = "  ".join(formatted_values) + "\n"

#     _update_file(filepath, header, new_line, step)

#     if state.config.io.chatter and step == 0:
#         print("Time evolution file initialized.")

def write_time_evolution(state):
    """
    Append time-evolution data to time_evolution.txt.

    The output contains global time-evolution quantities followed by
    per-species quantities for each label in ``state.labels``. The exact
    columns and their order are defined by the file header.

    Arguments
    ---------
    state : State
        The current simulation state.
    """
    filepath = os.path.join(
        state.config.io.base_dir,
        state.config.io.model_dir,
        "time_evolution.txt",
    )

    step = state.step_count
    t = state.t

    labels = list(state.labels)
    s = len(labels)

    # --- State arrays
    r = state.r
    rmid = state.rmid
    m = state.m
    rho = state.rho
    v2 = state.v2
    m_part = state.m_part

    # --- Innermost system quantities
    r0 = np.min(rmid[:, 0])

    v20_tot = sum_intensive_loglog_single(r0, rmid, rho * v2,) / state.rho0

    # --- Core quantities
    r_c = calc_r_c(rmid, rho, v2, CORE_HALF_RHO0,)

    m_c_species, v2_c_species = calc_core_species_quantities(r_c, r, m, v2,)

    m_c = np.sum(m_c_species)

    # In code units, the 4*pi is absorbed into the density scale,
    # so <rho>_c = 3 M_c / r_c^3.
    rho_c = 3.0 * m_c / r_c**3
    rho_c_species = 3.0 * m_c_species / r_c**3

    # Total mass-weighted velocity dispersion squared within the core.
    v2_c = np.sum(m_c_species * v2_c_species) / m_c

    # --- Equipartition within the core
    from pygtf2.util.calc import compute_eta_multi

    sigma_c_species = np.sqrt(v2_c_species)

    eta_c_arr = np.zeros(1, dtype=np.float64)
    eta_c_err_arr = np.zeros(1, dtype=np.float64)

    if s > 1 and np.max(m_part) > np.min(m_part):
        compute_eta_multi( m_part, sigma_c_species[:, None], eta_c_arr, eta_c_err_arr,)

    eta_c = eta_c_arr[0]

    # --- Global columns
    columns = [
        ("step", step),
        ("time", t),
        ("rho0_tot", state.rho0),
        ("v20_tot", v20_tot),
        ("r_c", r_c),
        ("v2_c", v2_c),
        ("m_c", m_c),
        ("rho_c_tot", rho_c),
        ("eta_c", eta_c),
    ]

    # --- Per-species columns
    percents = np.array(
        [0.01, 0.05, 0.10, 0.20, 0.50, 0.90],
        dtype=np.float64,
    )

    r50evo = state.r50evo[:, 1]

    for k, name in enumerate(labels):
        rho0_k = float(rho[k, 0])
        rho_c_k = rho_c_species[k]
        m_ck = m_c_species[k]

        radii = np.asarray(
            mass_fraction_radii(
                r[k],
                m[k],
                percents,
            ),
            dtype=np.float64,
        )

        columns.extend([
            (f"rho0[{name}]", rho0_k),
            (f"rho_c[{name}]", rho_c_k),
            (f"r01[{name}]", radii[0]),
            (f"r05[{name}]", radii[1]),
            (f"r10[{name}]", radii[2]),
            (f"r20[{name}]", radii[3]),
            (f"r50[{name}]", radii[4]),
            (f"r90[{name}]", radii[5]),
            (f"r50evo[{name}]", r50evo[k]),
            (f"m_c[{name}]", m_ck),
        ])

    # --- Build header
    header = "  ".join(
        f"{name:>13}"
        for name, _ in columns
    ) + "\n"

    # --- Build row
    formatted_values = []

    for _, value in columns:
        if isinstance(value, (int, np.integer)):
            formatted_values.append(f"{value:13d}")
        else:
            formatted_values.append(f"{value:13.6e}")

    new_line = "  ".join(formatted_values) + "\n"

    _update_file(
        filepath,
        header,
        new_line,
        step,
    )

    if state.config.io.chatter and step == 0:
        print("Time evolution file initialized.")

def _update_file(filepath, header, new_line, index):
    """
    Helper function to update a file.
    If the file doesn't exist, it initializes it.
    If the file does exist, it appends the new_line, erasing all lines with
    a first column >= index.

    Arguments
    ---------
    filepath : str
        Path to the file.
    header : str
        Header row.
    new_line : str
        Row to be appended.
    index : int
        Index to compare to determine where to place new_line
    """

    lines = []

    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            lines = f.readlines()

        if lines and lines[0].strip() == header.strip():
            lines = [lines[0]] + [line for line in lines[1:] if int(line.split()[0]) < index]
        else:
            lines = [header]
    else:
        lines = [header]

    lines.append(new_line)

    with open(filepath, "w") as f:
        f.writelines(lines)
