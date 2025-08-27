import numpy as np
from pygtfcode.parameters.constants import Constants as const
import os

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

    def dump_attrs(obj, indent=0):
        lines = []
        for key in sorted(vars(obj)):
            val = getattr(obj, key)
            if hasattr(val, '__dict__'):  # it's a nested config object
                lines.append(" " * indent + f"{key}:")
                lines.extend(dump_attrs(val, indent + 4))
            else:
                lines.append(" " * indent + f"{key}: {val}")
        return lines

    with open(filename, "w") as f:
        f.write(f"Model {io.model_no:03d} Metadata\n")
        f.write("=" * 40 + "\n\n")
        lines = dump_attrs(state.config)
        f.write("\n".join(lines) + "\n")

    if state.config.io.chatter:
        print(f"Model information written to model_metadata.txt")

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
    io = state.config.io
    prec = state.config.prec
    filepath = os.path.join(io.base_dir, io.model_dir, f"logfile.txt")
    chatter = io.chatter
    step = state.step_count
    nlog = io.nlog
    if ( step - start_step ) % nlog != 0:
        nlog = ( step - start_step ) % nlog

    header = f"{'step':>10}  {'time':>12}  {'<dt>':>12}  {'rho_c':>12}  {'v_max':>12}  {'Kn_min':>12}  {'<dt lim>':>8}  {'<dr lim>':>8}  {'<du lim>':>8}  {'<n_iter_cr>':>11}  {'<n_iter_dr>':>11}\n"
    new_line = f"{step:10d}  {state.t:12.6e}  {state.dt_cum / nlog:12.6e}  {state.rho[0]:12.6e}  {state.maxvel:12.6e}  {state.minkn:12.6e}  {state.dt_over_trelax_cum / prec.eps_dt / nlog:8.2e}  {state.dr_max_cum / prec.eps_dr / nlog:8.2e}  {state.du_max_cum / prec.eps_du / nlog:8.2e}  {state.n_iter_cr / nlog:11.5e}  {state.n_iter_dr / nlog:11.5e}\n"

    if step == start_step:
        new_line = new_line[:26] + f"         N/A" +  new_line[38:80] + f"       N/A       N/A       N/A          N/A          N/A\n"

    _update_file(filepath, header, new_line, step)

    state.n_iter_du = 0
    state.n_iter_cr = 0
    state.n_iter_dr = 0
    state.dt_cum = 0.0
    state.du_max_cum = 0.0
    state.dr_max_cum = 0.0
    state.dt_over_trelax_cum = 0.0

    if chatter:
        if step == 0:
            print("Log file initialized:")
        if step == start_step:
            print(header[:-1])
        print(new_line[:-1])

def write_profile_snapshot(state, initialize=False):
    """ 
    Write full radial profiles to disk.

    Arguments
    ---------
    state : State
        The current simulation state.
    initialize : bool
        If True, this is part of initializing the grid and should not increment the snapshot index.
    """
    filename = os.path.join(state.config.io.base_dir, state.config.io.model_dir, f"profile_{state.snapshot_index}.dat")

    with open(filename, "w") as f:
        header = (
            f"{'i':>6}  {'log_r':>12}  {'log_rmid':>12}  {'m':>12}  "
            f"{'rho':>12}  {'v2':>12}  {'p':>12}  {'trelax':>12}  {'kn':>12}\n"
        )
        f.write(header)
        for i in range(len(state.r) - 1):
            f.write(
                f"{i:6d}  "
                f"{np.log10(state.r[i+1]):12.6e}  "
                f"{np.log10(state.rmid[i]):12.6e}  "
                f"{state.m[i+1]:12.6e}  "
                f"{state.rho[i]:12.6e}  "
                f"{state.v2[i]:12.6e}  "
                f"{state.p[i]:12.6e}  "
                f"{state.trelax[i]:12.6e}  "
                f"{state.kn[i]:12.6e}\n"
            )
    
    append_snapshot_conversion(state)

    if state.config.io.chatter:
        if state.step_count == 0:
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
        f"{state.t * state.char.t0 * const.sec_to_Gyr:12.6e}  "
        f"{state.step_count:10d}\n"
    )

    _update_file(filepath, header, new_line, index)

def write_time_evolution(state):
    """
    Append time evolution data to time_evolution.dat

    Arguments
    ---------
    state : State
        The current simulation state.
    """
    filepath = os.path.join(state.config.io.base_dir, state.config.io.model_dir, f"time_evolution.txt")
    step = state.step_count

    header = (
        f"{'step':>10}  "
        f"{'time':>12}  "
        f"{'t_Gyr':>12}  "
        f"{'rho_c':>12}  "
        f"{'rhoc_Msunpc3':>12}  "
        f"{'v_max':>12}  "
        f"{'v_max_kms':>12}  "
        f"{'Kn_min':>12}  "
        f"{'mintrel':>12}  "
        f"{'mintrel_Gyr':>12}\n"
    )

    char = state.char
    t = state.t
    t_conv = char.t0 * const.sec_to_Gyr
    rho_c = state.rho[0]
    maxvel = state.maxvel
    mintrelax = state.mintrelax

    new_line = ( 
        f"{step:10d}  "
        f"{t:12.6e}  "
        f"{t * t_conv:12.6e}  "
        f"{rho_c:12.6e}  "
        f"{rho_c * char.rho_s *  1.0E-18:12.6e}  "
        f"{maxvel:12.6e}  "
        f"{maxvel * char.v0:12.6e}  "
        f"{state.minkn:12.6e}  "
        f"{mintrelax:12.6e}  "
        f"{mintrelax * t_conv:12.6e}\n" 
        )
    
    _update_file(filepath, header, new_line, step)

    if state.config.io.chatter:
        if step == 0:
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