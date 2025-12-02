import matplotlib.pyplot as plt

def plot_r_markers(r_slice):
    """
    r_slice : array of shape (s, m)
        Radii for each species (s species, m points each).
    """
    s, m = r_slice.shape
    fig, axes = plt.subplots(s, 1, figsize=(10, 0.75*s), sharex=True)

    if s == 1:
        axes = [axes]

    for k, ax in enumerate(axes):
        ax.set_xscale("log")
        for j, rj in enumerate(r_slice[k]):
            ax.axvline(rj, color="k", lw=1)
            ax.text(rj, -0.05, str(j), ha="center", va="top",
                    transform=ax.get_xaxis_transform(), fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_xlim(3e-3, 1e-1)
        ax.set_yticks([])
        ax.set_ylabel(f"species {k+1}", rotation=0, labelpad=25, va="center")

    axes[-1].set_xlabel("r (log scale)")
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

def plot_r_markers(r_slice):
    """
    r_slice : array of shape (s, m)
        Radii for each species (s species, m points each).
    """
    s, m = r_slice.shape
    fig, axes = plt.subplots(s, 1, figsize=(10, 0.75*s), sharex=True)

    if s == 1:
        axes = [axes]

    for k, ax in enumerate(axes):
        ax.set_xscale("log")
        for j, rj in enumerate(r_slice[k]):
            ax.axvline(rj, color="k", lw=1)
            ax.text(rj, -0.05, str(j), ha="center", va="top",
                    transform=ax.get_xaxis_transform(), fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_xlim(3e-3, 1e-1)
        ax.set_yticks([])
        ax.set_ylabel(f"species {k+1}", rotation=0, labelpad=25, va="center")

    axes[-1].set_xlabel("r (log scale)")
    plt.tight_layout()
    plt.show()

    ### OLD UNUSED METHODS HERE

# now I just have a single time stepping criterion based on trelax.
# the eps_du criterion is checked within the conduction step.
def compute_time_step(state) -> float:
    """
    Compute time step to be used for integration step.

    Arguments
    ---------
    state : State
        The current simulation state.

    Returns
    -------
    float
        The recommended time step.
    """
    if state.step_count == 1:
        return 1.0e-9
    
    prec = state.config.prec
    # Relaxation-limited time step
    dt1 = prec.eps_dt * state.mintrelax
    # Energy stability-limited time step
    tiny = np.finfo(np.float64).tiny
    du_max_safe = max(state.du_max, tiny)
    dt2 = state.dt * 0.95 * (prec.eps_du / du_max_safe)

    return float(min(dt1, dt2))

def write_log_entry_old(state, start_step):
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

    header = f"{'step':>10}  {'time':>12}  {'<dt>':>12}  {'rho_c':>12}  {'v_max':>12}  {'<dt lim>':>8}  {'<dr lim>':>8}  {'<du lim>':>8}  {'<n_iter_cr>':>11}  {'<n_iter_dr>':>11}\n"
    new_line = f"{step:10d}  {state.t:12.6e}  {state.dt_cum / nlog:12.6e}  {state.rho_c:12.6e}  {state.maxvel:12.6e}  {state.dt_over_trelax_cum / prec.eps_dt / nlog:8.2e}  {state.dr_max_cum / prec.eps_dr / nlog:8.2e}  {state.du_max_cum / prec.eps_du / nlog:8.2e}  {state.n_iter_cr / nlog:11.5e}  {state.n_iter_dr / nlog:11.5e}\n"

    if step == start_step:
        new_line = new_line[:26] + f"         N/A" +  new_line[38:66] + f"       N/A       N/A       N/A          N/A          N/A\n"

    _update_file(filepath, header, new_line, step)

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