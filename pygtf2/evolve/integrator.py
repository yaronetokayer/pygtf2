import numpy as np

def run_until_stop(state, start_step, **kwargs):
    """
    Repeatedly step forward until t >= t_halt or halting criterion met.
    """
    from pygtf2.io.write import write_profile_snapshot, write_log_entry, write_time_evolution

    # User halting criteria
    steps = kwargs.get('steps', None)
    time_limit = kwargs.get('stoptime', None)
    rho_c_limit = kwargs.get('rho_c', None)
    step_i = state.step_count if steps is not None else None
    time_i = state.t if time_limit is not None else None

    # Locals for speed + type hardening
    io = state.config.io
    sim = state.config.sim
    chatter = bool(io.chatter)
    t_halt = float(sim.t_halt)
    rho0_last_prof = float(state.rho_tot[0])
    rho0_last_tevol = float(state.rho_tot[0])
    rho_c_halt = float(sim.rho_c_halt)
    drho_prof = float(io.drho_prof)
    drho_tevol = float(io.drho_tevol)
    nlog = int(io.nlog)

    while state.t < t_halt:

        # Increment counter
        state.step_count += 1
        step_count = state.step_count

        # Compute proposed dt
        dt_prop = compute_time_step(state)

        # Integrate time step
        integrate_time_step(state, dt_prop, step_count)

        if step_count % 1000 == 0:
            print(f"Completed step {step_count}", end='\r', flush=True)

        rho0 = state.rho_tot[0]

        # Check halting criteria
        if rho0 > rho_c_halt:
            if step_count > 5e5:
                if chatter:
                    print("Simulation halted: central density exceeds halting value")
                break
        if np.isnan(rho0):
            if chatter:
                print("Simulation halted: central density is nan")
            break

        # User halting criteria
        if (
            (steps is not None and step_count - step_i >= steps)
            or (time_limit is not None and state.t - time_i >= time_limit)
            or (rho_c_limit is not None and rho0 >= rho_c_limit)
        ):
            if chatter:
                print("Simulation halted: user stopping condition reached")
            break

        # Check I/O criteria
        # Write profile to disk
        drho_for_prof = np.abs(rho0 - rho0_last_prof) / rho0_last_prof
        if drho_for_prof > drho_prof:
            rho0_last_prof = rho0
            write_profile_snapshot(state)

        # Track time evolution 
        drho_for_tevol = np.abs(rho0 - rho0_last_tevol) / rho0_last_tevol
        if drho_for_tevol > drho_tevol:
            rho0_last_tevol = rho0
            write_time_evolution(state)

        # Log
        if step_count % nlog == 0:
            write_log_entry(state, start_step)

    if state.t >= t_halt:
        if chatter:
            print("Simulation halted: max time exceeded")

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

def integrate_time_step(state, dt_prop, step_count):
    """
    Advance state by one time step.
    Applies conduction, revirialization, updates time, and checks stability diagnostics.

    Arguments
    ---------
    state : State
        The current simulation state.
    dt_prop : float
        Proposed dt value returned by compute_time_step
    step_count : int
        Step count
    """
    from pygtf2.evolve.transport import compute_luminosities, conduct_heat
    from pygtf2.evolve.hydrostatic import revirialize_drift_damp, compute_mass
    from pygtf2.evolve.realign import realign, realign_extensive

    # Store state attributes for fast access in loop and to pass into njit functions
    prec = state.config.prec
    char  = state.char

    c1 = float(char.c1); c2 = float(char.c2)
    mrat = state.mrat
    lnL = char.lnL

    r_orig  = np.asarray(state.r,   dtype=np.float64)
    m       = np.asarray(state.m,   dtype=np.float64)
    u_orig  = np.asarray(state.u,   dtype=np.float64)
    rho_orig= np.asarray(state.rho, dtype=np.float64)

    # Compute current luminosity array
    lum = compute_luminosities(c2, r_orig, u_orig, rho_orig, mrat, lnL)

    iter_cr = iter_dr = 0
    eps_du = float(prec.eps_du)
    eps_dr = float(prec.eps_dr)
    max_iter_cr = prec.max_iter_cr
    max_iter_dr = prec.max_iter_dr

    # Compute total enclosed mass including baryons, perturbers, etc.
    # May need to move into loop depending on how m is updated
    # Current version just returns m as is
    # m_tot = compute_mass(m)
    m_tot_orig = state.m_tot

    ### Step 1: Energy transport ###

    p_cond, du_max_new, dt_prop = conduct_heat(m, u_orig, rho_orig, lum, lnL, mrat, dt_prop, eps_du, c1)
    # p_cond, du_max_new, dt_prop = np.asarray(state.p, dtype=np.float64), 1e-5, dt_prop # FOR DEBUGGING!

    ### Step 2: Reestablish hydrostatic equilibrium ###

    status, r_new, rho_new, p_new, dr_max_new = revirialize_drift_damp(r_orig, rho_orig, p_cond, m_tot_orig)

    while True:
        if not np.all(r_new == r_new[0]):
            diff = r_new[1] - r_new[0]
            nonzero_mask = diff != 0
            n_nonzero = int(np.count_nonzero(nonzero_mask))
            if n_nonzero:
                max_abs = float(np.max(np.abs(diff[nonzero_mask])))
            else:
                max_abs = 0.0
            print(f"{step_count}: Nonzero positions: {n_nonzero}, max abs among them: {max_abs:.6g}")

        # Shell crossing
        if status == 'shell_crossing':
            print("crossed:")
            print(r_new[:,:10])
            raise RuntimeError("stopping")
            if iter_cr >= max_iter_cr:
                raise RuntimeError("Max iterations exceeded for shell crossing in conduction/revirialization step")
            dt_prop *= 0.5
            iter_cr += 1
            repeat_revir = False
            print(f"{step_count}: updated dt_prop to {dt_prop}")
            print(f"dr_max_new: {dr_max_new}")
            break # Exit inner loop, redo conduct_heat with original values and smaller dt

        # if step_count > 28:
        #     print(step_count, repeat_revir, ":")
        #     plot_r_markers(r_new[:,1:20])

        v2_new = p_new / rho_new
        r_real, rho_real, u_real, v2_real, p_real, m_real, m_tot_real = realign_extensive(r_new, rho_new, v2_new)

        # Check dr criterion
        """
        With new step to ensure equilibrium in initialization, no longer a need to accept larger dr in first time step.
        If needed, can reintroduce with 'and (step_count != 1):' in the if statement below.
        """
        if dr_max_new > eps_dr:
            # print(step_count, dr_max_new)
            if iter_dr >= max_iter_dr:
                print(step_count, dr_max_new)
                raise RuntimeWarning("Max iterations exceeded for dr in revirialization step")
            iter_dr += 1
            status, r_new, rho_new, p_new, dr_max_new = revirialize_drift_damp(r_real, rho_real, p_real, m_tot_real)
            # status, r_new, rho_new, p_new, dr_max_new = revirialize(r_new, rho_new, p_new, m_tot_orig)
            continue # Go to top of loop

        # Both criteria are met, break out of loop
        break

    # if step_count > -1:
    #     print(step_count, iter_dr, dr_max_new)
        # plot_r_markers(r_new[:,1:10])
    # v2_new = p_new / rho_new
    # r_real, rho_real, u_real, v2_real, p_real, m_real, m_tot_real = realign_extensive(r_new, rho_new, v2_new)

    ### Step 3: Update state variables ###

    state.r = r_real
    state.rho = rho_real
    state.p = p_real
    state.v2 = v2_real
    state.m = m_real
    # state.r = r_new
    # state.rho = rho_new
    # state.p = p_new
    # state.v2 = v2_new
    # state.m = m_new
    state.dr_max = dr_max_new
    state.du_max = du_max_new

    state.rmid = 0.5 * (r_real[:, 1:] + r_real[:, :-1])
    state.u = u_real
    sqrt_v2_real = np.sqrt(v2_real)
    state.trelax = 1.0 / (sqrt_v2_real * rho_real)
    # state.rmid = 0.5 * (r_new[:, 1:] + r_new[:, :-1])
    # state.u = 1.5 * v2_new
    # sqrt_v2_new = np.sqrt(v2_new)
    # state.trelax = 1.0 / (sqrt_v2_new * rho_new)

    state.maxvel    = float(np.max(sqrt_v2_real))
    # state.maxvel    = float(np.max(sqrt_v2_new))
    state.mintrelax = float(np.min(state.trelax))

    state.m_tot     = m_tot_real
    # state.m_tot     = m_tot_new
    state.rho_tot   = rho_new.sum(axis=0)
    state.p_tot     = p_new.sum(axis=0)
    state.v2_tot    = state.p_tot / state.rho_tot
    state.u_tot     = 1.5 * state.v2_tot

    # Diagnostics
    state.n_iter_cr += iter_cr
    state.n_iter_dr += iter_dr
    state.dt_cum += float(dt_prop)
    if step_count != 1:
        state.dr_max_cum += float(dr_max_new)
    state.du_max_cum += float(du_max_new)
    state.dt_over_trelax_cum += float(dt_prop / state.mintrelax)

    state.dt = float(dt_prop)
    state.t += float(dt_prop)

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