import numpy as np

def run_until_stop(state, start_step, **kwargs):
    """
    Repeatedly step forward until t >= t_halt or halting criterion met.
    """
    from pygtfcode.io.write import write_profile_snapshot, write_log_entry, write_time_evolution

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
    rho0_last_prof = float(state.rho[0])
    rho0_last_tevol = float(state.rho[0])
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

        rho0 = state.rho[0]

        # Check halting criteria
        if rho0 > rho_c_halt:
            if state.t > 50:
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
        return 1.0e-7
    
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
    from pygtfcode.evolve.transport import compute_luminosities, conduct_heat
    from pygtfcode.evolve.hydrostatic import revirialize, compute_mass

    # Store state attributes for fast access in loop and to pass into njit functions
    prec = state.config.prec
    sim  = state.config.sim
    init = state.config.init

    a = float(sim.a); b = float(sim.b); c = float(sim.c)
    sigma_m = float(state.char.sigma_m_char)
    cored = (init.profile == 'abg') and (float(init.gamma) < 1.0)

    r_orig  = np.asarray(state.r,   dtype=np.float64)
    m       = np.asarray(state.m,   dtype=np.float64)
    v2_orig = np.asarray(state.v2,  dtype=np.float64)
    p_orig  = np.asarray(state.p,   dtype=np.float64)
    u_orig  = np.asarray(state.u,   dtype=np.float64)
    rho_orig= np.asarray(state.rho, dtype=np.float64)

    # Compute current luminosity array
    lum = compute_luminosities(a, b, c, sigma_m, r_orig, v2_orig, p_orig, cored)

    iter_cr = iter_dr = 0
    eps_du = float(prec.eps_du)
    eps_dr = float(prec.eps_dr)
    max_iter_cr = prec.max_iter_cr
    max_iter_dr = prec.max_iter_dr
    converged = False
    repeat_revir = False

    # Compute total enclosed mass including baryons, perturbers, etc.
    # May need to move into loop depending on how m is updated
    # Current version just returns m as is
    m_tot = compute_mass(m)

    while not converged:
        ### Step 1: Energy transport ###
        p_cond, du_max_new, dt_prop = conduct_heat(m, u_orig, rho_orig, lum, dt_prop, eps_du)

        ### Step 2: Reestablish hydrostatic equilibrium ###
        while True:
            if repeat_revir:
                result = revirialize(r_new, rho_new, p_new, m_tot)
            else:
                result = revirialize(r_orig, rho_orig, p_cond, m_tot)

            # Shell crossing signaled by None
            if result is None:
                if iter_cr >= max_iter_cr:
                    raise RuntimeError("Max iterations exceeded for shell crossing in conduction/revirialization step")
                dt_prop *= 0.5
                iter_cr += 1
                repeat_revir = False
                break # Exit inner loop, redo conduct_heat with original values and smaller dt
            
            # Check dr criterion
            """
            With new step to ensure equilibrium in initialization, no longer a need to accept larger dr in first time step.
            If needed, can reintroduce with 'and (step_count != 1):' in the if statement below.
            """
            if result[3] > eps_dr:
                if iter_dr >= max_iter_dr:
                    raise RuntimeWarning("Max iterations exceeded for dr in revirialization step")
                iter_dr += 1
                r_new, rho_new, p_new, _ = result
                repeat_revir = True
                continue # Go to top of inner loop, repeat revirialize with new values

            # Both criteria are met, break out of inner and outer loop
            r_new, rho_new, p_new, dr_max_new = result
            converged = True
            break

    ### Step 3: Update state variables ###
    # m not updated in Lagrangian code

    v2_new = p_new / rho_new
    state.r = r_new
    state.rho = rho_new
    state.p = p_new
    state.v2 = v2_new
    state.dr_max = dr_max_new
    state.du_max = du_max_new

    state.rmid = 0.5 * (r_new[1:] + r_new[:-1])
    state.u = 1.5 * v2_new
    state.kn = 1.0 / (state.char.sigma_m_char * np.sqrt(p_new))
    sqrt_v2_new = np.sqrt(v2_new)
    state.trelax = 1.0 / (sqrt_v2_new * rho_new)

    state.maxvel    = float(np.max(sqrt_v2_new))
    state.minkn     = float(np.min(state.kn))
    state.mintrelax = float(np.min(state.trelax))

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