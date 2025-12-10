import numpy as np
from collections import deque
from pygtf2.io.write import write_profile_snapshot, write_log_entry, write_time_evolution
from pygtf2.evolve.transport import compute_luminosities, conduct_heat
from pygtf2.evolve.hydrostatic import revirialize_interp
from pygtf2.evolve.evaporate import evaporate
from pygtf2.util.calc import calc_rho_v2_r_c, calc_r50_spread

def run_until_stop(state, start_step, **kwargs):
    """
    Repeatedly step forward until t >= t_halt or halting criterion met.
    """
    ##################
    ### Set locals ###
    ##################

    # --- User halting criteria ---
    steps = kwargs.get('steps', None)
    time_limit = kwargs.get('stoptime', None)
    rho_c_limit = kwargs.get('rho_c', None)
    step_i = state.step_count if steps is not None else None
    time_i = state.t if time_limit is not None else None

    # --- Locals for speed + type hardening ---
    io = state.config.io
    sim = state.config.sim
    chatter = bool(io.chatter)
    t_halt = float(sim.t_halt)
    eps_dt = state.config.prec.eps_dt
    rho0_last_prof = float(state.rho_c)
    rho0_last_tevol = float(state.rho_c)
    r50_spread_last_tevol = float(state.r50_spread)
    rho_c_halt = float(sim.rho_c_halt)
    drho_prof = float(io.drho_prof)
    drho_tevol = float(io.drho_tevol)
    mrat = state.mrat
    if np.allclose(mrat, mrat[0]):
        use_r50 = False
    else:
        dr50_tevol = float(io.dr50_tevol)
        use_r50 = True
    nlog = int(io.nlog)
    nupdate = int(io.nupdate)

    # --- oscillation detection and throttling ---
    osc_window = 5
    osc_threshold_on  = 100     # avg spacing < 100 steps → oscillation detected
    osc_threshold_off = 10_000     # avg spacing > 10k steps → stable again

    min_prof_spacing = 500_000
    min_tevol_spacing = 100_000

    prof_desired_steps = deque(maxlen=osc_window)       # double-ended queue
    tevol_desired_steps = deque(maxlen=osc_window)

    prof_last_write = None
    tevol_last_write = None

    avg_spacing_prof = float('inf')
    avg_spacing_tevol = float('inf')

    prof_force_min = False
    tevol_force_min = False

    def detect_avg_spacing(dq):
        if len(dq) < osc_window:
            return float('inf')
        spacings = [dq[i+1] - dq[i] for i in range(len(dq)-1)]
        return sum(spacings) / len(spacings)

    #################
    ### Main loop ###
    #################

    while state.t < t_halt:

        ###########################
        ### 1. Integrate system ###
        ###########################

        # Increment counter
        state.step_count += 1
        step_count = state.step_count

        # Compute proposed dt
        dt_prop = eps_dt * state.mintrelax

        # Integrate time step
        integrate_time_step(state, dt_prop, step_count)

        if step_count % nupdate == 0:
            print(f"Completed step {step_count}", end='\r', flush=True)

        rho0 = state.rho_c
        r50_spread = state.r50_spread

        ###########################
        ### 2. Halting criteria ###
        ###########################

        # Hardcoded criteria
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

        #########################
        ### 3. Output to disk ###
        #########################

        # --- PROFILE OUTPUT ---
        drho_for_prof = abs((rho0 / rho0_last_prof) - 1.0)

        # Update desired spacing if criteria satisfied
        want_prof = (drho_for_prof > drho_prof)
        if want_prof:
            prof_desired_steps.append(step_count)
            avg_spacing_prof = detect_avg_spacing(prof_desired_steps)

        # Auto de-throttle (re-enable normal criteria)
        if prof_force_min and avg_spacing_prof > osc_threshold_off:
            if chatter:
                print(f"{step_count}: Profile output throttling disabled (system stabilized)")
            prof_force_min = False

        if prof_force_min:
            # Throttled mode
            if prof_last_write is None or (step_count - prof_last_write >= min_prof_spacing):
                rho0_last_prof = rho0
                prof_last_write = step_count
                write_profile_snapshot(state)

        else:
            # Normal mode
            if want_prof:
                rho0_last_prof = rho0
                prof_last_write = step_count
                write_profile_snapshot(state)

                # Detect oscillation
                if avg_spacing_prof < osc_threshold_on:
                    if step_count > osc_threshold_off:
                        if chatter:
                            print(f"{step_count}: Profile outputs too rapid — enabling throttling")
                        prof_force_min = True

        # --- TIME EVOLUTION OUTPUT ---
        drho_for_tevol = abs((rho0 / rho0_last_tevol) - 1.0)
        want_tevol = drho_for_tevol > drho_tevol

        if use_r50:
            denom = r50_spread_last_tevol if abs(r50_spread_last_tevol) > 1e-100 else 1e-100
            r50_spread_for_tevol = abs((r50_spread - r50_spread_last_tevol) / denom)
            if r50_spread_for_tevol > dr50_tevol:
                want_tevol = True

        # Update desired spacing if criteria satisfied
        if want_tevol:
            tevol_desired_steps.append(step_count)
            avg_spacing_tevol = detect_avg_spacing(tevol_desired_steps)
        
        # Auto de-throttle
        if tevol_force_min and avg_spacing_tevol > osc_threshold_off:
            if chatter:
                print(f"{step_count}: Time evolution output throttling disabled (system stabilized)")
            tevol_force_min = False

        if tevol_force_min:     # Throttled mode
            if tevol_last_write is None or (step_count - tevol_last_write >= min_tevol_spacing):
                rho0_last_tevol = rho0
                r50_spread_last_tevol = r50_spread
                tevol_last_write = step_count
                write_time_evolution(state)

        else:
            # Normal mode
            if want_tevol:
                rho0_last_tevol = rho0
                r50_spread_last_tevol = r50_spread
                tevol_last_write = step_count
                write_time_evolution(state)

                # Detect oscillation
                if avg_spacing_tevol < osc_threshold_on:
                    if step_count > osc_threshold_off:
                        if chatter:
                            print(f"{step_count}: Time evolution outputs too rapid — enabling throttling")
                        tevol_force_min = True

        ##############
        ### 4. Log ###
        ##############

        if step_count % nlog == 0:
            write_log_entry(state, start_step)

    if state.t >= t_halt:
        if chatter:
            print("Simulation halted: max time exceeded")

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
    # Store state attributes for fast access in loop and to pass into njit functions
    config = state.config
    char  = state.char
    bkg_param = state.bkg_param
    evap = config.sim.evap

    c1 = float(char.c1); c2 = float(char.c2)
    eps_du = float(config.prec.eps_du)
    mrat = state.mrat
    lnL = char.lnL

    r_orig      = np.asarray(state.r,   dtype=np.float64)
    rmid_orig   = np.asarray(state.rmid,   dtype=np.float64)
    m           = np.asarray(state.m,   dtype=np.float64)
    u_orig      = 1.5 * np.asarray(state.v2,  dtype=np.float64)
    rho_orig    = np.asarray(state.rho, dtype=np.float64)

    ### Step 1: Energy transport ###
    lum = compute_luminosities(c2, r_orig, u_orig, rho_orig, mrat, lnL)
    p_cond, v2_cond, du_max, dt_prop = conduct_heat(m, u_orig, rho_orig, lum, lnL, mrat, r_orig, dt_prop, eps_du, c1)

    # Apply evaporation
    if evap:
        evaporate(r_orig, rmid_orig, m, v2_cond, rho_orig, dt_prop) # Modifies rho and m in place
        p_cond = v2_cond * rho_orig

    ### Step 2: Reestablish hydrostatic equilibrium ###
    status, r_new, rho_new, p_new, dr_max, he_res = revirialize_interp(r_orig, rho_orig, p_cond, m, bkg_param)
        
    # Shell crossing
    if status == 'shell_crossing':
        raise RuntimeError(f"Step {step_count}: Shell crossing in conduction/revirialization step")

    ### Step 3: Update state variables ###

    state.r = r_new
    state.rho = rho_new
    state.p = p_new
    v2_new = p_new / rho_new
    state.v2 = v2_new
    state.du_max = du_max
    if evap:
        state.m = m

    rmid = 0.5 * (r_new[:, 1:] + r_new[:, :-1])
    state.rmid = rmid
    state.trelax = v2_new**(3.0/2.0) / rho_new

    state.mintrelax                     = float(np.min(state.trelax))
    state.rho_c, state.v2_c, state.r_c  = calc_rho_v2_r_c(rmid, rho_new, v2_new)
    state.r50_spread                    = calc_r50_spread(r_new, m)

    # Diagnostics
    state.dt_cum += float(dt_prop)
    state.du_max_cum += float(du_max)
    state.dt_over_trelax_cum += float(dt_prop / state.mintrelax)

    state.dt = float(dt_prop)
    state.t += float(dt_prop)