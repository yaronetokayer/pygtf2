import numpy as np 
from pygtf2.io.write import write_profile_snapshot, write_log_entry, write_time_evolution
from pygtf2.evolve.transport import conduct_imex_once, HEX_FIRST, COND_FIRST, STRANG_SPLIT
from pygtf2.evolve.hydrostatic import revirialize_interp_jacobi, STATUS_SHELL_CROSSING, STATUS_CAVITY_REACHED, STATUS_OK
from pygtf2.evolve.evaporate import evaporate
from pygtf2.evolve.binaries import binary_heating
from pygtf2.util.interpolate import sum_intensive_loglog_single
from pygtf2.util.calc import calc_r50_spread
from pygtf2.util.little_helpers import max_frac_change
from pygtf2.dev.debug import plot_r_markers

def run_until_stop(state, start_step, **kwargs):
    """
    Advance state in place with adaptive timesteps and periodic output.

    start_step is the step at entry to State.run, used for log averaging.
    Optional keywords steps and stoptime limit additional steps and elapsed
    simulation time; rho0 limits the total innermost density. Configured
    sim.t_halt remains an absolute time limit, and sim.rho0_halt is checked
    after 1000 total steps. Returns None.
    """
    ##################
    ### Set locals ###
    ##################

    # --- User halting criteria ---
    steps = kwargs.get('steps', None)
    time_limit = kwargs.get('stoptime', None)
    rho0_limit = kwargs.get('rho0', None)
    step_i = state.step_count if steps is not None else None
    time_i = state.t if time_limit is not None else None

    # --- Locals ---
    config  = state.config
    io      = config.io; sim = config.sim; prec = config.prec
    char    = state.char

    # Switches
    # evap = sim.evap
    binaries = sim.binaries; n_particle = sim.n_particle

    # Parameters
    c1 = float(char.c1); c2 = float(char.c2)
    eps_du = float(prec.eps_du); max_iter_du = prec.max_iter_du
    mrat = state.mrat; lnL = char.lnL
    bkg_param = state.bkg_param

    # Preallocations
    N = config.grid.ngrid
    Nm1 = N - 1
    work_sn1    = np.empty_like(state.rho, dtype=np.float64)
    work_sn2    = np.empty_like(state.rho, dtype=np.float64)
    work_snout1 = np.empty_like(state.m, dtype=np.float64)
    work_snout2 = np.empty_like(state.m, dtype=np.float64)
    work_n1     = np.empty(N,      dtype=np.float64)
    work_n2     = np.empty(N,      dtype=np.float64)
    work_n3     = np.empty(N,      dtype=np.float64)
    work_n4     = np.empty(N,      dtype=np.float64)
    work_nin1   = np.empty(Nm1,    dtype=np.float64)
    work_nin2   = np.empty(Nm1,    dtype=np.float64)
    work_nin3   = np.empty(Nm1,    dtype=np.float64)
    work_nin4   = np.empty(Nm1,    dtype=np.float64)
    work_nin5   = np.empty(Nm1,    dtype=np.float64)

    # Output options
    t_evol = bool(io.t_evol); profiles = bool(io.profiles)
    chatter = bool(io.chatter); t_halt = float(sim.t_halt); nlog = int(io.nlog); nupdate = int(io.nupdate)
    rho0_halt = float(sim.rho0_halt)
    if t_evol:
        rho0_last_tevol = float(state.rho0); drho_tevol = float(io.drho_tevol)
        r50_spread_last_tevol = float(state.r50_spread)
        if np.allclose(mrat, mrat[0]):
            use_r50 = False
        else:
            dr50_tevol = float(io.dr50_tevol)
            use_r50 = True
    if profiles:
        rho0_last_prof = float(state.rho0); drho_prof = float(io.drho_prof)

    safety = 0.99 # For timestepping

    #################
    ### Main loop ###
    #################

    stop_reason = None

    while state.t < t_halt:

        ###########################
        ### 1. Integrate system ###
        ###########################

        # Proposed next step; only commit after successful integration
        step_count = state.step_count + 1

        # Compute proposed dt
        if step_count == 1:
            dt_prop = 1.0 # We have no maxdu yet

        else:         
            #--- du-limited dt using proportional control
            err = eps_du / state.du_max
            fac = safety * err
            dt_prop = fac * state.dt

        # Integrate time step
        status = integrate_time_step(
            state, dt_prop, step_count,
            binaries, # evap,                                               # Switches
            eps_du, max_iter_du, c1, c2, mrat, lnL, bkg_param, n_particle,  # Parameters
            work_sn1, work_sn2,                                             # Preallocated arrays
            work_snout1, work_snout2,
            work_n1, work_n2, work_n3, work_n4,
            work_nin1, work_nin2, work_nin3, work_nin4, work_nin5,
        )

        if status == STATUS_CAVITY_REACHED:
            stop_reason = 'inner cavity reached'
            break

        # Accept the completed timestep
        state.step_count = step_count

        if step_count % nupdate == 0:
            print(f"Completed step {step_count}", end='\r', flush=True)

        rho0 = state.rho0
        r50_spread = state.r50_spread

        ###########################
        ### 2. Halting criteria ###
        ###########################

        # Hardcoded criteria
        if rho0 > rho0_halt and step_count > 1e3:
                stop_reason = 'central density exceeds halting value'
                break
        
        if np.isnan(rho0):
            stop_reason = 'central density is nan'
            break

        # User halting criteria
        if (
            (steps is not None and step_count - step_i >= steps)
            or (time_limit is not None and state.t - time_i >= time_limit)
            or (rho0_limit is not None and rho0 >= rho0_limit)
        ):
            stop_reason = 'user stopping condition reached'
            break

        #########################
        ### 3. Output to disk ###
        #########################

        # --- PROFILE OUTPUT ---
        if profiles:
            drho_for_prof = abs((rho0 / rho0_last_prof) - 1.0)
            if drho_for_prof > drho_prof:
                rho0_last_prof = rho0
                write_profile_snapshot(state)
                write_time_evolution(state) # Always have a time evolution column concurrent with a profile

        # --- TIME EVOLUTION OUTPUT ---
        if t_evol:
            drho_for_tevol = abs((rho0 / rho0_last_tevol) - 1.0)
            want_tevol = drho_for_tevol > drho_tevol

            if use_r50:
                denom = max(abs(r50_spread_last_tevol), 1.0e-6)
                r50_spread_for_tevol = abs((r50_spread - r50_spread_last_tevol) / denom)
                if r50_spread_for_tevol > dr50_tevol:
                    want_tevol = True

            if want_tevol:
                rho0_last_tevol = rho0
                r50_spread_last_tevol = r50_spread
                write_time_evolution(state)

        ##############
        ### 4. Log ###
        ##############

        if step_count % nlog == 0:
            write_log_entry(state, start_step)

    if stop_reason is None:
        stop_reason = 'max time exceeded'

    if chatter:
        print(f"Simulation halted: {stop_reason}")

def integrate_time_step(state, dt_prop, step_count,                 # State
                        binaries, # evap                            # Switches
                        eps_du, max_iter_du,
                        c1, c2, mrat, lnL, bkg_param, n_particle,   # Parameters
                        work_sn1, work_sn2,                         # Preallocated arrays
                        work_snout1, work_snout2,
                        work_n1, work_n2, work_n3, work_n4,
                        work_nin1, work_nin2, work_nin3, work_nin4, work_nin5,  
                        ):
    """
    Advance state by one time step.
    Applies conduction, revirialization, updates time, and checks stability diagnostics.

    Arguments
    ---------
    state : State
        The current simulation state.
    dt_prop : float
        Proposed timestep from the caller's adaptive controller.
    step_count : int
        Step count
    binaries : bool
        Activate heat generation due to binaries.
    eps_du : float
        Maximum allowed fractional change in v2 per time step.
    max_iter_du : int
        Maximum number of trial energy updates, including binary heating
        when enabled. Failure to accept a trial raises RuntimeError.
    c1, c2, mrat, lnL : float or array-like
        Model parameters.
    bkg_param : ndarray, shape (4,)
        Background potential parameters.
    n_particle : int
        Absolute particle number for heat generation due to binaries.
    work_sn1, work_sn2 : ndarray (s,N)
        Memory allocation for working arrays
    work_snout1, work_snout2 : ndarray (s,N+1)
        Total enclosed mass and other-species mass-derivative workspaces,
        respectively, for revirialization.
    work_n1, work_n2, work_n3, work_n4 : ndarray (N,)
        Memory allocation for working arrays
    work_nin1, work_nin2, work_nin3, work_nin4, work_nin5 : ndarray (N-1,)
        Memory allocation for working arrays
    """
    # Pointers for easy access
    m_bh        = float(state.m_bh)
    r           = np.asarray(state.r,       dtype=np.float64)
    m           = np.asarray(state.m,       dtype=np.float64)
    v2          = np.asarray(state.v2,      dtype=np.float64)
    rho         = np.asarray(state.rho,     dtype=np.float64)

    ### Step 1: Energy transport ###

    # Save initial state
    work_sn2[:,:] = v2

    dt_trial = dt_prop
    iter_du = -1

    for j in range(max_iter_du):
        if j > 0:  # start from initial state
            v2[:,:] = work_sn2

        # IMEX METHOD
        order = STRANG_SPLIT
        conduct_imex_once(
            v2, rho, r, m, m_bh,
            c1, c2, mrat, lnL,
            work_sn1, work_n1, work_n2, work_n3, work_n4,
            dt_trial, order,
        )

        # Other energy modules can be added here
        if binaries:
            binary_heating(rho, v2, r, dt_trial, mrat, n_particle)

        du_max = max_frac_change(v2, work_sn2)

        if du_max <= eps_du:
            iter_du = j
            break

        dt_trial *= 0.95 * eps_du / du_max

    if iter_du == -1:
        v2[:,:] = work_sn2
        raise RuntimeError(f"Step {step_count}: Max iterations exceeded in conduction step.")

    dt_prop = dt_trial

    # Apply evaporation
    # if evap:
    #     evaporate(r, rmid_orig, m, v2_cond, rho, dt_prop) # Modifies rho and m in place
    #     p = v2_cond * rho

    np.multiply(rho, v2, out=work_sn1) # work_sn1 used for p

    ### Step 2: Reestablish hydrostatic equilibrium ###
    status = revirialize_interp_jacobi(
        r, rho, work_sn1, m, m_bh, bkg_param,
        work_nin1, work_nin2, work_nin3, work_nin4, work_nin5, 
        work_n1, work_snout2, work_snout1,
        ) # Modifies r, rho, p in place

    if status == STATUS_SHELL_CROSSING:
        raise RuntimeError(f"Step {step_count}: Shell crossing in conduction/revirialization step.")

    elif status == STATUS_CAVITY_REACHED:
        v2[:,:] = work_sn2 # Restore latest v2
        return STATUS_CAVITY_REACHED

    ### Step 3: Update state variables ###

    # r and rho were modified in place already; no need to assign them
    # Still need to update v2 based on the new p and rho
    np.divide(work_sn1, rho, out=state.v2)
    
    # if evap:
    #     state.m = m

    np.add(r[:, 1:], r[:, :-1], out=state.rmid)
    state.rmid *= 0.5

    state.rho0          = sum_intensive_loglog_single(np.min(state.rmid[:,0]), state.rmid, rho)
    state.r50_spread    = calc_r50_spread(r, m, state.r50evo)

    # Diagnostics
    state.n_iter_du     += iter_du
    state.dt_cum        += float(dt_prop)
    state.du_max_cum    += float(du_max)

    state.du_max  = float(du_max)
    state.dt = float(dt_prop)
    state.t += float(dt_prop)

    return STATUS_OK