import numpy as np 
from pygtf2.io.write import write_profile_snapshot, write_log_entry, write_time_evolution
from pygtf2.evolve.transport import compute_luminosities, add_dv2dt_conduction, add_dv2dt_hex, apply_dv2dt, conduction_imex
from pygtf2.evolve.hydrostatic import revirialize_interp_gs, revirialize_interp_jacobi, STATUS_SHELL_CROSSING
from pygtf2.evolve.evaporate import evaporate
from pygtf2.evolve.binaries import binaries_heating
from pygtf2.util.calc import calc_rho_v2_r_c, calc_r50_spread, compute_rc_frac
from pygtf2.dev.debug import plot_r_markers

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

    # --- Locals ---
    config = state.config
    io = config.io
    sim = config.sim
    prec = config.prec
    char  = state.char

    # Switches
    # evap = sim.evap
    # binaries = sim.binaries
    conduct_imex = sim.conduct_imex

    # Parameters
    c1 = float(char.c1); c2 = float(char.c2)
    eps_du = float(prec.eps_du); eps_dt = prec.eps_dt
    mrat = state.mrat; lnL = char.lnL
    bkg_param = state.bkg_param

    # Preallocations
    p = np.empty_like(state.rho, dtype=np.float64)

    # Output options
    chatter = bool(io.chatter); t_halt = float(sim.t_halt); nlog = int(io.nlog); nupdate = int(io.nupdate)
    rho0_last_prof = float(state.rho_c); rho0_last_tevol = float(state.rho_c)
    r50_spread_last_tevol = float(state.r50_spread)
    rho_c_halt = float(sim.rho_c_halt)
    drho_prof = float(io.drho_prof); drho_tevol = float(io.drho_tevol)
    if np.allclose(mrat, mrat[0]):
        use_r50 = False
    else:
        dr50_tevol = float(io.dr50_tevol)
        use_r50 = True

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
        # integrate_time_step(state, dt_prop, step_count)
        integrate_time_step(state, dt_prop, step_count,
                            conduct_imex, # evap, binaries,         # Switches
                            eps_du, c1, c2, mrat, lnL, bkg_param,   # Parameters
                            p,                                      # Preallocated array
                            )

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
        if drho_for_prof > drho_prof:
            rho0_last_prof = rho0
            write_profile_snapshot(state)

        # --- TIME EVOLUTION OUTPUT ---
        drho_for_tevol = abs((rho0 / rho0_last_tevol) - 1.0)
        want_tevol = drho_for_tevol > drho_tevol

        if use_r50:
            denom = r50_spread_last_tevol if abs(r50_spread_last_tevol) > 1e-100 else 1e-100
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

    if state.t >= t_halt:
        if chatter:
            print("Simulation halted: max time exceeded")

def integrate_time_step(state, dt_prop, step_count,     # State
                        conduct_imex, # evap, binaries,   # Switches
                        eps_du,
                        c1, c2, mrat, lnL, bkg_param,   # Parameters
                        p,                              # Preallocation    
                        ):
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
    # Allocations
    r           = np.asarray(state.r,       dtype=np.float64)
    # rmid_orig   = np.asarray(state.rmid,    dtype=np.float64)
    m           = np.asarray(state.m,       dtype=np.float64)
    v2          = np.asarray(state.v2,      dtype=np.float64)
    rho         = np.asarray(state.rho,     dtype=np.float64)

    ### Step 1: Energy transport ###

    # IMEX METHOD
    if conduct_imex:
        # choose order: 0 = hex -> cond; 1 = cond -> hex; 2 = strang
        order = 2
        dv2_hex_work = np.empty_like(v2)
        du_cond_work = np.empty_like(v2)
        du_max, dt_prop = conduction_imex(
            v2, rho, r, m,
            c1, c2, mrat, lnL,
            dv2_hex_work, du_cond_work,
            dt_prop, eps_du, order,
        )

    # EXPLICIT METHOD
    else:
        lum     = np.zeros_like(r,  dtype=np.float64)
        dv2dt   = np.zeros_like(v2, dtype=np.float64)
        compute_luminosities(c2, r, v2, rho, mrat, lnL, lum)
        add_dv2dt_conduction(m, lum, dv2dt)
        add_dv2dt_hex(v2, rho, lnL, mrat, r, c1, dv2dt)
        du_max, dt_prop = apply_dv2dt(v2, dv2dt, dt_prop, eps_du)

    # Apply evaporation
    # if evap:
    #     evaporate(r, rmid_orig, m, v2_cond, rho, dt_prop) # Modifies rho and m in place
    #     p = v2_cond * rho

    # Apply heating
    # if binaries:
    #     v2_cond, p, eps_max = binaries_heating(rmid_orig, rho, v2_cond, dt_prop)

    ### Step 2: Reestablish hydrostatic equilibrium ###
    p[:,:] = rho * v2

    status = revirialize_interp_jacobi(r, rho, p, m, bkg_param) # Modifies r, rho, p in place

    # Shell crossing
    if status == STATUS_SHELL_CROSSING:
        raise RuntimeError(f"Step {step_count}: Shell crossing in conduction/revirialization step.")
        
    ### Step 3: Update state variables ###

    state.r         = r
    state.rho       = rho
    v2_new          = p / rho
    state.v2        = v2_new
    state.du_max    = du_max
    # if evap:
    #     state.m = m

    rmid            = 0.5 * (r[:, 1:] + r[:, :-1])
    state.rmid      = rmid
    state.trelax    = v2_new**(3.0/2.0) / rho

    state.mintrelax                     = float(np.min(state.trelax))
    state.rho_c, state.v2_c, state.r_c  = calc_rho_v2_r_c(rmid, rho, v2_new)
    state.r50_spread                    = calc_r50_spread(r, m, state.r50evo)
    # compute_rc_frac(r, m, state.r_c, state.rc_frac)

    # Diagnostics
    state.dt_cum                += float(dt_prop)
    state.du_max_cum            += float(du_max)
    state.dt_over_trelax_cum    += float(dt_prop / state.mintrelax)

    state.dt = float(dt_prop)
    state.t += float(dt_prop)