### Old routine to compute the pressures for HE.  assumed aligned grids

@njit(types.Tuple((float64[:, :], float64, float64))(float64[:, :], float64[:, :], float64[:, :], float64[:,:], float64[:]),
      fastmath=True, cache=True)
def compute_he_pressures(r, rho, p, m, bkg_param):
    """
    Compute pressure array such that the density profile will be in HE
    This is used after grid initialization to ensure stability in 
    early revirializations.
    Here we assume that radial grid points are aligned.

    Arguments
    ---------
    r : ndarray, shape (s, N+1)
        Edge radii per species.
    rho : ndarray, shape (s, N)
        Shell densities per species.
    p : ndarray, shape (s, N)
        Shell pressures per species.
    m_tot : ndarray, shape (s, N+1)
        Total enclosed mass at edges per species.
    bkg_param : ndarray, shape (4,)
        Parameters for background potential

    Returns
    -------
    p_new : ndarray, shape (s, N+1)
        Updated shell pressure per species.
    res_old : float
        HE residual of input arrays.
    res_new : float
        HE residual of output arrays.
    """
    s, Np1 = r.shape
    N = Np1 - 1
    m_tot = m.sum(axis=0)
    if bkg_param[0] != -1:
        m_tot += add_bkg_pot(r[0], bkg_param)

    # Compute residual of input profile
    res_old = compute_he_resid_norm(r, rho, p, m, bkg_param)

    # Backward sweep to update pressures
    p_new = p.copy()
    for i in range(N-2, -1, -1):
        srho = rho[:, i+1] + rho[:, i]
        dr = r[:, i+2] - r[:, i]
        p_new[:, i] = p_new[:, i+1] + srho * dr * m_tot[i+1] / (4.0 * r[:, i+1]**2)

    # Compute residual of output profile
    res_new = compute_he_resid_norm(r, rho, p_new, m, bkg_param)

    return p_new, res_old, res_new

### Old out of place luminosity calculator

@njit( float64[:, :](
    float64, float64[:, :], float64[:, :], float64[:, :], float64[:], float64[:,:]), 
      cache=True, fastmath=True 
)
def compute_luminosities(c2, r, u, rho, mrat, lnL) -> np.ndarray: 
    """
    Compute luminosity of each shell interface based on temperature gradient and conductivity. 
    e.g, Eq. (43) in Zhong and Shapiro (2025).
    
    Arguments
    ----------
    c2 : float
        Constant 'c2' in the luminosity formula.
    r : ndarray
        Radial grid points, including cell edges.
    u : ndarray
        Specific internal energy for each cell.
    rho : ndarray
        Density for each cell.
    mrat : ndarray
        Mass ratio of species (length = s)
    lnL : ndarray lnL array.
    
    Returns
    -------
    L: ndarray
        Luminosities at each shell boundary (same length as r).
    """
    s, Np1 = r.shape
    N = Np1 - 1
    L = np.zeros((s, Np1), np.float64) # Initialization takes care of boundary conditions
    
    for n in range(s):
        r_n = r[n] # (N+1,)
        u_n = u[n] # (N,)
        rho_n = rho[n] # (N,)
        
        # Centered interface values
        rhom = interp_linear_to_interfaces(r_n, rho_n)
        umed = interp_linear_to_interfaces(r_n, u_n)
        
        # dT/dr at interfaces
        dTdr = np.empty(N-1, dtype=np.float64)
        dTdr[0] = (u_n[1] - u_n[0]) / (r_n[2] - r_n[1])
        dTdr[1:] = 2.0 * (u_n[2:] - u_n[1:-1]) / (r_n[3:] - r_n[1:-2])
        
        # Prefactors
        fac = (r[n, 1:-1]**2) * (rhom / np.sqrt(umed))
        pref = (-c2) * (mrat[n] * lnL[n,n])
        
        L[n, 1:-1] = pref * fac * dTdr
        
    return L

# Old out of place combined explicit heat conduction

@njit(types.Tuple((float64[:, :], float64[:, :], float64, float64,))
      (float64[:, :], float64[:, :], float64[:, :], float64[:, :],
        float64[:, :], float64[:], float64[:, :], float64, float64, float64),
    cache=True)
def conduct_heat(m, u, rho, lum, lnL, mrat, r, dt_prop, eps_du, c1) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Conduct heat and update internal energies (u) and pressures (p) for all species.

    This routine applies two contributions to du/dt:
      - Intra-species conduction: flux divergence of interface luminosities (lum).
      - Inter-species heat exchange: volume-overlap weighted exchange using lnL and mrat.

    PdV work is ignored and density is held fixed. An adaptive limiter is applied
    to ensure the maximum relative change in u does not exceed eps_du; if needed the
    proposed dt_prop is scaled down and dt_eff returned.

    Notes
    -----
    - Luminosities `lum` are defined at cell interfaces (length N+1 per species).
    - Pressure is recomputed as p = (2/3) * rho * u_new.
    - The function returns the updated pressure array, the observed maximum relative
      change in u (dumax), and the effective timestep used (dt_eff). The updated
      internal energy array (u_new) is not returned.

    Parameters
    ----------
    m : ndarray, shape (s, N+1)
        Enclosed mass at cell interfaces for each species.
    u : ndarray, shape (s, N)
        Specific internal energy per cell for each species.
    rho : ndarray, shape (s, N)
        Density per cell for each species (assumed fixed here).
    lum : ndarray, shape (s, N+1)
        Luminosity at cell interfaces for each species (used to compute flux divergence).
    lnL : ndarray, shape (s, s)
        Inter-species coupling coefficients (matrix of lnL[n,k]).
    mrat : ndarray, shape (s,)
        Mass-ratio-like coefficient for each species.
    r : ndarray, shape (s, N+1)
        Radial coordinates including cell interfaces for each species.
    dt_prop : float
        Proposed timestep to apply for this update.
    eps_du : float
        Maximum allowed relative change in u (used by adaptive limiter).
    c1 : float
        Multiplicative constant for inter-species exchange term.

    Returns
    -------
    p_new : ndarray, shape (s, N)
        Updated pressure array after applying du and recomputing p = (2/3) * rho * u_new.
    v2_new : ndarray, shape (s, N)
        Updated v2 array after applying du and recomputing v2 = (2/3) u_new.
    dumax : float
        Observed maximum relative change in u (after any scaling).
    dt_eff : float
        Effective timestep actually used (dt_prop scaled by limiter if needed).
    """

    s, Np1 = m.shape
    N = Np1 - 1

    # Outputs
    u_new   = np.empty_like(u)
    v2_new  = np.empty_like(u)
    p_new   = np.empty_like(u)  # same (s, N) shape
    du      = np.empty_like(u)

    # ---------- intra-species conduction (flux divergence in mass coord) ----------
    # dudt_cond[n,i] = - (L[n,i+1]-L[n,i]) / (M[n,i+1]-M[n,i])
    dudt_cond = np.empty_like(u)
    for n in range(s):
        for i in range(N):
            dm = m[n, i+1] - m[n, i]
            dudt_cond[n, i] = - (lum[n, i+1] - lum[n, i]) / dm

    # ---------- inter-species heat exchange with volume-overlap weighting ----------
    # For each (n,i), sweep all species k!=n and all their cells j
    #   overlap [max(r_n_i, r_k_j), min(r_n_ip1, r_k_jp1)]
    #   VolRat = overlap_volume / vol(cell n,i)
    #   contribution ~ lnL[n,k] * rho[k,j] * VolRat * (mrat[k]*u[k,j] - mrat[n]*u[n,i]) / (u[k,j]+u[n,i])^(3/2)
    dudt_hex = np.zeros_like(u)

    for n in range(s):
        mrn = mrat[n]

        # precompute r^3 arrays for n and reuse
        rn3 = np.empty(N+1, dtype=np.float64)
        for i in range(N+1):
            rr = r[n, i]
            rn3[i] = rr*rr*rr

        for k in range(s):
            if k == n:
                continue
            lnL_nk = lnL[n, k]
            mrk    = mrat[k]

            # precompute r^3 for species k too
            rk3 = np.empty(N+1, dtype=np.float64)
            for j in range(N+1):
                rr = r[k, j]
                rk3[j] = rr*rr*rr

            # two-pointer sweep over i (species n) and j (species k)
            i = 0
            j = 0
            while i < N and j < N:
                rn_lo = r[n, i]
                rn_hi = r[n, i+1]
                rk_lo = r[k, j]
                rk_hi = r[k, j+1]

                # compute overlap interval
                # if no overlap, advance the pointer with the smaller upper bound
                if rk_hi <= rn_lo:
                    j += 1
                    continue
                if rn_hi <= rk_lo:
                    i += 1
                    continue

                # they overlap: [max(lo), min(hi)]
                # rrmin = max(rn_lo, rk_lo), rrmax = min(rn_hi, rk_hi)
                rrmin = rn_lo if rn_lo >= rk_lo else rk_lo
                rrmax = rn_hi if rn_hi <= rk_hi else rk_hi

                rrmin3 = rrmin*rrmin*rrmin
                rrmax3 = rrmax*rrmax*rrmax

                # shell volume of (n,i) in r^3 units, and overlap volume
                vol_n_r3 = rn3[i+1] - rn3[i]
                if vol_n_r3 > 0.0:
                    dvol_overlap_r3 = rrmax3 - rrmin3
                    if dvol_overlap_r3 > 0.0:
                        vol_ratio = dvol_overlap_r3 / vol_n_r3

                        un = u[n, i]
                        uj = u[k, j]
                        denom_uj_un = uj + un
                        if denom_uj_un > 0.0:
                            root = np.sqrt(denom_uj_un)
                            inv_p32 = 1.0 / (denom_uj_un * root)
                            term = lnL_nk * rho[k, j] * vol_ratio * (mrk * uj - mrn * un) * inv_p32
                            dudt_hex[n, i] += term
                # advance the pointer whose cell ends first
                if rn_hi <= rk_hi:
                    i += 1
                else:
                    j += 1

    dudt_hex *= c1

    # ---------- combine and propose update ----------
    dudt = dudt_cond + dudt_hex
    du[:, :] = dudt * dt_prop

    # ---------- adaptive limiter on max relative change ----------
    floor = 1e-40
    dumax = 0.0
    for n in range(s):
        for i in range(N):
            denom = abs(u[n, i])
            if denom < floor:
                denom = floor
            rat = abs(du[n, i]) / denom
            if rat > dumax:
                dumax = rat

    if dumax > eps_du:
        scale = 0.95 * (eps_du / dumax)
        du *= scale
        dumax *= scale
        dt_eff = dt_prop * scale
    else:
        dt_eff = dt_prop

    # ---------- update u and p ----------
    for n in range(s):
        for i in range(N):
            u_new[n, i] = u[n, i] + du[n, i]
            v2_new[n, i] = (2.0 / 3.0) * u_new[n, i]
            p_new[n, i] = (2.0 / 3.0) * rho[n, i] * u_new[n, i]

    return p_new, v2_new, float(dumax), float(dt_eff)


### OLD Machinery to throttle the output
from collections import deque

# --- oscillation detection and throttling ---
auto = io.write_auto

osc_window = 5
osc_threshold_on  = 10 #  100     # avg spacing < 100 steps → oscillation detected
osc_threshold_off = 10_000  # avg spacing > 10k steps → stable again

min_prof_spacing = 250_000
min_tevol_spacing = 50_000

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

# --- PROFILE OUTPUT ---
if auto:
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

else:
    if prof_last_write is None or (step_count - prof_last_write >= min_prof_spacing):
        prof_last_write = step_count
        write_profile_snapshot(state)

# --- TIME EVOLUTION OUTPUT ---

         if auto:
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

        else:
            if tevol_last_write is None or (step_count - tevol_last_write >= min_tevol_spacing):
                tevol_last_write = step_count
                write_time_evolution(state)