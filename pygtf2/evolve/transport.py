import numpy as np 
from numba import njit, float64, types, void
from pygtf2.util.interpolate import interp_linear_to_interfaces
from pygtf2.util.calc import solve_tridiagonal_thomas

_TINY64 = np.finfo(np.float64).tiny

### EXPLICIT METHOD

@njit(void(float64, float64[:, :], float64[:, :], float64[:, :], float64[:], float64[:, :], float64[:, :]), cache=True, fastmath=True)
def compute_luminosities(c2, r, v2, rho, mrat, lnL, lum):
    """ 
    Compute luminosity of each shell interface based on temperature gradient and conductivity.
    In place update.
    e.g, Eq. (43) in Zhong and Shapiro (2025).

    Arguments
    ----------
    c2 : float
        Constant 'c2' in the luminosity formula.
    r : ndarray
        Radial grid points, including cell edges.
    v2 : ndarray
        Square of 1D velocity dispersion for each cell. u = 1.5*v2.
    rho : ndarray
        Density for each cell.
    mrat : ndarray
        Mass ratio of species (length = s)
    lnL : ndarray
        lnL array.
    lum : ndarray
        Luminosities at each shell boundary (same length as r).
        Updated in-place.
    """
    s, Np1 = r.shape
    N = Np1 - 1

    for n in range(s):
        lum[n, 0] = 0.0
        lum[n, N] = 0.0

        r_n   = r[n]
        v2_n  = v2[n]
        rho_n = rho[n]

        rhom = interp_linear_to_interfaces(r_n, rho_n)
        umed = 1.5 * interp_linear_to_interfaces(r_n, v2_n)

        pref = (-c2) * (mrat[n] * lnL[n, n])

        for i in range(N - 1):
            if i == 0:
                dTdr = 1.5 * (v2_n[1] - v2_n[0]) / (r_n[2] - r_n[1])
            else:
                dTdr = 3.0 * (v2_n[i + 1] - v2_n[i]) / (r_n[i + 2] - r_n[i])

            fac = (r_n[i + 1] * r_n[i + 1]) * (rhom[i] / np.sqrt(umed[i]))
            lum[n, i + 1] = pref * fac * dTdr

@njit(void(float64[:, :], float64[:, :], float64[:, :]), cache=True, fastmath=True)
def add_dv2dt_conduction(m, lum, dv2dt):
    """
    Add the intra-species conduction contribution to dv2/dt.

    This routine computes the flux-divergence term from the interface luminosities
    `lum` and accumulates the result into the pre-zeroed array `dv2dt`.

    The update is performed in mass coordinates, with
        dudt = - dL / dm
    and converted to v2 using
        v2 = (2/3) u
    so that
        dv2dt = (2/3) dudt.

    Notes
    -----
    - This routine does not overwrite `dv2dt`; it adds to it.
    - The array `dv2dt` is assumed to have been zeroed before calling this routine.
    - The luminosities `lum` are defined at cell interfaces (length N+1 per species).

    Parameters
    ----------
    m : ndarray, shape (s, N+1)
        Enclosed mass at cell interfaces for each species.
    lum : ndarray, shape (s, N+1)
        Luminosity at cell interfaces for each species.
    dv2dt : ndarray, shape (s, N)
        Array to accumulate the conduction contribution to dv2/dt.
    """

    s, Np1 = m.shape
    N = Np1 - 1

    for n in range(s):
        for i in range(N):
            dm = m[n, i+1] - m[n, i]
            dv2dt[n, i] += -(2.0 / 3.0) * (lum[n, i+1] - lum[n, i]) / dm

@njit(void(float64[:, :], float64[:, :], float64[:, :], float64[:], float64[:, :], float64, float64[:, :]), cache=True, fastmath=True)
def add_dv2dt_hex(v2, rho, lnL, mrat, r, c1, dv2dt):
    """
    Add the inter-species heat-exchange contribution to dv2/dt.

    This routine computes the volume-overlap weighted heat exchange between
    species and accumulates the result into the pre-zeroed array `dv2dt`.

    For each cell (n,i), the contribution from all other species k != n is
    computed by sweeping overlapping shells and weighting by the fractional
    overlap in shell volume. The original exchange formula written in terms
    of internal energy u is converted here to work directly with
        u = 1.5 * v2.

    Notes
    -----
    - This routine does not overwrite `dv2dt`; it adds to it.
    - The array `dv2dt` is assumed to have been zeroed before calling this routine.
    - Both this routine and the conduction routine may be called in either order,
      since both simply accumulate their contribution into `dv2dt`.
    - The overlap sweep uses a two-pointer traversal in radius.

    Parameters
    ----------
    v2 : ndarray, shape (s, N)
        v2 array for each species and cell.
    rho : ndarray, shape (s, N)
        Density per cell for each species.
    lnL : ndarray, shape (s, s)
        Inter-species coupling coefficients.
    mrat : ndarray, shape (s,)
        Mass-ratio-like coefficient for each species.
    r : ndarray, shape (s, N+1)
        Radial coordinates including cell interfaces for each species.
    c1 : float
        Multiplicative constant for the inter-species exchange term.
    dv2dt : ndarray, shape (s, N)
        Array to accumulate the heat-exchange contribution to dv2/dt.
    """

    s, Np1 = r.shape
    N = Np1 - 1

    # Original exchange term was in u. Converting to v2 = (2/3)u gives:
    # dv2dt = c1 * (2/3) * [ ... in u ... ]
    # with u = 1.5*v2, so overall factor becomes:
    #   c1 * ((2/3) / sqrt(1.5))
    prefac = c1 * ((2.0 / 3.0) / np.sqrt(1.5))

    for n in range(s):
        mrn = mrat[n]

        for k in range(s):
            if k == n:
                continue

            lnL_nk = lnL[n, k]
            mrk = mrat[k]

            # two-pointer sweep over i (species n) and j (species k)
            i = 0
            j = 0
            while i < N and j < N:
                rn_lo = r[n, i]
                rn_hi = r[n, i + 1]
                rk_lo = r[k, j]
                rk_hi = r[k, j + 1]

                # compute overlap interval
                if rk_hi <= rn_lo:
                    j += 1
                    continue
                if rn_hi <= rk_lo:
                    i += 1
                    continue

                rrmin = rn_lo if rn_lo >= rk_lo else rk_lo
                rrmax = rn_hi if rn_hi <= rk_hi else rk_hi

                rn_lo3 = rn_lo * rn_lo * rn_lo
                rn_hi3 = rn_hi * rn_hi * rn_hi
                vol_n_r3 = rn_hi3 - rn_lo3

                if vol_n_r3 > 0.0:
                    rrmin3 = rrmin * rrmin * rrmin
                    rrmax3 = rrmax * rrmax * rrmax
                    dvol_overlap_r3 = rrmax3 - rrmin3

                    if dvol_overlap_r3 > 0.0:
                        vol_ratio = dvol_overlap_r3 / vol_n_r3

                        v2n = v2[n, i]
                        v2k = v2[k, j]
                        denom = v2k + v2n

                        if denom > 0.0:
                            root = np.sqrt(denom)
                            inv_p32 = 1.0 / (denom * root)
                            term = (
                                lnL_nk
                                * rho[k, j]
                                * vol_ratio
                                * (mrk * v2k - mrn * v2n)
                                * inv_p32
                            )
                            dv2dt[n, i] += prefac * term

                # advance the pointer whose cell ends first
                if rn_hi <= rk_hi:
                    i += 1
                else:
                    j += 1

@njit(types.Tuple((float64, float64))(float64[:, :], float64[:, :], float64, float64), cache=True, fastmath=True)
def apply_dv2dt(v2, dv2dt, dt_prop, eps_du):
    """
    Apply the accumulated dv2/dt with an adaptive limiter.

    This routine applies the proposed update
        dv2 = dv2dt * dt_prop
    to the input array `v2`, subject to a limiter that ensures the maximum
    relative change in v2 does not exceed `eps_du`. Since u = 1.5*v2, the
    relative change in v2 is identical to the relative change in u.

    Notes
    -----
    - The update is applied in place to `v2`.
    - The limiter is based on the maximum over all cells of
          abs(dv2) / abs(v2),
      with a small floor in the denominator for protection.
    - If the proposed step exceeds the allowed relative change, the timestep
      is scaled by
          0.95 * eps_du / dv2_max.

    Parameters
    ----------
    v2 : ndarray, shape (s, N)
        v2 array to be updated in place.
    dv2dt : ndarray, shape (s, N)
        Total accumulated dv2/dt.
    dt_prop : float
        Proposed timestep to apply for this update.
    eps_du : float
        Maximum allowed relative change in u (equivalently in v2).

    Returns
    -------
    dv2max : float
        Observed maximum relative change after any limiter scaling.
    dt_eff : float
        Effective timestep actually used.
    """

    s, N = v2.shape

    floor = _TINY64
    dv2max = 0.0

    # Find maximum proposed relative change
    for n in range(s):
        for i in range(N):
            dv2 = dv2dt[n, i] * dt_prop

            denom = abs(v2[n, i])
            if denom < floor:
                denom = floor

            rat = abs(dv2) / denom
            if rat > dv2max:
                dv2max = rat

    # Adaptive limiter
    if dv2max > eps_du:
        scale = 0.95 * (eps_du / dv2max)
        dv2max *= scale
        dt_eff = dt_prop * scale
    else:
        dt_eff = dt_prop

    # Apply update in place
    for n in range(s):
        for i in range(N):
            v2[n, i] += dv2dt[n, i] * dt_eff

    return float(dv2max), float(dt_eff)

### IMEX METHOD

@njit(void(float64[:, :], float64[:, :],float64[:, :], float64[:], float64[:, :],float64[:, :], float64, float64), cache=True, fastmath = True)
def compute_hex_dv2(v2, rho, lnL, mrat, r, dv2_work, dt, c1):
    """
    Compute the raw inter-species heat-exchange increment dv2_work
    over timestep dt, without applying it.

    Parameters
    ----------
    v2 : ndarray, shape (s, N)
        One-dimensional velocity dispersion squared for each species.
    rho : ndarray, shape (s, N)
        Density per cell for each species.
    lnL : ndarray, shape (s, s)
        Inter-species coupling coefficients.
    mrat : ndarray, shape (s,)
        Mass-ratio-like coefficient for each species.
    r : ndarray, shape (s, N+1)
        Radial cell interfaces for each species.
    dv2_work : ndarray, shape (s, N)
        Output workspace, overwritten in place.
    dt : float
        Timestep for this raw increment.
    c1 : float
        Multiplicative constant for inter-species exchange term.
    """
    s, N = v2.shape

    # Zero workspace
    for n in range(s):
        for i in range(N):
            dv2_work[n, i] = 0.0

    # Original exchange term was written in u.
    # Converting to v2 = (2/3)u gives:
    # prefac = c1 * ((2/3) / sqrt(1.5))
    prefac = c1 * ((2.0 / 3.0) / np.sqrt(1.5))

    for n in range(s):
        mrn = mrat[n]

        for k in range(s):
            if k == n:
                continue

            lnL_nk = lnL[n, k]
            mrk = mrat[k]

            i = 0
            j = 0
            while i < N and j < N:
                rn_lo = r[n, i]
                rn_hi = r[n, i + 1]
                rk_lo = r[k, j]
                rk_hi = r[k, j + 1]

                if rk_hi <= rn_lo:
                    j += 1
                    continue
                if rn_hi <= rk_lo:
                    i += 1
                    continue

                rrmin = rn_lo if rn_lo >= rk_lo else rk_lo
                rrmax = rn_hi if rn_hi <= rk_hi else rk_hi

                rn_lo3 = rn_lo * rn_lo * rn_lo
                rn_hi3 = rn_hi * rn_hi * rn_hi
                vol_n_r3 = rn_hi3 - rn_lo3

                if vol_n_r3 > 0.0:
                    rrmin3 = rrmin * rrmin * rrmin
                    rrmax3 = rrmax * rrmax * rrmax
                    dvol_overlap_r3 = rrmax3 - rrmin3

                    if dvol_overlap_r3 > 0.0:
                        vol_ratio = dvol_overlap_r3 / vol_n_r3

                        v2n = v2[n, i]
                        v2k = v2[k, j]
                        denom = v2k + v2n

                        if denom > 0.0:
                            root = np.sqrt(denom)
                            inv_p32 = 1.0 / (denom * root)

                            dv2_work[n, i] += (
                                prefac
                                * lnL_nk
                                * rho[k, j]
                                * vol_ratio
                                * (mrk * v2k - mrn * v2n)
                                * inv_p32
                                * dt
                            )

                if rn_hi <= rk_hi:
                    i += 1
                else:
                    j += 1

@njit(float64(float64[:, :], float64[:, :]), cache=True, fastmath=True)
def hex_du_max(v2, dv2_work):
    """
    Compute max |dv2|/|v2| over all species and cells.
    By convention we call this du_max for consistency with the rest
    of the codebase.
    """
    s, N = v2.shape
    floor = _TINY64
    du_max = 0.0

    for n in range(s):
        for i in range(N):
            denom = abs(v2[n, i])
            if denom < floor:
                denom = floor

            rat = abs(dv2_work[n, i]) / denom
            if rat > du_max:
                du_max = rat

    return du_max

@njit(void(float64[:, :], float64[:, :], float64), cache=True)
def apply_scaled_dv2(v2, dv2_work, scale):
    """
    Apply v2 += scale * dv2_work in place.
    """
    s, N = v2.shape

    for n in range(s):
        for i in range(N):
            v2[n, i] += scale * dv2_work[n, i]

@njit(types.Tuple((float64, float64))(float64[:, :], float64[:, :], float64, float64), cache=True, fastmath=True)
def hex_limit_dt(v2, dv2_work, dt, eps_du):
    """
    Compute du_max and the hex-limited effective timestep.
    Assumes dv2_work was computed using the trial dt.
    """
    du_max = hex_du_max(v2, dv2_work)
    dt_eff = dt

    if du_max > eps_du:
        dt_eff = dt * (0.95 * eps_du / du_max)

    return du_max, dt_eff

@njit(void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64, float64), cache=True, fastmath=True)
def build_tridiag_system(a, b, c, d, rk, mk, rhok_int, uk, pref, dt):
    """
    Construct tridiagonal coefficients: a_i du_i-1 + b_i du_i + c_i du_i+1 = d_i
    a, b, c, and d are updated in place

    Arguments
    ---------
    a : ndarray, shape (N,)
        Subdiagonal coefficients (multiply du_{i-1}) for interior nodes j=1..N.
    b : ndarray, shape (N,)
        Main diagonal coefficients (multiply du_i) for interior nodes.
    c : ndarray, shape (N,)
        Superdiagonal coefficients (multiply d_{u+1}) for interior nodes.
    d : ndarray, shape (N,)
        Right-hand side vector for the interior nodes.
    rk : ndarray, shape (N+1,)
        Edge radii.
    mk : ndarray, shape (N+1,)
        Enclosed mass at edges.
    rhok_int : ndarray, shape (N-1,)
        Densities interpolated to shell edges
    uk : ndarray, shape (N,)
        Specific internal energy.
    pref : float
        prefactor for species k
    dt : float
        timestep
    """
    drc     = 0.5 * (rk[2:] - rk[:-2])      # (N-1,)
    delu    = uk[1:] - uk[:-1]              # (N-1,)
    su      = uk[1:] + uk[:-1]              # (N-1,)
    sqrt2   = 1.41421356237309

    # Interior cells
    facL    = rhok_int[:-1] * rk[1:-2]**2 / drc[:-1]
    facR    = rhok_int[1:] * rk[2:-1]**2 / drc[1:]
    su12L   = 1 / np.sqrt(su[:-1])
    su12R   = 1 / np.sqrt(su[1:])
    dusu32L = 0.5 * delu[:-1] / su[:-1]**(3.0/2.0)
    dusu32R = 0.5 * delu[1:] / su[1:]**(3.0/2.0)

    a[1:-1] = facL * ( su12L + dusu32L )
    b[1:-1] = -1 * (
        facR * ( su12R + dusu32R )
        + facL * ( su12L - dusu32L )
        + ( ( mk[2:-1] - mk[1:-2] ) / ( sqrt2 * pref * dt ) )
    )
    c[1:-1] = facR * ( su12R - dusu32R )
    d[1:-1] = (
        facL * delu[:-1] / np.sqrt(su[:-1])
        - facR * delu[1:] / np.sqrt(su[1:])
    )

    # i = 1
    a[0] = 0.0
    b[0] = -1 *  (
        su12L[0] + dusu32L[0]
        + ( mk[1] * drc[0] / ( rhok_int[0] * rk[1]**2 * pref * sqrt2 * dt ) )
    )
    c[0] = su12L[0] - dusu32L[0]
    d[0] = - delu[0] / np.sqrt(su[0])

    # i = N
    a[-1] = su12R[-1] + dusu32R[-1]
    b[-1] = (
        dusu32R[-1] - su12R[-1]
        - ( 
            (mk[-1] - mk[-2]) * drc[-1] 
            / ( rhok_int[-1] * rk[-2]**2 * pref * sqrt2 * dt )
        )
    )
    c[-1] = 0.0
    d[-1] = delu[-1] / np.sqrt(su[-1])

@njit(void(float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64, float64[:], float64[:, :], float64[:, :], float64), cache=True, fastmath=True)
def conduct_implicit(v2, rho, r, m, c2, mrat, lnL, du_trial, dt,):
    """
    Implicit intra-species conduction step on v2.
    Use a fixed dt - no timestep limiting in this step - we find that the hex step limits in almost all cases.

    For each species, solve a tridiagonal system for du, but only commit the
    update once the limiter is satisfied.

    The tridiagonal system is defined by:
        a_i du_i-1 + b_i du_i + c_i du_i+1 = d_i
    
    v2 is updated in-place. u = 1.5 * v2
    """
    s, N = v2.shape

    a = np.empty(N, dtype=np.float64)
    b = np.empty(N, dtype=np.float64)
    c = np.empty(N, dtype=np.float64)
    d = np.empty(N, dtype=np.float64)

    for k in range(s):
        rk       = r[k]
        mk       = m[k]
        rhok     = rho[k]
        uk       = 1.5 * v2[k]
        rhok_int = interp_linear_to_interfaces(rk, rhok)

        pref = c2 * (mrat[k] * lnL[k, k])

        build_tridiag_system(a, b, c, d, rk, mk, rhok_int, uk, pref, dt)
        solve_tridiagonal_thomas(a, b, c, d, du_trial[k])

    for k in range(s):
        for i in range(N):
            v2[k, i] += (2.0 / 3.0) * du_trial[k, i]

@njit(float64(float64[:, :], float64[:, :]), cache=True)
def cond_du_max(v2, du_cond):
    """
    Compute max |du|/|u| over all species and cells, where u = 1.5*v2.
    """
    s, N = v2.shape
    floor = _TINY64
    du_max = 0.0

    for k in range(s):
        for i in range(N):
            u_old = 1.5 * v2[k, i]
            denom = abs(u_old)
            if denom < floor:
                denom = floor

            rat = abs(du_cond[k, i]) / denom
            if rat > du_max:
                du_max = rat

    return du_max

@njit(types.Tuple((float64, float64))(float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64, float64, float64[:], float64[:, :], float64[:, :], float64[:, :], float64, float64, types.int64), cache=True, fastmath=True)
def conduction_imex(
    v2, rho, r, m, c1, c2, mrat, lnL,
    dv2_hex_work, du_cond_work, dt, eps_du, order,
    ) -> tuple[np.float64, np.float64]:
    """
    Apply one split conduction step with a shared timestep determined by
    the heat-exchange (hex) operator.

    order:
        0 -> hex(dt)      then cond(dt)
        1 -> cond(dt)     then hex(dt)
             but dt is still chosen from hex on the initial state
        2 -> Strang:
             hex(dt/2) -> cond(dt) -> hex(dt/2)
             where dt is chosen from the initial half-step hex limiter

    Returns
    -------
    du_max : float
        Maximum realized fractional change from either operator.
        For hex this is max |dv2|/|v2|, and for conduction it is
        max |du|/|u|. These are equivalent fractional diagnostics
        under your naming convention.
    dt_eff : float
        Effective shared timestep accepted for this split step.
        For Strang this is the full-step dt_eff, so each hex half-step
        uses 0.5 * dt_eff.
    """

    du_hex_max = 0.0
    du_cond_max = 0.0
    dt_eff = dt

    # ------------------------------------------------------------
    # order == 0 : hex first, then conduction
    # ------------------------------------------------------------
    if order == 0:
        compute_hex_dv2(v2, rho, lnL, mrat, r, dv2_hex_work, dt, c1)
        du_hex_trial = hex_du_max(v2, dv2_hex_work)

        if du_hex_trial > eps_du:
            dt_eff = dt * (0.95 * eps_du / du_hex_trial)

        scale = 1.0
        if dt > 0.0:
            scale = dt_eff / dt

        apply_scaled_dv2(v2, dv2_hex_work, scale)
        du_hex_max = du_hex_trial * scale

        conduct_implicit(v2, rho, r, m, c2, mrat, lnL, du_cond_work, dt_eff)
        du_cond_max = cond_du_max(v2, du_cond_work)

    # ------------------------------------------------------------
    # order == 1 : conduction first, then hex
    # dt still chosen from hex on the initial state
    # ------------------------------------------------------------
    elif order == 1:
        compute_hex_dv2(v2, rho, lnL, mrat, r, dv2_hex_work, dt, c1)
        du_hex_trial = hex_du_max(v2, dv2_hex_work)

        if du_hex_trial > eps_du:
            dt_eff = dt * (0.95 * eps_du / du_hex_trial)

        conduct_implicit(v2, rho, r, m, c2, mrat, lnL, du_cond_work, dt_eff)
        du_cond_max = cond_du_max(v2, du_cond_work)

        scale = 1.0
        if dt > 0.0:
            scale = dt_eff / dt

        apply_scaled_dv2(v2, dv2_hex_work, scale)
        du_hex_max = du_hex_trial * scale

    # ------------------------------------------------------------
    # order == 2 : Strang splitting
    # dt chosen from initial half-step hex limiter
    # ------------------------------------------------------------
    else:
        half_dt = 0.5 * dt

        # First half-step hex on initial state determines dt_eff
        compute_hex_dv2(v2, rho, lnL, mrat, r, dv2_hex_work, half_dt, c1)
        du_hex1_trial = hex_du_max(v2, dv2_hex_work)

        half_dt_eff = half_dt
        if du_hex1_trial > eps_du:
            half_dt_eff = half_dt * (0.95 * eps_du / du_hex1_trial)

        dt_eff = 2.0 * half_dt_eff

        scale1 = 1.0
        if half_dt > 0.0:
            scale1 = half_dt_eff / half_dt

        apply_scaled_dv2(v2, dv2_hex_work, scale1)
        du_hex1_max = du_hex1_trial * scale1

        # Full conduction step with the shared full dt_eff
        conduct_implicit(v2, rho, r, m, c2, mrat, lnL, du_cond_work, dt_eff)
        du_cond_max = cond_du_max(v2, du_cond_work)

        # Recompute second half-step hex on the updated state,
        # but do not re-limit dt; just apply the accepted half-step.
        compute_hex_dv2(v2, rho, lnL, mrat, r, dv2_hex_work, half_dt_eff, c1)
        du_hex2_max = hex_du_max(v2, dv2_hex_work)

        apply_scaled_dv2(v2, dv2_hex_work, 1.0)

        du_hex_max = du_hex1_max
        if du_hex2_max > du_hex_max:
            du_hex_max = du_hex2_max

    du_max = du_hex_max
    if du_cond_max > du_max:
        du_max = du_cond_max

    return float(du_max), float(dt_eff)