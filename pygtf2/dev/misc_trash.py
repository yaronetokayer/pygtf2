### EXPLICIT METHOD

_TINY64 = np.finfo(np.float64).tiny

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

### OLD FOR IMEX

@njit(void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64, float64), cache=True, fastmath=True)
def build_tridiag_system_VEC(a, b, c, d, rk, mk, rhok_int, uk, pref, dt):
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
        Superdiagonal coefficients (multiply du_{i+1}) for interior nodes.
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

@njit(
    types.Tuple((float64, float64, types.int64))(
        float64[:, :],   # v2
        float64[:, :],   # rho
        float64[:, :],   # r
        float64[:, :],   # m
        float64,         # c1
        float64,         # c2
        float64[:],      # mrat
        float64[:, :],   # lnL
        float64[:, :],   # v2_work
        float64[:, :],   # v2_old_work
        float64[:],      # a_work
        float64[:],      # b_work
        float64[:],      # c_work
        float64[:],      # d_work
        float64,         # dt
        float64,         # eps_du
        types.int64,     # order
        types.int64,     # max_iter
    ),
    cache=True, fastmath=True,
)
def conduct_imex_dulim(
    v2, rho, r, m,
    c1, c2, mrat, lnL,
    v2_work, v2_old_work, a_work, b_work, c_work, d_work,
    dt, eps_du, order, max_iter,
):
    """
    Apply one split conduction step with a timestep limited by the net
    fractional change across the complete IMEX step.

    order:
        HEX_FIRST -> hex(dt)      then cond(dt)
        COND_FIRST -> cond(dt)     then hex(dt)
        STRANG_SPLIT -> Strang:
             hex(dt/2) -> cond(dt) -> hex(dt/2)
    The trial is accepted when

        max_{species, shell}
            |v2_final - v2_initial|
            / max(v2_initial, tiny)

        <= eps_du

    Returns
    -------
    du_max : float
        Maximum net fractional change over the accepted full split step.
    dt_used : float
        Accepted trial timestep. On failure, this is the next reduced
        timestep estimate.
    du_iter : int
        Zero-based iteration on which the step was accepted, or -1 if
        no timestep was accepted after max_iter attempts.
    """
    s, N = v2.shape

    tiny = _TINY64
    safety = 0.95

    # Save the original state once. Every trial starts from this state.
    for k in range(s):
        for i in range(N):
            v2_old_work[k, i] = v2[k, i]

    dt_trial = dt
    du_max = 0.0

    for j in range(max_iter):
        # The first trial already starts from the original state.
        # Subsequent trials must undo the rejected trial completely.
        if j > 0:
            for k in range(s):
                for i in range(N):
                    v2[k, i] = v2_old_work[k, i]

        if order == HEX_FIRST:
            hex_explicit(v2, rho, lnL, mrat, r, v2_work, dt_trial, c1,)
            conduct_implicit(v2, rho, r, m, a_work, b_work, c_work, d_work, c2, mrat, lnL, v2_work, dt_trial)

        elif order == COND_FIRST:
            conduct_implicit(v2, rho, r, m, a_work, b_work, c_work, d_work, c2, mrat, lnL, v2_work, dt_trial)
            hex_explicit(v2, rho, lnL, mrat, r, v2_work, dt_trial, c1,)

        elif order == STRANG_SPLIT:
            half_dt_trial = 0.5 * dt_trial

            # First half-step hex on initial state determines dt_eff
            hex_explicit(v2, rho, lnL, mrat, r, v2_work, half_dt_trial, c1,)

            conduct_implicit(v2, rho, r, m, a_work, b_work, c_work, d_work, c2, mrat, lnL, v2_work, dt_trial,)

            # Recompute the second half-step on the updated state.
            hex_explicit(v2, rho, lnL, mrat, r, v2_work, half_dt_trial, c1,)

        else:
            raise ValueError("unrecognized order code in conduction step")

        # ------------------------------------------------------------
        # Measure the net change across the complete split step.
        # The denominator is always the original pre-step value.
        # ------------------------------------------------------------
        du_max = 0.0

        for k in range(s):
            for i in range(N):
                v2_old = v2_old_work[k, i]

                denom = v2_old
                if denom <= tiny:
                    denom = tiny

                du = abs(v2[k, i] - v2_old) / denom

                if du > du_max:
                    du_max = du

        # Accept the complete step.
        if du_max <= eps_du:
            return du_max, dt_trial, j

        # Estimate a smaller timestep and rerun the complete step.
        dt_trial *= safety * eps_du / du_max

    # No trial was accepted. Do not leave the last rejected state in v2.
    for k in range(s):
        for i in range(N):
            v2[k, i] = v2_old_work[k, i]

    return du_max, dt_trial, -1

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

@njit(void(float64[:, :], float64[:, :], float64), cache=True)
def apply_scaled_dv2(v2, dv2_work, scale):
    """
    Apply v2 += scale * dv2_work in place.
    """
    s, N = v2.shape

    for n in range(s):
        for i in range(N):
            v2[n, i] += scale * dv2_work[n, i]

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

### Old routines from HE

@njit(void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:] ), cache=True, fastmath=True,)
def build_tridiag_system(r, rho, p, m_tot, a, b, c, y):
    """
    Fill preallocated arrays a, b, c, y with the tridiagonal system
    for the interior fractional radius shifts.

    Parameters
    ----------
    r : ndarray, shape (N,)
        Edge radii.
    rho : ndarray, shape (N-1,)
        Shell-centered densities.
    p : ndarray, shape (N-1,)
        Shell-centered pressures.
    m_tot : ndarray, shape (N,)
        Total enclosed mass at the same edge radii as `r`.
    a, b, c, y : ndarray, shape (N-2,)
        Preallocated output arrays to fill in place.

    Notes
    -----
    - The unknown vector x contains the interior fractional displacements x_j = Δr_j / r_j
      (excluding the fixed inner and outer edges), so the returned arrays all have length N-2.
    - The routine linearizes the hydrostatic update using finite differences and geometric
      volume factors. Small numerical floors are applied to pressure differences and density sums
      to prevent divide-by-zero or overflow. The outputs are arranged for direct use with the
      tridiagonal solver used elsewhere in this module.
    """
    tiny = _TINY64

    n = r.shape[0] - 2  # number of interior unknowns

    for j in range(n):
        iL = j
        iC = j + 1
        iR = j + 2

        rL = r[iL]
        rC = r[iC]
        rR = r[iR]

        dr = rR - rL
        inv_dr = 1.0 / dr

        dP = p[j + 1] - p[j]
        drho = rho[j + 1] + rho[j]

        # floors to avoid divide-by-zero / inf
        if abs(dP) < tiny:
            if dP >= 0.0:
                dP = tiny
            else:
                dP = -tiny

        if drho < tiny:
            drho = tiny

        rL3 = rL * rL * rL
        rC3 = rC * rC * rC
        rR3 = rR * rR * rR

        r3a = rR3 / (rR3 - rC3)
        r3c = rC3 / (rC3 - rL3)
        r3b = r3a - 1.0
        r3d = r3c - 1.0

        q1 = rR * inv_dr
        q2 = q1 - 1.0

        dd = -(4.0 / m_tot[iC]) * ((rC * rC) * inv_dr) * (dP / drho)

        c1 = 5.0 * dd * (p[j + 1] / dP) - 3.0 * (rho[j + 1] / drho)
        c2 = 5.0 * dd * (p[j] / dP)     + 3.0 * (rho[j] / drho)

        y[j] = dd - 1.0
        a[j] = r3d * c2 - q2
        b[j] = -2.0 - r3b * c1 - r3c * c2
        c[j] = r3a * c1 + q1

@njit(types.void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:]), cache=True, fastmath=False)
def build_tridiag_system_log(r, rho, p, m_tot, a, b, c, y):
    """
    Fill preallocated arrays ``a``, ``b``, ``c``, and ``y`` with the
    tridiagonal system A·x = y for interior radial corrections in the
    logarithmic form of the HE equation.

    Arguments
    ---------
    r : ndarray
        Radial edge coordinates, length = n + 1.
    rho : ndarray
        Shell-centered densities, length = n.
    p : ndarray
        Shell-centered pressures, length = n.
    m_tot : ndarray
        Total enclosed mass at edge points, length = n + 1.
    a : ndarray
        Preallocated output array of length n - 1 for the subdiagonal
        entries, ``a[i] = A[i, i-1]``.
    b : ndarray
        Preallocated output array of length n - 1 for the main diagonal
        entries, ``b[i] = A[i, i]``.
    c : ndarray
        Preallocated output array of length n - 1 for the superdiagonal
        entries, ``c[i] = A[i, i+1]``.
    y : ndarray
        Preallocated output array of length n - 1 for the right-hand side.

    Notes
    -----
    This function performs in-place updates only and returns nothing.

    The output arrays must already be allocated with length ``n - 1``,
    where ``n = len(rho) = len(p)``.
    """
    tiny = _TINY64
    n_out = p.shape[0] - 1

    # Nothing to do if there are no interior unknowns.
    if n_out <= 0:
        return

    # ------------------------------------------------------------------
    # Row 0 is *not* assembled using the general formula, because we
    # overwrite it with the special boundary condition anyway.
    #
    # The main loop below handles only i >= 1.
    # ------------------------------------------------------------------
    if n_out > 1:
        # Initialize the sliding window for the first interior row that
        # is actually assembled by the general formula, namely i = 1:
        #
        #   left  edge -> r[1]
        #   center edge -> r[2]
        #   right edge -> r[3]
        #
        # We also cache powers and logs that can be rolled forward from
        # one iteration to the next to reduce repeated work.
        rL = r[1]
        rC = r[2]
        rR = r[3]

        rL3 = rL * rL * rL
        rC2 = rC * rC
        rC3 = rC2 * rC
        rR3 = rR * rR * rR

        log_rL = math.log(rL)
        log_rC = math.log(rC)
        log_rR = math.log(rR)

        log_pL = math.log(p[1])
        log_pR = math.log(p[2])

        for i in range(1, n_out):
            # Geometry factors for the current 3-point stencil.
            denomL = rC3 - rL3
            denomR = rR3 - rC3
            inv_denomL = 1.0 / denomL
            inv_denomR = 1.0 / denomR

            rL3rL3 = rL3 * inv_denomL
            rR3rC3 = rR3 * inv_denomR
            rC3rC3 = rC3 * inv_denomR
            rC3rL3 = rC3 * inv_denomL
            rC2rC3 = rC2 * inv_denomR
            rC2rL3 = rC2 * inv_denomL

            # Local thermodynamic state.
            pL = p[i]
            pR = p[i + 1]
            rhoL = rho[i]
            rhoR = rho[i + 1]

            dlnp = log_pR - log_pL
            dlnr = 0.5 * (log_rR - log_rL)

            sp = pL + pR
            sr = rhoL + rhoR

            # Floors to avoid divide-by-zero/inf.
            if abs(sp) < tiny:
                sp = math.copysign(tiny, sp)
            if abs(dlnr) < tiny:
                dlnr = math.copysign(tiny, dlnr)

            inv_sp = 1.0 / sp
            inv_dlnr = 1.0 / dlnr
            inv_dlnr2 = inv_dlnr * inv_dlnr

            dpdr = 0.5 * dlnp * inv_dlnr2
            srsp = sr * inv_sp

            m_edge = m_tot[i + 1]
            mr = m_edge / rC
            m_over_sp = m_edge * inv_sp
            mr_over_sp = mr * inv_sp

            afac = 5.0 * inv_dlnr + mr_over_sp * (5.0 * pL * srsp - 3.0 * rhoL)
            cfac = 5.0 * inv_dlnr + mr_over_sp * (3.0 * rhoR - 5.0 * pR * srsp)

            bfac3 = 5.0 * pR * srsp - 3.0 * rhoR
            bfac4 = 3.0 * rhoL - 5.0 * pL * srsp

            a[i] = dpdr - rL3rL3 * afac
            b[i] = (
                5.0 * inv_dlnr * (rC3rC3 + rC3rL3)
                - mr * srsp
                - m_over_sp * (rC2rC3 * bfac3 + rC2rL3 * bfac4)
            )
            c[i] = -dpdr - rR3rC3 * cfac
            y[i] = -mr * srsp - dlnp * inv_dlnr

            # Advance the sliding window:
            #
            # Old: (rL, rC, rR) = (r[i],   r[i+1], r[i+2])
            # New: (rL, rC, rR) = (r[i+1], r[i+2], r[i+3])
            #
            # The same rolling update is used for log(r) and log(p),
            # so each new iteration computes only one new log(r) and
            # one new log(p).
            if i + 1 < n_out:
                rL = rC
                rC = rR
                rR = r[i + 3]

                rL3 = rC3
                rC2 = rC * rC
                rC3 = rC2 * rC
                rR3 = rR * rR * rR

                log_rL = log_rC
                log_rC = log_rR
                log_rR = math.log(rR)

                log_pL = log_pR
                log_pR = math.log(p[i + 2])

    # ------------------------------------------------------------------
    # Enforce dp/dr = 0 for i = 1 exactly as in the original code.
    #
    # Since row 0 is always replaced by this boundary condition, we only
    # assemble it here once and never build the discarded general row.
    # ------------------------------------------------------------------
    rL = r[0]
    rC = r[1]
    rR = r[2]

    rC2 = rC * rC
    rC3 = rC2 * rC
    rR3 = rR * rR * rR
    rL3 = rL * rL * rL

    denomL = rC3 - rL3
    denomR = rR3 - rC3
    inv_denomL = 1.0 / denomL
    inv_denomR = 1.0 / denomR

    rR3rC3 = rR3 * inv_denomR
    rC3rC3 = rC3 * inv_denomR
    rC3rL3 = rC3 * inv_denomL

    a[0] = 0.0
    b[0] = 5.0 * (rC3rC3 + rC3rL3)
    c[0] = -5.0 * rR3rC3
    y[0] = -(math.log(p[1]) - math.log(p[0]))

### Old routine to compute the pressures for HE.  assumed aligned grids

@njit(types.Tuple((float64[:, :], float64, float64))(float64[:, :], float64[:, :], float64[:, :], float64[:,:], float64[:]),
      fastmath=True, cache=True)
def compute_he_pressures(r, rho, p, m, bkg_param):
    """
    Compute pressure array such that the density profile will be in HE
    This legacy routine predates the current initialization path.
    Here we assume that radial grid points are aligned.

    Arguments
    ---------
    r : ndarray, shape (s, N+1)
        Edge radii per species.
    rho : ndarray, shape (s, N)
        Shell densities per species.
    p : ndarray, shape (s, N)
        Shell pressures per species.
    m : ndarray, shape (s, N+1)
        Each species' enclosed mass at its edges.
    bkg_param : ndarray, shape (4,)
        Parameters for background potential

    Returns
    -------
    p_new : ndarray, shape (s, N)
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
    lnL : ndarray, shape (s, s)
        Coulomb logarithm coefficients.
    
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
    - The function returns the updated pressure and v2 arrays, the observed maximum relative
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

### OLD FUNCTION to compute rc_frac

@njit((float64[:, :], float64[:, :], float64, float64[:]), fastmath=True, cache=True)
def compute_rc_frac(r, m, r_c, rc_frac):
    """
    Compute f_k = M_k(<r_c)/Mtot(<r_c) for each species, in place

    Arguments
    ---------
    r : array-like, shape (s, N+1)
        Radii arrays per species
    m : array-like, shape (s, N+1)
        Mass arrays per species
    r_c : float
        Core radius
    rc_frac : array-like, shape (s,)
        Preallocated species fractions, overwritten in place.
    """
    s, _ = r.shape

    denom = sum_extensive_loglog_single(r_c, r, m)
    for k in range(s):
        rc_frac[k] = interp_species_loglog_single(r_c, r, m, k) / denom
        
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