import numpy as np
import math
from numba import njit, float64, types, void, int64
from pygtf2.util.interpolate import interp_m_enc, interp_m_enc_and_K
from pygtf2.util.calc import add_bkg_pot, solve_tridiagonal_thomas

STATUS_OK = 0
STATUS_SHELL_CROSSING = 1
_TINY64 = np.finfo(np.float64).tiny

@njit(float64[:](float64[:]), cache=True, fastmath=True)
def compute_mass(m) -> np.ndarray:
    """
    Placeholder function to compute mass used in build_tridiag_system.
    Accounts for baryons, perturbers, etc. in future implementations.

    Arguments
    ---------
    m : ndarray
        Enclosed fluid mass at each radial grid point.

    Returns
    -------
    ndarray
        Total mass for hydrostatis equilibrium calculations.
    """

    return m

@njit(void(float64[:], float64[:], float64[:], float64[:], float64[:]), cache=True, fastmath=True)
def update_r_p_rho(r, x, p, rho, work):
    """
    Updates r, and then finds p, rho, and v2 based on exact volume ratios.
    Ensures positivity and stability.
    All updates are performed in place.

    Parameters
    ----------
    r : ndarray, shape (N+1,)
        Edge radii. Updated in place.
    x : ndarray, shape (N-1,)
        Interior fractional stretches. x_j = dr_j / r_j for j=1..N-1, shape (N-1,)
    p : ndarray, shape (N,)
        Shell-centered pressures. Updated in place.
    rho : ndarray, shape (N,)
        Shell-centered densities. Updated in place.
    work : ndarray, shape (N,)
        Scratch array used to store old shell volumes.
    """
    tiny = _TINY64
    gamma = 5.0 / 3.0
    n = p.shape[0]

    # Store old shell volumes
    for j in range(n):
        rL = r[j]
        rR = r[j + 1]
        work[j] = rR * rR * rR - rL * rL * rL

    # Update interior radii in place
    for j in range(n - 1):
        r[j + 1] *= (1.0 + x[j])

    # Update rho and p from volume ratios
    for j in range(n):
        rL = r[j]
        rR = r[j + 1]
        V_new = rR * rR * rR - rL * rL * rL

        if V_new < tiny:
            V_new = tiny

        ratio = work[j] / V_new
        rho[j] *= ratio
        p[j] *= ratio ** gamma

@njit(void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:]), cache=True, fastmath=True)
def build_tridiag_system_mother(r, rho, p, m_tot, K_k, a, b, c, y):
    """
    Fill preallocated arrays a, b, c, y with the tridiagonal system
    for the interior fractional radius shifts.

    This version includes the first-order variation of the interpolated
    other-species enclosed mass with respect to motion of the current
    species grid.

    Parameters
    ----------
    r : ndarray, shape (N+1,)
        Edge radii.
    rho : ndarray, shape (N,)
        Shell-centered densities.
    p : ndarray, shape (N,)
        Shell-centered pressures.
    m_tot : ndarray, shape (N+1,)
        Total enclosed mass at the same edge radii as `r`.
    K_k : ndarray, shape (N+1,)
        dM_other/dr evaluated on species-k's edge grid.
        Only interior values K_k[1:-1] are used.
    a, b, c, y : ndarray, shape (N-1,)
        Preallocated output arrays to fill in place.
    """
    tiny = _TINY64

    n = r.shape[0] - 2 # number of interior unknowns

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
            dP = tiny if dP >= 0.0 else -tiny

        if drho < tiny:
            drho = tiny

        Mi = m_tot[iC]
        if Mi < tiny:
            Mi = tiny

        rL3 = rL * rL * rL
        rC3 = rC * rC * rC
        rR3 = rR * rR * rR

        r3a = rR3 / (rR3 - rC3)
        r3c = rC3 / (rC3 - rL3)
        r3b = r3a - 1.0
        r3d = r3c - 1.0

        q1 = rR * inv_dr
        q2 = q1 - 1.0

        dd = -(4.0 / Mi) * ((rC * rC) * inv_dr) * (dP / drho)

        c1 = 5.0 * dd * (p[j + 1] / dP) - 3.0 * (rho[j + 1] / drho)
        c2 = 5.0 * dd * (p[j] / dP)     + 3.0 * (rho[j] / drho)

        # Extra term to account for shifting M_other grid
        lam = rC * K_k[iC] / Mi

        y[j] = dd - 1.0
        a[j] = r3d * c2 - q2
        b[j] = -2.0 - r3b * c1 - r3c * c2 + lam
        c[j] = r3a * c1 + q1

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
      (excluding the fixed inner and outer edges), so the returned arrays all have length M-2.
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

@njit(float64(float64[:, :], float64[:, :], float64[:, :], float64[:,:], float64[:]), fastmath=True,cache=True)
def compute_he_resid_norm(r, rho, p, m, bkg_param):
    """
    Compute an (unscaled) norm of the HE residual
    """
    s, Np1 = r.shape
    res_vec = np.empty((s, Np1-2), dtype=np.float64)
    m_totk = np.empty(Np1, dtype=np.float64) # Pre-allocate for interp_m_enc
    add_bkg_flag = bkg_param[0] != -1
    
    dp = p[:, 1:] - p[:, :-1]
    srho = rho[:, 1:] + rho[:, :-1]
    dr = r[:, 2:] - r[:, :-2]
    rC = r[:, 1:-1]

    for k in range(s):
        interp_m_enc(k, r, m, m_totk)
        if add_bkg_flag:
            m_totk += add_bkg_pot(r[k], bkg_param)
        res_vec[k] = - (4.0 / m_totk[1:-1]) * (rC[k]**2 / dr[k]) * (dp[k] / srho[k]) - 1.0

    return np.linalg.norm(res_vec)

@njit(void(float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64[:]), fastmath=True, cache=True)
def compute_he_pressures(r, rho, p, m, bkg_param):
    """
    In-place hydrostatic-equilibrium pressure update for unaligned radial grids.

    This version does NOT compute residuals.

    Parameters
    ----------
    r : ndarray, shape (s, N+1)
        Edge radii per species.
    rho : ndarray, shape (s, N)
        Shell densities per species.
    p : ndarray, shape (s, N)
        Shell pressures per species. Updated in place.
    m : ndarray, shape (s, N+1)
        Enclosed-mass-like data used by interp_m_enc().
    bkg_param : ndarray, shape (4,)
        Parameters for background potential.
    """
    s, Np1 = r.shape
    N = Np1 - 1
    use_bkg = bkg_param[0] != -1.0
    quarter = 0.25

    # Scratch buffer for total enclosed mass on species-k grid
    m_totk = np.empty(Np1, dtype=np.float64)

    for k in range(s):
        # Fill m_totk[:] with total enclosed mass evaluated on species-k grid
        interp_m_enc(k, r, m, m_totk)

        if use_bkg:
            # Assumes add_bkg_pot accepts a 1D radius array and returns a 1D mass array
            m_totk += add_bkg_pot(r[k], bkg_param)

        rk = r[k]
        rhok = rho[k]
        pk = p[k]

        # Backward sweep:
        # pk[N-1] is treated as the outer boundary value and left unchanged
        for i in range(N - 2, -1, -1):
            rip1 = rk[i + 1]
            pk[i] = pk[i + 1] + (
                (rhok[i + 1] + rhok[i]) *
                (rk[i + 2] - rk[i]) *
                m_totk[i + 1] *
                (quarter / (rip1 * rip1))
            )

@njit(types.Tuple((float64, float64))(float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64[:]), fastmath=True, cache=True)
def compute_he_pressures_with_resid(r, rho, p, m, bkg_param):
    """
    In-place hydrostatic-equilibrium pressure update for unaligned radial grids.

    This version computes and returns the old and new HE residual norms.

    Parameters
    ----------
    r : ndarray, shape (s, N+1)
        Edge radii per species.
    rho : ndarray, shape (s, N)
        Shell densities per species.
    p : ndarray, shape (s, N)
        Shell pressures per species. Updated in place.
    m : ndarray, shape (s, N+1)
        Enclosed-mass-like data used by interp_m_enc().
    bkg_param : ndarray, shape (4,)
        Parameters for background potential.

    Returns
    -------
    res_old : float
        HE residual of input arrays.
    res_new : float
        HE residual after in-place pressure update.
    """
    res_old = compute_he_resid_norm(r, rho, p, m, bkg_param)
    compute_he_pressures(r, rho, p, m, bkg_param)
    res_new = compute_he_resid_norm(r, rho, p, m, bkg_param)
    return res_old, res_new

@njit(int64(float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64[:]), cache=True, fastmath=True)
def revirialize_interp_gs(r, rho, p, m, bkg_param) -> int:
    """
    Multi-species re-virialization.  Updates r, rho, and p in place.  No diagnostics.

    Solves for radius adjustments and updates physical quantities for all species.
    Species generally do not have aligned radial bins, so enclosed mass from the
    other species is interpolated onto the current species grid.

    Updates to per-species r arrays are fed back into next species revir - this is a Gauss-Seidel-like scheme.

    Parameters
    ----------
    r : ndarray, shape (s, N+1)
        Edge radii per species. Updated in place.
    rho : ndarray, shape (s, N)
        Shell densities per species. Updated in place.
    p : ndarray, shape (s, N)
        Shell pressures per species. Updated in place.
    m : ndarray, shape (s, N+1)
        Total enclosed mass at edges, per species. Not updated.
    bkg_param : ndarray, shape (4,)
        Parameters for background potential.

    Returns
    -------
    status : int
        STATUS_OK if successful,
        STATUS_SHELL_CROSSING if any radii cross.

    Notes
    -----
    This function solves a tridiagonal system to compute radius corrections for each species,
    then updates density and pressure accordingly. If any radii cross, the function returns
    'shell_crossing'. Since updates are in place, the arrays may already be partially or fully 
    modified when that happens.
    """
    s, Np1 = r.shape
    add_bkg_flag = bkg_param[0] != -1

    # Pre-allocate tridiagonal coefficient and work arrays
    n_int   = Np1 - 2
    a       = np.empty(n_int, dtype=np.float64)
    b       = np.empty(n_int, dtype=np.float64)
    c       = np.empty(n_int, dtype=np.float64)
    y       = np.empty(n_int, dtype=np.float64)
    xk      = np.empty(n_int, dtype=np.float64)
    vol_old = np.empty(Np1 - 1, dtype=np.float64)
    m_totk  = np.empty(Np1, dtype=np.float64)
    K_k     = np.empty(Np1, dtype=np.float64)

    for k in range(s):
        # interp_m_enc(k, r, m, m_totk)
        interp_m_enc_and_K(k, r, m, m_totk, K_k)

        if add_bkg_flag: # For the future: make this in-place too
            m_totk += add_bkg_pot(r[k], bkg_param)
        
        # build_tridiag_system(r[k], rho[k], p[k], m_totk, a, b, c, y)
        build_tridiag_system_mother(r[k], rho[k], p[k], m_totk, K_k, a, b, c, y)
        solve_tridiagonal_thomas(a, b, c, y, xk)
        
        update_r_p_rho(r[k], xk, p[k], rho[k], vol_old)

    for k in range(s):
        for i in range(Np1 - 1):
            if r[k, i + 1] - r[k, i] <= 0.0:
                return STATUS_SHELL_CROSSING

    return STATUS_OK

@njit(int64(float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64[:]), cache=True, fastmath=True)
def revirialize_interp_jacobi(r, rho, p, m, bkg_param) -> int:
    """
    Multi-species re-virialization, Jacobi-style in the inter-species coupling.

    Updates r, rho, and p in place, but uses a frozen copy of the original r
    array for every species' re-virialization during this sweep.  This prevents
    earlier species' radius updates from feeding back into later species within
    the same call.

    Parameters
    ----------
    r : ndarray, shape (s, N+1)
        Edge radii per species. Updated in place.
    rho : ndarray, shape (s, N)
        Shell densities per species. Updated in place.
    p : ndarray, shape (s, N)
        Shell pressures per species. Updated in place.
    m : ndarray, shape (s, N+1)
        Total enclosed mass at edges, per species. Not updated.
    bkg_param : ndarray, shape (4,)
        Parameters for background potential.

    Returns
    -------
    status : int
        STATUS_OK if successful,
        STATUS_SHELL_CROSSING if any radii cross.
    """
    s, Np1 = r.shape
    add_bkg_flag = bkg_param[0] != -1

    n_int   = Np1 - 2
    a       = np.empty(n_int, dtype=np.float64)
    b       = np.empty(n_int, dtype=np.float64)
    c       = np.empty(n_int, dtype=np.float64)
    y       = np.empty(n_int, dtype=np.float64)
    xk      = np.empty(n_int, dtype=np.float64)
    vol_old = np.empty(Np1 - 1, dtype=np.float64)
    K_all   = np.empty((s, Np1), dtype=np.float64)

    # One precomputed enclosed-mass profile per species
    m_tot_all = np.empty((s, Np1), dtype=np.float64)

    # Pass 1: compute all interpolated enclosed-mass profiles before any updates
    for k in range(s):
        # interp_m_enc(k, r, m, m_tot_all[k])
        interp_m_enc_and_K(k, r, m, m_tot_all[k], K_all[k])
        if add_bkg_flag:
            m_tot_all[k] += add_bkg_pot(r[k], bkg_param)

    # Pass 2: update each species using the frozen profiles
    for k in range(s):
        # build_tridiag_system(r[k], rho[k], p[k], m_tot_all[k], a, b, c, y)
        build_tridiag_system_mother(r[k], rho[k], p[k], m_tot_all[k], K_all[k], a, b, c, y)
        solve_tridiagonal_thomas(a, b, c, y, xk)
        update_r_p_rho(r[k], xk, p[k], rho[k], vol_old)

    for k in range(s):
        for i in range(Np1 - 1):
            if r[k, i + 1] - r[k, i] <= 0.0:
                return STATUS_SHELL_CROSSING

    return STATUS_OK

@njit(types.Tuple((int64, float64, float64))(float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64[:]), cache=True, fastmath=True,)
def revirialize_interp_gs_diagnostics(r, rho, p, m, bkg_param) -> tuple[int, float, float]:
    """
    Multi-species re-virialization.  Updates r, rho, and p in place.  With diagnostics.
    To be used during state initialization.

    Solves for radius adjustments and updates physical quantities for all species.
    Species generally do not have aligned radial bins, so enclosed mass from the
    other species is interpolated onto the current species grid.

    Updates to per-species r arrays are fed back into next species revir - this is found to be necessary.

    Parameters
    ----------
    r : ndarray, shape (s, N+1)
        Edge radii per species. Updated in place.
    rho : ndarray, shape (s, N)
        Shell densities per species. Updated in place.
    p : ndarray, shape (s, N)
        Shell pressures per species. Updated in place.
    m : ndarray, shape (s, N+1)
        Total enclosed mass at edges, per species. Not updated.
    bkg_param : ndarray, shape (4,)
        Parameters for background potential.

    Returns
    -------
    status : int
        STATUS_OK if successful,
        STATUS_SHELL_CROSSING if any radii cross.
    dr_max : float
        Global maximum |dr/r| across all species.
    he_res : float
        Norm of HE residual for updated profile.
        If shell crossing occurs, returns -1.0 as a sentinel.

    Notes
    -----
    This function solves a tridiagonal system to compute radius corrections for each species,
    then updates density and pressure accordingly. If any radii cross, the function returns
    'shell_crossing'. Since updates are in place, the arrays may already be partially or fully 
    modified when that happens.
    """
    s, Np1 = r.shape
    dr_max = 0.0
    add_bkg_flag = bkg_param[0] != -1

    # Pre-allocate tridiagonal coefficient and work arrays
    n_int   = Np1 - 2
    a       = np.empty(n_int, dtype=np.float64)
    b       = np.empty(n_int, dtype=np.float64)
    c       = np.empty(n_int, dtype=np.float64)
    y       = np.empty(n_int, dtype=np.float64)
    xk      = np.empty(n_int, dtype=np.float64)
    vol_old = np.empty(Np1 - 1, dtype=np.float64)
    m_totk  = np.empty(Np1, dtype=np.float64)
    K_k     = np.empty(Np1, dtype=np.float64)

    for k in range(s):
        # interp_m_enc(k, r, m, m_totk)
        interp_m_enc_and_K(k, r, m, m_totk, K_k)

        if add_bkg_flag: # For the future: make this in-place too
            m_totk += add_bkg_pot(r[k], bkg_param)
        
        # build_tridiag_system(r[k], rho[k], p[k], m_totk, a, b, c, y)
        build_tridiag_system_mother(r[k], rho[k], p[k], m_totk, K_k, a, b, c, y)
        solve_tridiagonal_thomas(a, b, c, y, xk)
        
        local_max = np.max(np.abs(xk))
        if local_max > dr_max:
            dr_max = local_max
        
        update_r_p_rho(r[k], xk, p[k], rho[k], vol_old)

    for k in range(s):
        for i in range(Np1 - 1):
            if r[k, i + 1] - r[k, i] <= 0.0:
                return STATUS_SHELL_CROSSING, dr_max, -1.0

    he_res = compute_he_resid_norm(r, rho, p, m, bkg_param)

    return STATUS_OK, dr_max, he_res

@njit(types.Tuple((int64, float64, float64))(float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64[:]), cache=True, fastmath=True)
def revirialize_interp_jacobi_diagnostics(r, rho, p, m, bkg_param) -> int:
    """
    Multi-species re-virialization.  Jacobi-style in the inter-species coupling..  With diagnostics.
    To be used during state initialization.

    Solves for radius adjustments and updates physical quantities for all species.
    Species generally do not have aligned radial bins, so enclosed mass from the
    other species is interpolated onto the current species grid.

    Parameters
    ----------
    r : ndarray, shape (s, N+1)
        Edge radii per species. Updated in place.
    rho : ndarray, shape (s, N)
        Shell densities per species. Updated in place.
    p : ndarray, shape (s, N)
        Shell pressures per species. Updated in place.
    m : ndarray, shape (s, N+1)
        Total enclosed mass at edges, per species. Not updated.
    bkg_param : ndarray, shape (4,)
        Parameters for background potential.

    Returns
    -------
    status : int
        STATUS_OK if successful,
        STATUS_SHELL_CROSSING if any radii cross.
    dr_max : float
        Global maximum |dr/r| across all species.
    he_res : float
        Norm of HE residual for updated profile.
        If shell crossing occurs, returns -1.0 as a sentinel.

    Notes
    -----
    This function solves a tridiagonal system to compute radius corrections for each species,
    then updates density and pressure accordingly. If any radii cross, the function returns
    'shell_crossing'. Since updates are in place, the arrays may already be partially or fully 
    modified when that happens.
    """
    s, Np1 = r.shape
    dr_max = 0.0
    add_bkg_flag = bkg_param[0] != -1

    n_int   = Np1 - 2
    a       = np.empty(n_int, dtype=np.float64)
    b       = np.empty(n_int, dtype=np.float64)
    c       = np.empty(n_int, dtype=np.float64)
    y       = np.empty(n_int, dtype=np.float64)
    xk      = np.empty(n_int, dtype=np.float64)
    vol_old = np.empty(Np1 - 1, dtype=np.float64)
    K_all   = np.empty((s, Np1), dtype=np.float64)

    # One precomputed enclosed-mass profile per species
    m_tot_all = np.empty((s, Np1), dtype=np.float64)

    # Pass 1: compute all interpolated enclosed-mass profiles before any updates
    for k in range(s):
        # interp_m_enc(k, r, m, m_tot_all[k])
        interp_m_enc_and_K(k, r, m, m_tot_all[k], K_all[k])
        if add_bkg_flag:
            m_tot_all[k] += add_bkg_pot(r[k], bkg_param)

    # Pass 2: update each species using the frozen profiles
    for k in range(s):
        # build_tridiag_system(r[k], rho[k], p[k], m_tot_all[k], a, b, c, y)
        build_tridiag_system_mother(r[k], rho[k], p[k], m_tot_all[k], K_all[k], a, b, c, y)
        solve_tridiagonal_thomas(a, b, c, y, xk)

        local_max = np.max(np.abs(xk))
        if local_max > dr_max:
            dr_max = local_max

        update_r_p_rho(r[k], xk, p[k], rho[k], vol_old)

    for k in range(s):
        for i in range(Np1 - 1):
            if r[k, i + 1] - r[k, i] <= 0.0:
                return STATUS_SHELL_CROSSING, dr_max, -1.0
            
    he_res = compute_he_resid_norm(r, rho, p, m, bkg_param)

    return STATUS_OK, dr_max, he_res