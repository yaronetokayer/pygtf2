import numpy as np
from numba import njit, float64, types, void, int64
from multigtf.util.interpolate import interp_m_enc, interp_m_enc_and_K
from multigtf.util.calc import add_bkg_pot, solve_tridiagonal_thomas, add_bkg_K

STATUS_OK = 0
STATUS_SHELL_CROSSING = 1
_TINY64 = np.finfo(np.float64).tiny

@njit(void(float64[:], float64[:], float64[:], float64[:], float64[:]), cache=True, fastmath=True)
def update_r_p_rho(r, x, p, rho, work):
    """
    Update interior radii, then density and pressure from shell-volume ratios.
    Pressure follows an adiabatic exponent of 5/3; v2 is not an argument.
    New volumes are floored at the smallest positive float for division.
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
        dM_other/dr evaluated on species-k's edge grid, where
        "other" includes both the other species and any
        static background mass distribution.
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
            add_bkg_pot(r[k], bkg_param, m_totk)
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
        Each species' own enclosed mass at its radial edges.
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
            add_bkg_pot(r[k], bkg_param, m_totk)

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
        Each species' own enclosed mass at its radial edges.
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

@njit(
    int64(
        float64[:, :], # r
        float64[:, :], # rho
        float64[:, :], # p
        float64[:, :], # m
        float64[:],    # bkg_param
        float64[:],    # a 
        float64[:],    # b
        float64[:],    # c 
        float64[:],    # y
        float64[:],    # xk
        float64[:],    # vol_old
        float64[:, :], # K_all
        float64[:, :], # m_tot_all
    ), 
    cache=True, fastmath=True,
)
def revirialize_interp_jacobi(
    r, rho, p, m, bkg_param,
    a, b, c, y, xk, vol_old, K_all, m_tot_all,
    ) -> int:
    """
    Multi-species re-virialization, Jacobi-style in the inter-species coupling.

    Updates r, rho, and p in place after computing every species' total
    enclosed mass and other-mass derivative on the original grids. These
    frozen coupling arrays prevent earlier radius updates from affecting
    later species within the same sweep.

    Parameters
    ----------
    r : ndarray, shape (s, N+1)
        Edge radii per species. Updated in place.
    rho : ndarray, shape (s, N)
        Shell densities per species. Updated in place.
    p : ndarray, shape (s, N)
        Shell pressures per species. Updated in place.
    m : ndarray, shape (s, N+1)
        Each species' own enclosed mass at its edges. Not updated.
    bkg_param : ndarray, shape (4,)
        Parameters for background potential.
    a, b, c, y, xk : ndarray, shape (N-1,)
        Scratch arrays for the tridiagonal solve; overwritten in place.
    vol_old : ndarray, shape (N,)
        Scratch array for old shell volumes.
    K_all, m_tot_all : ndarray, shape (s, N+1)
        Scratch arrays for other-mass derivatives and total enclosed masses.

    Returns
    -------
    status : int
        STATUS_OK if successful,
        STATUS_SHELL_CROSSING if any radii cross.
    """
    s, Np1 = r.shape
    add_bkg_flag = bkg_param[0] != -1

    # Pass 1: compute all interpolated enclosed-mass profiles before any updates
    for k in range(s):
        # interp_m_enc(k, r, m, m_tot_all[k])
        interp_m_enc_and_K(k, r, m, m_tot_all[k], K_all[k])
        if add_bkg_flag:
            add_bkg_pot(r[k], bkg_param, m_tot_all[k])
            add_bkg_K(r[k], bkg_param, K_all[k])

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

@njit(
    types.Tuple((int64, float64, float64))(
        float64[:, :], 
        float64[:, :], 
        float64[:, :], 
        float64[:, :], 
        float64[:],
        float64[:],    # a 
        float64[:],    # b
        float64[:],    # c 
        float64[:],    # y
        float64[:],    # xk
        float64[:],    # vol_old
        float64[:, :], # K_all
        float64[:, :], # m_tot_all
    ), 
    cache=True, fastmath=True
)
def revirialize_interp_jacobi_diagnostics(
    r, rho, p, m, bkg_param,
    a, b, c, y, xk, vol_old, K_all, m_tot_all,
    ) -> tuple:
    """
    Apply Jacobi-style multi-species revirialization and return diagnostics.
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
        Each species' own enclosed mass at its edges. Not updated.
    bkg_param : ndarray, shape (4,)
        Parameters for background potential.
    a, b, c, y, xk : ndarray, shape (N-1,)
        Scratch arrays for the tridiagonal solve; overwritten in place.
    vol_old : ndarray, shape (N,)
        Scratch array for old shell volumes.
    K_all, m_tot_all : ndarray, shape (s, N+1)
        Scratch arrays for other-mass derivatives and total enclosed masses.

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
    the integer STATUS_SHELL_CROSSING. Since updates are in place, the arrays may already be partially or fully
    modified when that happens.
    """
    s, Np1 = r.shape
    dr_max = 0.0
    add_bkg_flag = bkg_param[0] != -1

    # Pass 1: compute all interpolated enclosed-mass profiles before any updates
    for k in range(s):
        # interp_m_enc(k, r, m, m_tot_all[k])
        interp_m_enc_and_K(k, r, m, m_tot_all[k], K_all[k])
        if add_bkg_flag:
            add_bkg_pot(r[k], bkg_param, m_tot_all[k])
            add_bkg_K(r[k], bkg_param, K_all[k])

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
