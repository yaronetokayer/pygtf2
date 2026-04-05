import numpy as np
from numba import njit, float64, types, void, int64
from pygtf2.util.interpolate import interp_m_enc
from pygtf2.util.calc import add_bkg_pot, solve_tridiagonal_thomas

STATUS_OK = 0
STATUS_SHELL_CROSSING = 1

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

@njit(
    void(
        float64[:],  # r
        float64[:],  # x
        float64[:],  # p
        float64[:],  # rho
        float64[:]   # work
    ), cache=True, fastmath=True,
)
def _update_r_p_rho(r, x, p, rho, work):
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
    tiny = 2.2250738585072014e-308  # np.finfo(np.float64).tiny
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

@njit(
    void(
        float64[:],  # r
        float64[:],  # rho
        float64[:],  # p
        float64[:],  # m_tot
        float64[:],  # a
        float64[:],  # b
        float64[:],  # c
        float64[:]   # y
    ), cache=True, fastmath=True,
)
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
    tiny = 2.2250738585072014e-308  # np.finfo(np.float64).tiny

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

@njit(types.Tuple((float64[:], float64[:], float64[:], float64[:]))
      (float64[:], float64[:], float64[:], float64[:]), cache=True, fastmath=False)
def build_tridiag_system_log(r, rho, p, m_tot) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct the tridiagonal system A·x = y for interior radial corrections.
    These are the coefficients for the log form of the HE equation.

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

    Returns
    -------
    a, b, c, y : ndarray
        Tuple of 1D arrays (each length = n-1) defining the tridiagonal system
        for the interior unknowns:
          - a: subdiagonal (A[i, i-1])
          - b: main diagonal (A[i, i])
          - c: superdiagonal (A[i, i+1])
          - y: right-hand side vector
    """
    # Geometric volume factors
    rL = r[:-2]         # Left radial grid points
    rR = r[2:]          # Right radial grid points
    rC = r[1:-1]        # Central radial grid points
    
    rC2 = rC**2
    rC3 = rC2 * rC
    rR3 = rR**3
    rL3 = rL**3

    rL3rL3 = rL3 / (rC3 - rL3)
    rR3rC3 = rR3 / (rR3 - rC3)
    rC3rC3 = rC3 / (rR3 - rC3)
    rC3rL3 = rC3 / (rC3 - rL3)
    rC2rC3 = rC2 / (rR3 - rC3)
    rC2rL3 = rC2 / (rC3 - rL3)
    
    lnr = np.empty_like(r)
    lnr[1:] = np.log(r[1:])             # Don't take ln0 - lnr[0] never used anyway
    lnr[0]  = lnr[1]                    # Arbitrary finite placeholder
    dlnr = 0.5 * ( lnr[2:] - lnr[:-2] ) # Central difference

    pL = p[:-1]
    pR = p[1:]
    rhoL = rho[:-1]
    rhoR = rho[1:]
    lnp = np.log(p)
    dlnp = lnp[1:] - lnp[:-1]           # Right-sided difference

    sr = rho[:-1] + rho[1:]
    sp = p[:-1] + p[1:]

    mr = m_tot[1:-1] / r[1:-1]

    # floors to avoid divide-by-zero/inf
    tiny = np.finfo(np.float64).tiny
    sp   = np.where(np.abs(sp)   < tiny, np.copysign(tiny, sp),   sp)
    dlnr   = np.where(np.abs(dlnr)   < tiny, np.copysign(tiny, dlnr),   dlnr)

    dpdr = 0.5 * dlnp / dlnr**2
    srsp = sr / sp

    # Terms in final expressions
    afac = 5.0 / dlnr + (mr / sp) * (5.0 * pL * srsp - 3.0 * rhoL)
    bfac1 = 5.0 / dlnr
    bfac2 = m_tot[1:-1] / sp
    bfac3 = 5.0 * pR  * srsp - 3.0 * rhoR
    bfac4 = 3.0 * rhoL - 5.0 * pL * srsp
    cfac = 5.0 / dlnr + (mr / sp) * (3.0 * rhoR - 5.0 * pR * srsp)
    dfac = 2.0 * dpdr * dlnr

    y = -mr * srsp - dfac

    a = dpdr - rL3rL3 * afac                                                                # Subdiagonal
    b = bfac1 * (rC3rC3 + rC3rL3) - mr * srsp - bfac2 * (rC2rC3 * bfac3 + rC2rL3 * bfac4)   # Main diagonal
    c = -dpdr - rR3rC3 * cfac                                                               # Superdiagonal

    # Enforce dp/dr = 0 for i=1
    a[0] = 0.0
    b[0] = 5.0 * ( rC3rC3[0] + rC3rL3[0] )
    c[0] = -5.0 * rR3rC3[0]
    y[0] = -dlnp[0]

    c[-1] = 0.0 # Outside of matrix

    return a, b, c, y

@njit(float64(float64[:, :], float64[:, :], float64[:, :], float64[:,:], float64[:]),
    fastmath=True,cache=True)
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

@njit(int64(float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64[:]),
      cache=True, fastmath=True)
def revirialize_interp_hybrid(r, rho, p, m, bkg_param) -> int:
    """
    Multi-species re-virialization, Jacobi-style in the inter-species coupling.

    In-place version that is functionally equivalent to the older out-of-place
    implementation: later species see earlier species' updated radii through an
    auxiliary radius copy used by interp_m_enc, while each species' own solve
    is still built from the original in-place state at the moment that species
    is processed.

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

    # This plays the role of the old r_copy:
    # used only for inter-species mass interpolation, and updated after each species.
    r_interp = r.copy()

    # Pre-allocate tridiagonal coefficient and work arrays
    n_int = Np1 - 2
    a  = np.empty(n_int, dtype=np.float64)
    b  = np.empty(n_int, dtype=np.float64)
    c  = np.empty(n_int, dtype=np.float64)
    y  = np.empty(n_int, dtype=np.float64)
    xk = np.empty(n_int, dtype=np.float64)
    vol_old = np.empty(Np1 - 1, dtype=np.float64)
    m_totk = np.empty(Np1, dtype=np.float64)

    for k in range(s):
        # Use the frozen radii snapshot for all species interpolation
        interp_m_enc(k, r_interp, m, m_totk)

        if add_bkg_flag:
            m_totk += add_bkg_pot(r[k], bkg_param)

        # Build the system from the frozen radii for species k
        build_tridiag_system(r[k], rho[k], p[k], m_totk, a, b, c, y)
        solve_tridiagonal_thomas(a, b, c, y, xk)

        # Make sure the update is applied relative to the original radii
        _update_r_p_rho(r[k], xk, p[k], rho[k], vol_old)
        r_interp[k, :] = r[k, :]

    for k in range(s):
        for i in range(Np1 - 1):
            if r[k, i + 1] - r[k, i] <= 0.0:
                return STATUS_SHELL_CROSSING

    return STATUS_OK

@njit(int64(float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64[:]),
      cache=True, fastmath=True)
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

    # Freeze the original radii for this entire sweep (Jacobi outer coupling)
    r_old = r.copy()

    # Pre-allocate tridiagonal coefficient and work arrays
    n_int = Np1 - 2
    a  = np.empty(n_int, dtype=np.float64)
    b  = np.empty(n_int, dtype=np.float64)
    c  = np.empty(n_int, dtype=np.float64)
    y  = np.empty(n_int, dtype=np.float64)
    xk = np.empty(n_int, dtype=np.float64)
    vol_old = np.empty(Np1 - 1, dtype=np.float64)
    m_totk = np.empty(Np1, dtype=np.float64)

    for k in range(s):
        # Use the frozen radii snapshot for all species interpolation
        interp_m_enc(k, r_old, m, m_totk)

        if add_bkg_flag:
            m_totk += add_bkg_pot(r_old[k], bkg_param)

        # Build the system from the frozen radii for species k
        build_tridiag_system(r_old[k], rho[k], p[k], m_totk, a, b, c, y)
        solve_tridiagonal_thomas(a, b, c, y, xk)

        # Make sure the update is applied relative to the original radii
        r[k, :] = r_old[k, :]
        _update_r_p_rho(r[k], xk, p[k], rho[k], vol_old)

    for k in range(s):
        for i in range(Np1 - 1):
            if r[k, i + 1] - r[k, i] <= 0.0:
                return STATUS_SHELL_CROSSING

    return STATUS_OK

@njit(int64(float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64[:]),
      cache=True, fastmath=True)
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
    n_int = Np1 - 2
    a  = np.empty(n_int, dtype=np.float64)
    b  = np.empty(n_int, dtype=np.float64)
    c  = np.empty(n_int, dtype=np.float64)
    y  = np.empty(n_int, dtype=np.float64)
    xk = np.empty(n_int, dtype=np.float64)
    vol_old = np.empty(Np1 - 1, dtype=np.float64)
    m_totk = np.empty(Np1, dtype=np.float64)

    for k in range(s):
        interp_m_enc(k, r, m, m_totk)

        if add_bkg_flag: # For the future: make this in-place too
            m_totk += add_bkg_pot(r[k], bkg_param)
        
        build_tridiag_system(r[k], rho[k], p[k], m_totk, a, b, c, y)
        solve_tridiagonal_thomas(a, b, c, y, xk)
        
        _update_r_p_rho(r[k], xk, p[k], rho[k], vol_old)

    for k in range(s):
        for i in range(Np1 - 1):
            if r[k, i + 1] - r[k, i] <= 0.0:
                return STATUS_SHELL_CROSSING

    return STATUS_OK

@njit(
    types.Tuple((int64, float64, float64))(
        float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64[:]
    ), cache=True, fastmath=True,
)
def revirialize_interp_jacobi_diagnostics(r, rho, p, m, bkg_param) -> tuple[int, float, float]:
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

    # Freeze the original radii for this entire sweep (Jacobi outer coupling)
    r_old = r.copy()

    # Pre-allocate tridiagonal coefficient and work arrays
    n_int = Np1 - 2
    a  = np.empty(n_int, dtype=np.float64)
    b  = np.empty(n_int, dtype=np.float64)
    c  = np.empty(n_int, dtype=np.float64)
    y  = np.empty(n_int, dtype=np.float64)
    xk = np.empty(n_int, dtype=np.float64)
    vol_old = np.empty(Np1 - 1, dtype=np.float64)
    m_totk = np.empty(Np1, dtype=np.float64)

    for k in range(s):
        interp_m_enc(k, r_old, m, m_totk)

        if add_bkg_flag: # For the future: make this in-place too
            m_totk += add_bkg_pot(r_old[k], bkg_param)
        
        build_tridiag_system(r_old[k], rho[k], p[k], m_totk, a, b, c, y)
        solve_tridiagonal_thomas(a, b, c, y, xk)
        
        local_max = np.max(np.abs(xk))
        if local_max > dr_max:
            dr_max = local_max
        
        # Make sure the update is applied relative to the original radii
        r[k, :] = r_old[k, :]
        _update_r_p_rho(r[k], xk, p[k], rho[k], vol_old)

    for k in range(s):
        for i in range(Np1 - 1):
            if r[k, i + 1] - r[k, i] <= 0.0:
                return STATUS_SHELL_CROSSING, dr_max, -1.0

    he_res = compute_he_resid_norm(r, rho, p, m, bkg_param)

    return STATUS_OK, dr_max, he_res

@njit(
    types.Tuple((int64, float64, float64))(
        float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64[:]
    ), cache=True, fastmath=True,
)
def revirialize_interp_gs_diagnostics(
    r, rho, p, m, bkg_param
) -> tuple[int, float, float]:
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
    n_int = Np1 - 2
    a  = np.empty(n_int, dtype=np.float64)
    b  = np.empty(n_int, dtype=np.float64)
    c  = np.empty(n_int, dtype=np.float64)
    y  = np.empty(n_int, dtype=np.float64)
    xk = np.empty(n_int, dtype=np.float64)
    vol_old = np.empty(Np1 - 1, dtype=np.float64)
    m_totk = np.empty(Np1, dtype=np.float64)

    for k in range(s):
        interp_m_enc(k, r, m, m_totk)

        if add_bkg_flag: # For the future: make this in-place too
            m_totk += add_bkg_pot(r[k], bkg_param)
        
        build_tridiag_system(r[k], rho[k], p[k], m_totk, a, b, c, y)
        solve_tridiagonal_thomas(a, b, c, y, xk)
        
        local_max = np.max(np.abs(xk))
        if local_max > dr_max:
            dr_max = local_max
        
        _update_r_p_rho(r[k], xk, p[k], rho[k], vol_old)

    for k in range(s):
        for i in range(Np1 - 1):
            if r[k, i + 1] - r[k, i] <= 0.0:
                return STATUS_SHELL_CROSSING, dr_max, -1.0

    he_res = compute_he_resid_norm(r, rho, p, m, bkg_param)

    return STATUS_OK, dr_max, he_res