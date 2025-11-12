import numpy as np
from numba import njit, float64, types
from pygtf2.util.interpolate import interp_m_enc
from pygtf2.util.calc import add_bkg_pot

def revirialize_interp(r, rho, p, m, bkg_param) -> tuple[str, np.ndarray | None, np.ndarray | None, np.ndarray | None, float | None]:
    """
    Multi-species re-virialization.
    Solves for radius adjustments and updates physical quantities for all species.
    Assumes all species have aligned radial bins.
    Interpolates the mass enclosed for the radial bins of each species.
    Updates to per-species r arrays are fed back into next species revir - this is found to be necessary.

    Parameters
    ----------
    r : ndarray, shape (s, N+1)
        Edge radii per species.
    rho : ndarray, shape (s, N)
        Shell densities per species.
    p : ndarray, shape (s, N)
        Shell pressures per species.
    m : ndarray, shape (s, N+1)
        Total enclosed mass at edges, per species.
    bkg_param : ndarray, shape (4,)
        Parameters for background potential

    Returns
    -------
    status : str
        'ok' if successful, 'shell_crossing' if any radii cross.
    r_new : ndarray or None, shape (s, N+1)
        Updated edge radii per species, or None if shell crossing.
    rho_new : ndarray or None, shape (s, N)
        Updated shell densities per species, or None if shell crossing.
    p_new : ndarray or None, shape (s, N)
        Updated shell pressures per species, or None if shell crossing.
    dr_max : float or None
        Global maximum |dr/r| across all species.
    he_res : float or None
        Norm of HE residual for updated profile, or None if shell crossing.

    Notes
    -----
    This function solves a tridiagonal system to compute radius corrections for each species,
    then updates density and pressure accordingly. If any radii cross, the function returns
    'shell_crossing' and None for all outputs except status.
    """
    s, _ = r.shape
    r_copy = r.copy()
    # r_new   = np.empty_like(r)
    rho_new = np.empty_like(rho)
    p_new   = np.empty_like(p)
    dr_max = 0.0
    add_bkg_flag = bkg_param[0] != -1

    for k in range(s):
        m_totk = interp_m_enc(k, r_copy, m)
        if add_bkg_flag:
            m_totk += add_bkg_pot(r[k], bkg_param)
        a, b, c, y = build_tridiag_system(r[k], rho[k], p[k], m_totk)
        xk = solve_tridiagonal_thomas(a, b, c, y)
        dr_max = max(dr_max, float(np.max(np.abs(xk))))
        rk, pk, rhok = _update_r_p_rho(r[k], xk, p[k], rho[k])
        r_copy[k]  = rk
        # r_new[k]   = rk
        p_new[k]   = pk
        rho_new[k] = rhok
        # print('species', k)
        # for i in range(len(a)):
        #     print(a[i], b[i], c[i], y[i])

    if np.any((r_copy[:,1:] - r_copy[:,:-1]) <= 0.0):
        return 'shell_crossing', r_copy, None, None, dr_max, None

    he_res = compute_he_resid_norm(r_copy, rho_new, p_new, m, bkg_param)
    # he_res = 1.0

    return 'ok', r_copy, rho_new, p_new, dr_max, he_res

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

@njit(types.Tuple((float64[:], float64[:], float64[:]))
      (float64[:], float64[:], float64[:], float64[:]),
      cache=True, fastmath=True)
def _update_r_p_rho(r, x, p, rho) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Updates r, and then finds p, rho, and v2 based on exact volume ratios.
    Ensures positivity and stability.

    r: edge radii, shape (N+1,)
    x: interior stretch, x_j = dr_j / r_j for j=1..N-1, shape (N-1,)
    p, rho: shell-centered arrays, shape (N,)
    """
    r_new = r.copy()
    r_new[1:-1] *= (1.0 + x)  # inner/outer edges fixed

    V_old = r[1:]**3 - r[:-1]**3
    V_new = r_new[1:]**3 - r_new[:-1]**3

    # guard against underflow
    tiny = np.finfo(np.float64).tiny
    V_new = np.maximum(V_new, tiny)

    ratio = V_old / V_new
    gamma = 5.0 / 3.0

    rho_new = rho * ratio
    p_new   = p * ratio**gamma

    return r_new, p_new, rho_new

@njit(types.Tuple((float64[:], float64[:], float64[:], float64[:]))
      (float64[:], float64[:], float64[:], float64[:]), cache=True, fastmath=True)
def build_tridiag_system(r, rho, p, m_tot) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the tridiagonal linear system A·x = y for the interior fractional radius shifts.

    Parameters
    ----------
    r : ndarray, shape (N,)
        Edge radii (N = number of edges = number of shells + 1).
    rho : ndarray, shape (N-1,)
        Shell-centered densities.
    p : ndarray, shape (N-1,)
        Shell-centered pressures.
    m_tot : ndarray, shape (N,)
        Total enclosed mass at the same edge radii as `r`.

    Returns
    -------
    a : ndarray, shape (N-2,)
        Subdiagonal coefficients (multiply x_{j-1}) for interior nodes j=1..N-2.
    b : ndarray, shape (N-2,)
        Main diagonal coefficients (multiply x_j) for interior nodes.
    c : ndarray, shape (N-2,)
        Superdiagonal coefficients (multiply x_{j+1}) for interior nodes.
    y : ndarray, shape (N-2,)
        Right-hand side vector for the interior nodes.

    Notes
    -----
    - The unknown vector x contains the interior fractional displacements x_j = Δr_j / r_j
      (excluding the fixed inner and outer edges), so the returned arrays all have length M-2.
    - The routine linearizes the hydrostatic update using finite differences and geometric
      volume factors. Small numerical floors are applied to pressure differences and density sums
      to prevent divide-by-zero or overflow. The outputs are arranged for direct use with the
      tridiagonal solver used elsewhere in this module.
    """
    rL = r[:-2]         # Left radial grid points
    rR = r[2:]          # Right radial grid points
    rC = r[1:-1]        # Central radial grid points

    # Central differences
    dr = rR - rL
    inv_dr = 1.0 / dr

    # Pressure gradient and density sum for difference equations
    dP   = p[1:] - p[:-1]                           # Pressure difference
    drho = rho[1:] + rho[:-1]

    # floors to avoid divide-by-zero/inf
    tiny = np.finfo(np.float64).tiny
    dP   = np.where(np.abs(dP)   < tiny, np.copysign(tiny, dP),   dP)
    drho = np.where(drho         < tiny, tiny,                  drho)

    # Geometric volume factors
    rR3 = rR**3
    rC3 = rC**3
    rL3 = rL**3
    r3a = rR3 / (rR3 - rC3)
    r3c = rC3 / (rC3 - rL3)
    r3b = r3a - 1.0
    r3d = r3c - 1.0

    q1 = rR * inv_dr
    q2 = q1 - 1.0

    dd = -(4.0 / m_tot[1:-1]) * ( (rC * rC) * inv_dr ) * (dP / drho)

    c1 = 5.0 * dd * (p[1:] / dP) - 3.0 * (rho[1:] / drho)
    c2 = 5.0 * dd * (p[:-1] / dP) + 3.0 * (rho[:-1] / drho)

    y = dd - 1.0

    a = r3d * c2 - q2                               # Subdiagonal
    b = -2.0 - r3b * c1 - r3c * c2                  # Main diagonal, except first element
    c = r3a * c1 + q1                               # Superdiagonal

    # Enforce dp/dr = 0 for i=1 - seems to work either way
    # den1 = rR3[0] - rC3[0]   # Δ(r^3)_1
    # den0 = rC3[0] - rL3[0]   # Δ(r^3)_0

    # a[0] = 0.0
    # b[0] = 5.0 * rC3[0] * ( p[1] / den1 + p[0] / den0 )
    # c[0] = -5.0 * p[1] * (rR3[0] / den1)   # note: rC3[1] == rR3[0]
    # y[0] = -(p[1] - p[0])

    return a, b, c, y

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

@njit(float64[:](float64[:], float64[:], float64[:], float64[:]), cache=True, fastmath=True)
def solve_tridiagonal_thomas(a, b, c, y) -> np.ndarray:
    """
    Solve a tridiagonal system Ax = y using the Thomas algorithm.
    This is an implementation from numerical recipes.
.
    Parameters
    ----------
    a : ndarray
        Subdiagonal (length n-1)
    b : ndarray
        Main diagonal, except first element (length n-1)
    c : ndarray
        Superdiagonal (length n-1)
    y : ndarray
        Right-hand side vector (length n-1)

    Returns
    -------
    x : ndarray
        Solution vector (length n)
    """
    n = b.size

    u   = np.empty(n, dtype=np.float64)
    gam = np.empty(n, dtype=np.float64)
    bet = b[0]
    u[0] = y[0] / bet

    for i in range(1, n):
        gam[i] = c[i-1] / bet
        bet = b[i] - a[i] * gam[i]
        u[i] = ( y[i] - a[i] * u[i-1] ) / bet

    for i in range(n - 2, -1, -1):
        u[i] -= gam[i+1] * u[i+1]

    return u

@njit(float64(float64[:, :], float64[:, :], float64[:, :], float64[:,:], float64[:]),
    fastmath=True,cache=True)
def compute_he_resid_norm(r, rho, p, m, bkg_param):
    """
    Compute an (unscaled) norm of the HE residual
    """
    s, Np1 = r.shape
    res_vec = np.empty((s, Np1-2), dtype=np.float64)
    add_bkg_flag = bkg_param[0] != -1
    
    dp = p[:, 1:] - p[:, :-1]
    srho = rho[:, 1:] + rho[:, :-1]
    dr = r[:, 2:] - r[:, :-2]
    rC = r[:, 1:-1]

    for k in range(s):
        m_totk = interp_m_enc(k, r, m)
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