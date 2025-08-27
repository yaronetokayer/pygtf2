import numpy as np
from numba import njit, float64, types

def revirialize(r, rho, p, m_tot) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    """
    Re-virializes the system state by solving for radius adjustments and updating physical quantities.

    Arguments
    ---------
    r : ndarray
        Radial coordinates.
    rho : ndarray
        Density values.
    p : ndarray
        Pressure values.
    m : ndarray
        Total enclosed mass at each radial grid point, including baryons/perturbers.

    Returns
    -------
    tuple or None
        Tuple containing:
            - r_new : ndarray
                Updated radial coordinates.
            - rho_new : ndarray
                Updated density values.
            - p_new : ndarray
                Updated pressure values.
            - v2_new : ndarray
                Updated velocity dispersion.
            - dr_max_new : float
                Maximum absolute change in radius.
        Returns None if any updated velocity dispersion is unphysical (negative).

    Notes
    -----
    The function solves a tridiagonal system to compute radius corrections, then updates
    density, pressure, and velocity dispersion accordingly. If any velocity dispersion
    becomes negative, the function returns None.
    """

    # Solve for corrections to r
    a, b, c, y = build_tridiag_system(r, rho, p, m_tot)
    x = solve_tridiagonal_frank(a, b, c, y)

    # Update arrays accordingly
    r_new, p_new, rho_new = _update_r_p_rho(r, x, p, rho)

    # Check for shell crossing
    if np.any((r_new[1:] - r_new[:-1]) <= 0.0):
        return None  
    
    dr_max_new = float(np.max(np.abs(x)))

    return r_new, rho_new, p_new, dr_max_new

@njit(float64[:](float64[:]), cache=True, fastmath=True)
def compute_mass(m) -> np.ndarray:
    """
    Placeholder funcion to compute mass used in build_tridiag_system.
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
    Ensured positivity and stability.

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
def build_tridiag_system(r, rho, p, m_tot) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct the tridiagonal matrix system (A·X = Y) used in the revirialization step.

    Arguments
    ---------
    r : ndarray
        Radial grid points (length = n + 1)
    rho : ndarray
        Density at each radial grid point (length = n)
    p : ndarray
        Pressure at each radial grid point (length = n)
    m_tot : ndarray
        Total enclosed mass at each radial grid point, including baryons/perturbers (length = n + 1)

    Returns
    -------
    ab : ndarray
        Banded matrix (3, n-1) for use with solve_banded
    y : ndarray
        Right-hand side of the linear system (length = n-1)
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

    return a, b, c, y

@njit(float64[:](float64[:], float64[:], float64[:], float64[:]), cache=True, fastmath=True)
def solve_tridiagonal_frank(a, b, c, y):
    """
    Solve a tridiagonal system Ax = y using the Thomas algorithm.
    This is Frank's implementation from numerical recipes.
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
