import numpy as np
from numba import njit, float64, boolean, types

@njit(float64[:](float64, float64, float64, float64,
                 float64[:], float64[:], float64[:], boolean),
      cache=True, fastmath=True)
def compute_luminosities(a, b, c, sigma_m, r, v2, p, cored) -> np.ndarray:
    """ 
    Compute luminosity of each shell interface based on temperature gradient and conductivity.
    e.g, Eq. (2) in Nishikawa et al. 2020.

    Arguments
    ----------
    a : float
        Constant 'a' in the conductivity formula.
    b : float
        Constant 'b' in the conductivity formula.
    c : float
        Constant 'c' in the conductivity formula.
    sigma_m : float
        Interaction cross section in dimensionless units.
    r : ndarray
        Radial grid points, including cell edges (length = ngrid + 1).
    v2 : ndarray
        Velocity dispersion squared for each cell (length = ngrid).
    p : ndarray
        Pressure for each cell (length = ngrid).
    cored : bool
        Whether the system has a central core (i.e., ABG with gamma < 1.0).

    Returns
    -------
    lum : ndarray
        Luminosities at each shell boundary (same length as r).
    """
    lum = np.empty(r.shape, dtype=np.float64)

    # Compute temperature gradient and midpoints using cell-centered values
    dTdr = ( v2[1:] - v2[:-1] ) / ( r[2:] - r[:-2] )
    vmed = np.sqrt( 0.5 * ( v2[1:] + v2[:-1] ) )
    pmed = 0.5 * ( p[1:] + p[:-1] )

    # One sided difference for cored profiles (i.e., ABG with gamma < 1)
    if cored:
        dTdr[0] = (v2[1] - v2[0]) / (r[2] - r[1])
        vmed[0] = np.sqrt(v2[0])
        pmed[0] = p[0]

    fac1 = -3.0 * vmed * r[1:-1]**2
    fac2 = (a / b) * sigma_m**2 + ( (1.0 / c) / pmed )

    lum[1:-1] = (fac1 / fac2) * dTdr

    # Boundary conditions
    lum[0] = 0.0
    lum[-1] = 0.0

    return lum

@njit(types.Tuple((float64[:], float64, float64))(
    float64[:], float64[:], float64[:], float64[:], float64, float64
    ), cache=True, fastmath=True)
def conduct_heat(m, u, rho, lum, dt_prop, eps_du) -> tuple[np.ndarray, float, float]:
    """
    Conduct heat and adjust internal energies accordingly.
    Ignores PdV work and assumes fixed density.
    Updates internal energy and recomputes pressure.
    Updates dt_prop if necessary based on max relative change in u.

    Arguments
    ---------
    m : np.ndarray
        Enclosed mass array
    u : np.ndarray
        Internal energy array
    rho : np.ndarray
        Density array
    lum : np.ndarray
        Array of luminosities from compute_luminosities (length = len(state.r))
    dt_prop : float
        Current timestep duration
    eps_du : float
        Maximum allowed relative change in u for convergence

    Returns
    -------
    p : np.ndarray
        Updated pressure array.
    dumax : float
        Max relative change in u.
    dt_prop : float
        Modified timestep.
    """

    dudt = -( lum[1:] - lum[:-1] ) / ( m[1:] - m[:-1] )
    du = dudt * dt_prop

    tiny = np.finfo(np.float64).tiny
    abs_u = np.abs(u)
    abs_u[abs_u < tiny] = tiny
    dumax = np.max(np.abs(du) / abs_u)

    if dumax > eps_du:
        scale = 0.95 * (eps_du / dumax)
        dt_eff = dt_prop * scale
        dumax *= scale
        du *= scale
    else:
        dt_eff = dt_prop

    u_new = u + du
    p_new = ( 2.0 / 3.0 ) * rho * u_new

    return p_new, float(dumax), float(dt_eff)