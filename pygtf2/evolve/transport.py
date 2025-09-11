import numpy as np
from numba import njit, float64, types

@njit(
    float64[:, :](float64, float64[:, :], float64[:, :], float64[:, :], float64[:], float64[:,:]),
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
    lnL : ndarray
        lnL array.

    Returns
    -------
    L: ndarray
        Luminosities at each shell boundary (same length as r).
    """
    s, Np1 = r.shape
    L = np.zeros((s, Np1), np.float64) # Initialization takes care of boundary conditions

    for n in range(s):
        # Closure relations
        u_n     = u[n]                 # (N,)
        rho_n   = rho[n]                # (N,)

        # Centered interface values
        rhom = 0.5 * (rho_n[1:] + rho_n[:-1]) # (N-1,)
        umed = 0.5 * (u_n[1:]   + u_n[:-1])   # (N-1,)
        dTdr = 2.0 * (u_n[1:] - u_n[:-1]) / (r[n, 2:] - r[n, :-2])

        fac = (r[n, 1:-1]**2) * (rhom / np.sqrt(umed))
        pref = (-c2) * (mrat[n] * lnL[n,n])

        L[n, 1:-1] = pref * fac * dTdr

    return L

@njit(types.Tuple((float64[:,:], float64, float64))(
    float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:], float64, float64, float64
    ), cache=True, fastmath=True)
def conduct_heat(m, u, rho, lum, lnL, mrat, dt_prop, eps_du, c1) -> tuple[np.ndarray, float, float]:
    """
    Conduct heat and adjust internal energies accordingly.
    Ignores PdV work and assumes fixed density.
    Updates internal energy and recomputes pressure.
    Updates dt_prop if necessary based on max relative change in u.

     See Eq. (42) in Zhong and Shapiro (2025).

    Arguments
    ---------
    m : np.ndarray
        Enclosed mass array
    u : np.ndarray
        Internal energy array
    rho : np.ndarray
        Density array
    lum : np.ndarray
        Array of luminosities from compute_luminosities
    lnL : ndarray
        lnL array
    mrat : ndarray
        Mass ratio of species (length = s)
    dt_prop : float
        Current timestep duration
    eps_du : float
        Maximum allowed relative change in u for convergence
    c1 : float
        Constant 'c1' in the luminosity formula

    Returns
    -------
    p : np.ndarray
        Updated pressure array.
    dumax : float
        Max relative change in u.
    dt_prop : float
        Modified timestep.
    """

    s, Np1 = m.shape
    N = Np1 - 1

    # Outputs
    u_new = np.empty_like(u)
    p_new = np.empty_like(u)  # same (s, N) shape
    du    = np.empty_like(u)

    # ---------- flux-divergence term (per species, per cell) ----------
    # dudt_cond[n, i] = - (L[n, i+1] - L[n, i]) / (M[n, i+1] - M[n, i])
    dudt_cond = np.empty_like(u)
    for n in range(s):
        for i in range(N):
            denom = m[n, i+1] - m[n, i]
            dudt_cond[n, i] = - (lum[n, i+1] - lum[n, i]) / denom

    # ---------- binary heat-exchange source (sum over j != n) ----------
    # dudt_hex[n, i] = c1 * sum_{j != n} lnL[n, j] * rho[j, i] *
    #                  (mrat[j]*u[j, i] - mrat[n]*u[n, i]) / (u[j, i] + u[n, i])^(3/2)
    dudt_hex = np.zeros_like(u)
    for n in range(s):
        for j in range(s):
            if j == n:
                continue
            lnL_nj = lnL[n, j]
            mrj    = mrat[j]
            mrn    = mrat[n]
            for i in range(N):
                uj = u[j, i]
                un = u[n, i]
                denom = (uj + un)**1.5
                term = lnL_nj * rho[j, i] * (mrj * uj - mrn * un) / denom
                dudt_hex[n, i] += term
    dudt_hex *= c1

    # ---------- combine, adapt dt if needed ----------
    dudt = dudt_cond + dudt_hex
    du[:, :] = dudt * dt_prop

    # max relative change across all species/cells
    tiny = np.finfo(np.float64).tiny
    dumax = 0.0
    for n in range(s):
        for i in range(N):
            abs_u = un = u[n, i]
            if abs_u < tiny:
                abs_u = tiny
            rat = abs(du[n, i]) / abs_u
            if rat > dumax:
                dumax = rat

    if dumax > eps_du:
        scale = 0.95 * (eps_du / dumax)
        dt_eff = dt_prop * scale
        dumax *= scale
        du *= scale
    else:
        dt_eff = dt_prop

    # ---------- update u and p ----------
    for n in range(s):
        for i in range(N):
            u_new[n, i] = u[n, i] + du[n, i]
            p_new[n, i] = (2.0 / 3.0) * rho[n, i] * u_new[n, i]

    return p_new, float(dumax), float(dt_eff)