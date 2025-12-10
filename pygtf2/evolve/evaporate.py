import numpy as np
from numba import njit, float64, void
from pygtf2.util.interpolate import sum_extensive_loglog

@njit
def geomspace_numba(xmin, xmax, num):
    out = np.empty(num, dtype=np.float64)
    log_min = np.log(xmin)
    log_max = np.log(xmax)
    step = (log_max - log_min) / (num - 1)
    for i in range(num):
        out[i] = np.exp(log_min + step * i)
    return out

@njit
def erfc_numba(x):
    # Abramowitz & Stegun approximation
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    
    sign = 1.0
    if x < 0:
        sign = -1.0
    t = 1.0 / (1.0 + p * abs(x))
    y = (((((a5*t + a4)*t) + a3)*t + a2)*t + a1) * t
    erf_val = 1.0 - y * np.exp(-x*x)
    return 1.0 - sign * erf_val

@njit
def interp_numba(x, xp, fp):
    """
    xp, fp: 1D increasing arrays
    returns linear interpolation at x
    """
    n = xp.size
    if x <= xp[0]:
        return fp[0]
    if x >= xp[n-1]:
        return fp[n-1]

    # Binary search
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (hi + lo) // 2
        if xp[mid] > x:
            hi = mid
        else:
            lo = mid

    # Linear interp
    t = (x - xp[lo]) / (xp[hi] - xp[lo])
    return fp[lo] + t * (fp[hi] - fp[lo])

@njit(void(float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64), fastmath=True, cache=True)
def evaporate(r, rmid, m, v2, rho, dt):
    """
    Compute evaporation of particles from the halo beyond the escape radius.
    
    Parameters
    ----------
    r : ndarray, shape (s, N+1)
        Radial bin edges
    rmid : ndarray, shape (s, N)
        Radial bin centers
    m : ndarray, shape (s, N+1)
        Mass enclosed within each radial bin edge (modified in place)
    v2 : ndarray, shape (s, N)
        Mean squared velocity in each radial bin
    rho : ndarray, shape (s, N)
        Density in each radial bin (modified in place)
    dt : float
        Duration of current timestep
    m_esc : float
        Mass threshold to define r_esc
    """
    s, Np1 = r.shape
    N = Np1 - 1

    # --- Compute r_esc
    m_target = 0.95

    r_shared = np.empty(Np1, dtype=np.float64)
    r_shared[0] = 0.0
    r_shared_min = np.min(r[:, 1])
    r_shared_max = np.max(r[:, -1])
    r_shared[1:] = geomspace_numba(r_shared_min, r_shared_max, N)

    m_tot = sum_extensive_loglog(r_shared, r, m)
    r_esc = interp_numba(m_target, m_tot, r_shared)

    # --- Loop through all s × N cells and update rho where needed
    for i in range(s):
        for j in range(N):
            rm = rmid[i, j]

            if rm > r_esc:      # evaporation applies only outside r_esc
                v2_ij = v2[i, j]
                rho_ij = rho[i, j]

                # Escape velocity and sigma
                v_esc = np.sqrt(2.0 / rm)
                sig   = np.sqrt(v2_ij / 3.0)

                x = v_esc / (np.sqrt(2.0) * sig)

                fk_esc = erfc_numba(x) + (2.0 / np.sqrt(np.pi)) * x * np.exp(-x*x)

                t_cross = rm / sig
                t_evap  = t_cross / fk_esc

                param = 1e6
                rho[i, j] = rho_ij * (1.0 -  param * dt / t_evap)
            # else: do nothing

    # --- Update m using cumsum over each row
    for i in range(s):
        # compute cell volumes for row i
        for j in range(N):
            cell_vol = (r[i, j+1]**3 - r[i, j]**3) / 3.0
            m[i, j+1] = m[i, j] + cell_vol * rho[i, j]

### READABLE NUMPY VERSION:
# def evaporate(r, rmid, m, v2, rho, dt, m_esc=0.95):
#     """
#     Compute evaporation of particles from the halo beyond the escape radius.
    
#     Parameters
#     ----------
#     r : ndarray, shape (s, N+1)
#         Radial bin edges
#     rmid : ndarray, shape (s, N)
#         Radial bin centers
#     m : ndarray, shape (s, N+1)
#         Mass enclosed within each radial bin edge (modified in place)
#     v2 : ndarray, shape (s, N)
#         Mean squared velocity in each radial bin
#     rho : ndarray, shape (s, N)
#         Density in each radial bin (modified in place)
#     dt : float
#         Duration of current timestep
#     m_esc : float
#         Mass threshold to define r_esc
#     """

#     s, Np1 = r.shape
#     N = Np1 - 1

#     #--- Compute r_esc
#     r_shared     = np.zeros((Np1,))
#     r_shared_min = np.min(r[:,1])
#     r_shared_max = np.max(r[:,-1])
#     r_shared[1:] = np.geomspace(r_shared_min, r_shared_max, num=N, endpoint=True, dtype=float)

#     m_tot = sum_extensive_loglog(r_shared, r, m)
#     r_esc = np.interp(m_esc, m_tot, r_shared)

#     #--- Restrict attention to halo beyond r_esc
#     mask = rmid > r_esc      # shape (s, N)

#     rmid_mask = rmid[mask]
#     v2_mask   = v2[mask]
#     rho_mask  = rho[mask]

#     #--- Compute escape velocity and 1D velocity dispersion
#     v_esc = np.sqrt(2.0 / rmid_mask)
#     sig   = np.sqrt(v2_mask / 3.0)

#     #--- Compute escape fraction
#     x = v_esc / (np.sqrt(2) * sig)
#     fk_esc = erfc(x) + (2.0 / np.sqrt(np.pi)) * x * np.exp(-x*x)

#     #--- Compute t_evap
#     t_cross = rmid_mask / sig
#     t_evap  = t_cross / fk_esc

#     #--- Update rho in place
#     rho[mask] = rho_mask * (1.0 - dt/t_evap)

#     #--- Update m in place
#     cell_volumes = (r[:,1:]**3 - r[:,:-1]**3) * (1.0/3.0)
#     dm_cells = cell_volumes * rho
#     m[:,1:] = np.cumsum(dm_cells, axis=1)