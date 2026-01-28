import numpy as np
from pygtf2.util.interpolate import sum_intensive_loglog
from numba import njit, types, float64

@njit(types.Tuple((float64[:,:], float64[:,:], float64))(float64[:,:], float64[:,:], float64[:,:], float64),
    fastmath=True,cache=True)
def binaries_heating(rmid, rho, v2, dt):
    s, N = rmid.shape
    
    # Constants, see e.g., Bettwieser (1985MNRAS.215..499B)
    pow = 2.0
    c = 1.0e-9 # arbitrary right now
    l = 1.0

    v2_new  = np.zeros_like(v2, dtype=np.float64)
    p_new   = np.zeros_like(v2, dtype=np.float64)
    eps_max = 0.0

    # per species update
    for k in range(s):
        if s == 1:
            rho_tot = rho[k]
            v2_tot  = v2[k]
        else:
            rmidk   = rmid[k]
            rho_tot = sum_intensive_loglog(rmidk, rmid, rho)
            v2_tot  = sum_intensive_loglog(rmidk, rmid, rho*v2) / rho_tot   # Mass-weighted
        eps         = c * rho_tot**pow / v2_tot**(l/2.0)
        v2k_new     = v2[k] + eps * dt
        v2_new[k,:] = v2k_new
        p_new[k,:]  = rho[k] * v2k_new
        for e in eps:
            if e > eps_max:
                eps_max = e
    
    return v2_new, p_new, eps_max