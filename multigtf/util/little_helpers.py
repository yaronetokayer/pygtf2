import numpy as np 
from numba import njit, float64, void

_TINY64 = np.finfo(np.float64).tiny

@njit(float64(float64[:,:], float64[:,:]), cache=True, fastmath=True)
def max_frac_change(v2, v2_old):
    s, N = v2.shape
    du_max = 0.0

    for k in range(s):
        for i in range(N):
            denom = v2_old[k, i]
            if denom <= _TINY64:
                denom = _TINY64

            du = abs(v2[k, i] - v2_old[k, i]) / denom

            if du > du_max:
                du_max = du

    return du_max