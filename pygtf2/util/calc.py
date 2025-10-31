import numpy as np
from numba import njit, float64, int64
from pygtf2.util.interpolate import sum_intensive_loglog_single

def calc_rho_c(rmid, rho):
    """
    Computes central density of system at smallest non-zero radial point.

    Arguments
    ---------
    rmid : ndarray, shape (s, N)
        Midpoint radii per species.
    rho : ndarray, shape (s, N)
        Shell densities per species.

    Returns
    -------
    rho_c : str
        Central density
    """
    r0 = np.min(rmid[:,:])

    rho_c = sum_intensive_loglog_single(r0, rmid, rho)

    return float(rho_c)