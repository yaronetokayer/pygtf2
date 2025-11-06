import numpy as np
from numba import njit, float64

@njit(float64[:](float64[:], float64, float64), fastmath=True, cache=True)
def hernq_static(r, m_tot, r_s):
    """
    Static Hernquist profile.

    Arguments
    ---------
    r : ndarray
    m_tot : float
    r_s : float

    Returns
    -------
    ndarray
    """
    return m_tot * r**2 / (r + r_s)**2