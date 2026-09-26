import numpy as np
from numba import njit, float64, void


@njit(float64(float64, float64, float64), fastmath=True, cache=True,)
def hernq_static_scalar(r, m_tot, r_s):
    denom = r + r_s
    return m_tot * r * r / (denom * denom)

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
    out = np.empty_like(r)

    for i in range(r.shape[0]):
        out[i] = hernq_static_scalar(r[i], m_tot, r_s)

    return out
