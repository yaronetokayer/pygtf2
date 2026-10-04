import numpy as np
from numba import njit, float64, void


@njit(float64(float64, float64, float64), fastmath=True, cache=True,)
def hernq_static_scalar(r, m_tot, r_s):
    denom = r + r_s
    return m_tot * r * r / (denom * denom)

@njit(float64[:](float64[:], float64, float64), fastmath=True, cache=True)
def hernq_static(r, m_tot, r_s):
    """
    Return Hernquist enclosed mass M(<r) = m_tot*r**2/(r+r_s)**2.

    r is a one-dimensional array of radii and r_s is the scale radius in
    the same length units. m_tot sets the total mass and output mass units.
    Returns an array matching r; inputs are not modified.
    """
    out = np.empty_like(r)

    for i in range(r.shape[0]):
        out[i] = hernq_static_scalar(r[i], m_tot, r_s)

    return out
