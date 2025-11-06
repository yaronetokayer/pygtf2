import numpy as np
from numba import njit, types, float64
from pygtf2.util.interpolate import sum_intensive_loglog_single

def add_bkg_pot(r, m, bkg):
    """
    Add a background potential to the mass array.

    Arguments
    ---------
    r : ndarray, shape (N+1,)
        Edge radii
    m : ndarray, shape (N+1,)
        Enclosed mass at edge radii
    bkg : str
        Background potential selected

    Returns
    -------
    m_tot : ndarray, shape (N+1,)
        The original m array with the background potential added
    """

    pass

@njit(types.float64(types.float64[:, ::1], types.float64[:, ::1]), fastmath=True, cache=True)
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

@njit(float64[:](float64[:], float64[:], float64[:]), fastmath=True, cache=True)
def mass_fraction_radii(r_edges, m_edges, fracs):
    """
    Compute radii at which a given set of mass fractions of the total enclosed mass are reached.

    Parameters
    ----------
    r_edges : array_like, shape (N,)
        Radii of the radial edges (grid points). Expected to be monotonic (typically increasing).
    m_edges : array_like, shape (N,)
        Cumulative (enclosed) mass evaluated at each radius in ``r_edges``.
        The final element ``m_edges[-1]`` is interpreted as the total mass of the species.
    fracs : array_like, shape (M,)
        Mass fractions of the total mass at which the radius is requested (typically in [0, 1]).
        Can be any real values; see Notes for behavior outside the nominal range.

    Returns
    -------
    numpy.ndarray, shape (M,)
        Array of radii corresponding to each requested mass fraction in ``fracs``.
        If the total mass (``m_edges[-1]``) is <= 0, an array of NaNs with the same shape as ``fracs`` is returned.

    Raises
    ------
    ValueError
        If the input arrays have incompatible lengths or if ``m_edges`` and ``r_edges`` do not have at least two points.
        (The implementation assumes at least two edges for meaningful interpolation.)

    Notes
    -----
    - The function computes target enclosed masses as ``fracs * m_tot`` where ``m_tot = m_edges[-1]`` and finds the interval
      in ``m_edges`` that brackets each target. It then linearly interpolates in radius between the two bracketing edges.
    - The search advances monotonically through the mass-edge array for successive targets (amortized O(N + M) cost),
      so performance is best when ``fracs`` are provided in non-decreasing order. The function still works for unsorted ``fracs``.
    - The routine assumes ``m_edges`` is non-decreasing and corresponds to the same ordering as ``r_edges``. No internal sorting is performed.
    - Behavior for targets outside the range of ``m_edges``:
      - Targets >= total mass map to ``r_edges[-1]``.
      - Targets less than the first edge mass are handled by linear interpolation using the first interval and may produce radii
        smaller than ``r_edges[0]`` (i.e., backward extrapolation).
    - Exact matches to an edge mass return the corresponding edge radius.

    Examples
    --------
    >>> import numpy as np
    >>> r = np.array([0.0, 1.0, 2.0])
    >>> m = np.array([0.0, 10.0, 20.0])
    >>> mass_fraction_radii(r, m, np.array([0.25, 0.5, 1.0]))
    array([0.5, 1.0, 2.0])
    """
    # m_edges should be enclosed mass at edges, with m_edges[-1] = species total
    m_tot = m_edges[-1]
    if m_tot <= 0:
        return np.full(fracs.size, np.nan)
    target = fracs * m_tot
    out = np.empty(fracs.size)
    # linear-in-radius search & interpolation on edges
    j = 0
    for i, mt in enumerate(target):
        while j+1 < m_edges.size and m_edges[j+1] < mt:
            j += 1
        if j+1 == m_edges.size:
            out[i] = r_edges[-1]
        else:
            m0, m1 = m_edges[j], m_edges[j+1]
            r0, r1 = r_edges[j], r_edges[j+1]
            t = 0.0 if m1 == m0 else (mt - m0) / (m1 - m0)
            out[i] = r0 + t * (r1 - r0)
    return out