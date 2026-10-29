import numpy as np
from numba import njit, float64, int64, void

# -----------------------------------------------------------------------------
# Private scalar helpers
# -----------------------------------------------------------------------------

@njit(cache=True, fastmath=True, inline="always")
def _loglog_interval_index(r_eval, r):
    """
    Return the lower index of the interval used for piecewise log-log
    interpolation/extrapolation on a monotonic 1D radial grid.

    If the grid begins at r=0, index 0 is never used to form a logarithmic
    slope; the first finite-radius interval [1, 2] is used for inward
    extrapolation instead.
    """
    n = r.shape[0]

    i_min = 1 if r[0] == 0.0 else 0
    i_max = n - 2

    if r_eval <= r[i_min]:
        return i_min

    if r_eval >= r[n - 1]:
        return i_max

    # Find the last index i such that r[i] <= r_eval.
    lo = i_min
    hi = n - 1

    while lo < hi:
        mid = (lo + hi + 1) // 2
        if r[mid] <= r_eval:
            lo = mid
        else:
            hi = mid - 1

    i = lo
    if i > i_max:
        i = i_max

    return i

@njit(cache=True, fastmath=True, inline="always")
def _interp_loglog_single(r_eval, r, x):
    """
    Piecewise power-law interpolation/extrapolation of a positive quantity
    defined on a monotonic 1D radial grid.

    For grids beginning at r=0, x[0] is returned exactly at the origin and
    the first finite-radius interval is used for inward extrapolation at
    positive radii.
    """
    if r_eval == r[0]:
        return x[0]

    # Guard against log(0) for a non-positive evaluation radius. This mirrors
    # the previous public routines for extrapolation away from an explicit
    # origin.
    r_eval_log = r_eval if r_eval > 0.0 else 1e-300

    i = _loglog_interval_index(r_eval_log, r)

    r0 = r[i]
    r1 = r[i + 1]
    x0 = x[i]
    x1 = x[i + 1]

    # Safe fallback for quantities/intervals that cannot be represented in
    # log-log space. For outward extrapolation, hold the outermost value;
    # otherwise hold the lower-interval value.
    if r0 <= 0.0 or r1 <= r0 or x0 <= 0.0 or x1 <= 0.0:
        if r_eval_log >= r[r.shape[0] - 1]:
            return x[x.shape[0] - 1]
        return x0

    a = (np.log(x1) - np.log(x0)) / (np.log(r1) - np.log(r0))
    return x0 * np.exp(a * np.log(r_eval_log / r0))

@njit(cache=True, fastmath=True, inline="always")
def _interp_loglog_single_with_derivative(r_eval, r, x):
    """
    As ``_interp_loglog_single``, but also return dx/dr for the local
    piecewise power law.
    """
    if r_eval == r[0]:
        return x[0], 0.0

    r_eval_log = r_eval if r_eval > 0.0 else 1e-300

    i = _loglog_interval_index(r_eval_log, r)

    r0 = r[i]
    r1 = r[i + 1]
    x0 = x[i]
    x1 = x[i + 1]

    if r0 <= 0.0 or r1 <= r0 or x0 <= 0.0 or x1 <= 0.0:
        if r_eval_log >= r[r.shape[0] - 1]:
            return x[x.shape[0] - 1], 0.0
        return x0, 0.0

    a = (np.log(x1) - np.log(x0)) / (np.log(r1) - np.log(r0))
    x_eval = x0 * np.exp(a * np.log(r_eval_log / r0))
    dxdr = a * x_eval / r_eval_log

    return x_eval, dxdr

# -----------------------------------------------------------------------------
# Linear interpolation
# -----------------------------------------------------------------------------

@njit(float64[:](float64[:], float64[:]), fastmath=True, cache=True)
def interp_linear_to_interfaces(r_edges_1d, q_cells_1d) -> np.ndarray:
    """
    Linearly interpolate a cell-centered quantity q to interface locations
    using the non-uniform-spacing-aware formula:

        fac_i      = (r_i - r_{i-1}) / (r_{i+1} - r_{i-1})
        q_interface = q_{i-1} + fac_i * (q_i - q_{i-1})

    Parameters
    ----------
    r_edges_1d : (N+1,) float64
        Edge (interface) radii, monotonic increasing.
    q_cells_1d : (N,) float64
        Cell-centered quantity defined between edges.

    Returns
    -------
    out : (N-1,) float64
        Interpolated values at interior interfaces i=1..N-1.
    """
    num = r_edges_1d[1:-1] - r_edges_1d[:-2]
    den = r_edges_1d[2:] - r_edges_1d[:-2]
    fac = num / den

    qL = q_cells_1d[:-1]
    qR = q_cells_1d[1:]

    return qL + fac * (qR - qL)

# -----------------------------------------------------------------------------
# Midpoint-defined intensive quantities
# -----------------------------------------------------------------------------

@njit(float64[:, :](float64[:], float64[:, :], float64[:, :]), fastmath=True, cache=True)
def interp_intensive_loglog(rmid0, rmid, x) -> np.ndarray:
    """
    Log-log interpolate each species onto ``rmid0`` without summing.

    Parameters
    ----------
    rmid0 : (N0,) float64
        Target midpoint radii.
    rmid : (s, N) float64
        Per-species midpoint radii; each row is monotonic increasing and
        begins at positive radius.
    x : (s, N) float64
        Per-species intensive values at those midpoints.

    Returns
    -------
    out : (s, N0) float64
        Interpolated values for each species.
    """
    s = rmid.shape[0]
    M = rmid0.shape[0]

    out = np.empty((s, M), dtype=np.float64)

    for k in range(s):
        for t in range(M):
            out[k, t] = _interp_loglog_single(
                rmid0[t],
                rmid[k],
                x[k],
            )

    return out


@njit(float64[:](float64[:], float64[:, :], float64[:, :]), fastmath=True, cache=True)
def sum_intensive_loglog(rmid0, rmid, x):
    """
    Sum a midpoint-defined intensive quantity from all species onto ``rmid0``
    using piecewise power-law interpolation/extrapolation.

    This accumulates directly into the output array rather than first
    allocating a full (species, radius) interpolation array.
    """
    s = rmid.shape[0]
    M = rmid0.shape[0]

    out = np.zeros(M, dtype=np.float64)

    for k in range(s):
        for t in range(M):
            out[t] += _interp_loglog_single(
                rmid0[t],
                rmid[k],
                x[k],
            )

    return out


@njit(float64(float64, float64[:, :], float64[:, :]), fastmath=True, cache=True)
def sum_intensive_loglog_single(rmid0, rmid, x) -> float:
    """
    Sum a midpoint-defined intensive quantity over all species at one radius.
    """
    out = 0.0

    for k in range(rmid.shape[0]):
        out += _interp_loglog_single(
            rmid0,
            rmid[k],
            x[k],
        )

    return out


# -----------------------------------------------------------------------------
# Edge-defined extensive quantities
# -----------------------------------------------------------------------------

@njit(float64[:](float64[:], float64[:, :], float64[:, :]), fastmath=True, cache=True)
def sum_extensive_loglog(r0, r, x) -> np.ndarray:
    """
    Sum an edge-defined extensive quantity from all species onto ``r0`` using
    piecewise power-law interpolation/extrapolation.

    Grids may begin at r=0. At an explicit origin, x[:, 0] is used exactly;
    positive radii interior to the first finite edge use the first finite
    log-log interval for inward extrapolation.
    """
    s = r.shape[0]
    M = r0.shape[0]

    out = np.zeros(M, dtype=np.float64)

    for k in range(s):
        for t in range(M):
            out[t] += _interp_loglog_single(
                r0[t],
                r[k],
                x[k],
            )

    return out

@njit(float64(float64, float64[:, :], float64[:, :]), fastmath=True, cache=True)
def sum_extensive_loglog_single(r0, r, x) -> float:
    """
    Sum an edge-defined extensive quantity over all species at one radius.
    """
    out = 0.0

    for k in range(r.shape[0]):
        out += _interp_loglog_single(
            r0,
            r[k],
            x[k],
        )

    return out


# -----------------------------------------------------------------------------
# Single-species log-log interpolation
# -----------------------------------------------------------------------------

@njit(float64[:](float64[:], float64[:, :], float64[:, :], int64), fastmath=True, cache=True)
def interp_species_loglog(r0, r, x, k):
    """
    Log-log interpolation/extrapolation of species ``k`` from ``r[k]`` onto
    the target radii ``r0``.
    """
    M = r0.shape[0]
    out = np.empty(M, dtype=np.float64)

    for t in range(M):
        out[t] = _interp_loglog_single(
            r0[t],
            r[k],
            x[k],
        )

    return out

@njit(float64(float64, float64[:, :], float64[:, :], int64), fastmath=True, cache=True)
def interp_species_loglog_single(r0, r, x, k) -> float:
    """
    Log-log interpolation/extrapolation of species ``k`` at one radius.
    """
    return _interp_loglog_single(
        r0,
        r[k],
        x[k],
    )

# -----------------------------------------------------------------------------
# Total enclosed mass on a selected species grid
# -----------------------------------------------------------------------------

@njit(void(int64, float64[:, :], float64[:, :], float64[:]), fastmath=True, cache=True)
def interp_m_enc(k, r, m, m_tot_on_k):
    """
    Fill ``m_tot_on_k`` with total enclosed mass evaluated on species-k's
    radial grid.

    Species k's own enclosed mass is copied directly because it is already
    defined on the target grid. Other species are log-log interpolated or
    extrapolated onto that grid.

    Parameters
    ----------
    k : int
        Species index.
    r : (s, N+1) float64
        Edge radii per species.
    m : (s, N+1) float64
        Enclosed mass per species at edges.
    m_tot_on_k : (N+1,) float64
        Preallocated output buffer, filled in place.
    """
    s, Np1 = r.shape
    rk = r[k]

    # Species k already lives on the target grid.
    for t in range(Np1):
        m_tot_on_k[t] = m[k, t]

    m_tot_on_k[0] = 0.0

    # Add all other species after interpolation onto species k's grid.
    for j in range(s):
        if j == k:
            continue

        for t in range(1, Np1):
            m_tot_on_k[t] += _interp_loglog_single(
                rk[t],
                r[j],
                m[j],
            )

    m_tot_on_k[0] = 0.0

@njit(void(int64, float64[:, :], float64[:, :], float64[:], float64[:]), fastmath=True, cache=True)
def interp_m_enc_and_K(k, r, m, m_tot_on_k, K_on_k):
    """
    Fill ``m_tot_on_k`` with total enclosed mass on species-k's radial grid
    and ``K_on_k`` with dM_other/dr on the same grid.

    Species k's own enclosed mass is copied directly into ``m_tot_on_k`` and
    contributes nothing to ``K_on_k`` because it is Lagrangian on this grid.
    Only the interpolated contributions from the other species enter K.

    Parameters
    ----------
    k : int
        Species index.
    r : (s, N+1) float64
        Edge radii per species.
    m : (s, N+1) float64
        Enclosed mass per species at edges.
    m_tot_on_k : (N+1,) float64
        Preallocated output buffer for total enclosed mass.
    K_on_k : (N+1,) float64
        Preallocated output buffer for dM_other/dr.
        Only interior values K_on_k[1:N] are physically used downstream.
    """
    s, Np1 = r.shape
    N = Np1 - 1
    rk = r[k]

    # Species k already lives on the target grid; initialize K to zero.
    for t in range(Np1):
        m_tot_on_k[t] = m[k, t]
        K_on_k[t] = 0.0

    m_tot_on_k[0] = 0.0
    K_on_k[0] = 0.0

    # Add interpolated mass and derivative contributions from other species.
    for j in range(s):
        if j == k:
            continue

        for t in range(1, Np1):
            m_j, dm_j_dr = _interp_loglog_single_with_derivative(
                rk[t],
                r[j],
                m[j],
            )

            m_tot_on_k[t] += m_j
            K_on_k[t] += dm_j_dr

    # This endpoint derivative is not used downstream; retain existing API
    # behavior explicitly.
    K_on_k[N] = 0.0