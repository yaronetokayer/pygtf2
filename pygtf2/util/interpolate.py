import numpy as np
from numba import njit, float64, int64, void

@njit(float64[:](float64[:], float64[:]), fastmath=True, cache=True)
def interp_linear_to_interfaces(r_edges_1d, q_cells_1d) -> np.ndarray:
    """
    Linearly interpolate a cell-centered quantity q to interface locations
    using the non-uniform-spacing-aware formula:

        fac_i   = (r_i   - r_{i-1}) / (r_{i+1} - r_{i-1})     for i = 1..N-1
        q_{i|i+1} = q_i + fac_i * (q_{i+1} - q_i)

    Here r_* are edge (interface) radii with length N+1, q_cells has length N,
    and the returned array has length N-1 (interfaces i=1..N-1).

    Parameters
    ----------
    r_edges_1d : (N+1,) float64
        Edge (interface) radii, monotonic increasing.
    q_cells_1d : (N,) float64
        Cell-centered quantity defined between edges.

    Returns
    -------
    out : (N-1,) float64
        Interpolated values at interfaces i=1..N-1.
    """
    # interfaces we fill are i = 1..N-1  -> indices 1: N in edge space
    num = r_edges_1d[1:-1] - r_edges_1d[:-2]          # r_i   - r_{i-1}
    den = r_edges_1d[2:]   - r_edges_1d[:-2]          # r_{i+1} - r_{i-1}
    fac = num / den                                    # shape (N-1,)

    qL = q_cells_1d[:-1]                               # left cell value (i)
    qR = q_cells_1d[1:]                                # right cell value (i+1)
    return qL + fac * (qR - qL)                        # shape (N-1,)

@njit(float64[:, :](float64[:], float64[:, :], float64[:, :]),
      fastmath=True, cache=True)
def interp_intensive_loglog(rmid0, rmid, x) -> np.ndarray:
    """
    Log–log interpolate each species onto rmid0 without summing.

    Parameters
    ----------
    rmid0 : (N0,) float64
        Target midpoints where the summed quantity is evaluated.
    rmid  : (s, N) float64
        Per-species midpoints (monotonic increasing per row; rmid[:, 0] > 0).
    x     : (s, N) float64
        Per-species intensive values at those midpoints (positive for logs).    

    Returns
    -------
    out : (s, N0) float64
        Interpolated values for each species.
    """
    s, N = rmid.shape
    M = rmid0.shape[0]
    out = np.zeros((s, M), dtype=np.float64)

    for j in range(s):
        rj = rmid[j]   # (N,)
        xj = x[j]      # (N,)

        # Loop over target midpoints
        for t in range(M):
            rt = rmid0[t]
            rt_eval = rt if rt > 0.0 else 1e-300

            # Binary search: find j1 such that rj[j1] <= rt < rj[j1+1]
            # Returns last index with rj[idx] <= rt
            lo = 0
            hi = N - 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if rj[mid] <= rt_eval:
                    lo = mid
                else:
                    hi = mid - 1
            j1 = lo

            # Clamp for extrapolation to use nearest interval slope
            if j1 < 0:
                j1 = 0
            elif j1 > N - 2:
                j1 = N - 2

            r0 = rj[j1]
            r1 = rj[j1 + 1]
            x0 = xj[j1]
            x1 = xj[j1 + 1]

            # Piecewise power-law (log–log) interpolation/extrapolation
            a = (np.log(x1) - np.log(x0)) / (np.log(r1) - np.log(r0))
            out[j, t] = x0 * np.exp(a * np.log(rt_eval / r0))

    return out

@njit(float64[:](float64[:], float64[:, :], float64[:, :]),
      fastmath=True, cache=True)
def sum_intensive_loglog(rmid0, rmid, x):
    """
    Sum an intensive, midpoint-defined quantity from all species onto a new
    midpoint grid rmid0, using log–log (piecewise power-law) interpolation
    between adjacent midpoints, with extrapolation outside each species' domain.

    Parameters
    ----------
    rmid0 : (N0,) float64
        Target midpoints where the summed quantity is evaluated.
    rmid  : (s, N) float64
        Per-species midpoints (monotonic increasing per row; rmid[:, 0] > 0).
    x     : (s, N) float64
        Per-species intensive values at those midpoints (positive for logs).

    Returns
    -------
    out : (N0,) float64
        Summed intensive quantity evaluated at rmid0.
    """
    interp = interp_intensive_loglog(rmid0, rmid, x)  # (s, N0)
    return np.sum(interp, axis=0)

# @njit(float64[:](float64[:], float64[:, :], float64[:, :]), fastmath=True, cache=True)
# def sum_intensive_loglog(rmid0, rmid, x) -> np.ndarray:
#     """
#     Sum an intensive, midpoint-defined quantity from all species onto a new
#     midpoint grid rmid0, using log–log (piecewise power-law) interpolation
#     between adjacent midpoints, with extrapolation outside each species' domain.

#     Parameters
#     ----------
#     rmid0 : (N0,) float64
#         Target midpoints where the summed quantity is evaluated.
#     rmid  : (s, N) float64
#         Per-species midpoints (monotonic increasing per row; rmid[:, 0] > 0).
#     x     : (s, N) float64
#         Per-species intensive values at those midpoints (positive for logs).

#     Returns
#     -------
#     out : (N0,) float64
#         Summed intensive quantity evaluated at rmid0.
#     """
#     s, N = rmid.shape
#     M = rmid0.shape[0]
#     out = np.zeros(M, dtype=np.float64)

#     for j in range(s):
#         rj = rmid[j]   # (N,)
#         xj = x[j]      # (N,)

#         # Loop over target midpoints
#         for t in range(M):
#             rt = rmid0[t]
#             # Avoid log(0) if someone passes rt==0 (shouldn't for midpoints)
#             if rt <= 0.0:
#                 rt_eval = 1e-300
#             else:
#                 rt_eval = rt

#             # Binary search: find j1 such that rj[j1] <= rt < rj[j1+1]
#             # Returns last index with rj[idx] <= rt
#             lo = 0
#             hi = N - 1
#             while lo < hi:
#                 mid = (lo + hi + 1) // 2  # upper mid to prevent infinite loop
#                 if rj[mid] <= rt_eval:
#                     lo = mid
#                 else:
#                     hi = mid - 1
#             j1 = lo

#             # Clamp for extrapolation to use nearest interval slope
#             if j1 < 0:
#                 j1 = 0
#             elif j1 > N - 2:
#                 j1 = N - 2

#             r0 = rj[j1]
#             r1 = rj[j1 + 1]
#             x0 = xj[j1]
#             x1 = xj[j1 + 1]

#             # Piecewise power-law (log–log) interpolation/extrapolation
#             a = (np.log(x1) - np.log(x0)) / (np.log(r1) - np.log(r0))
#             x_interp = x0 * np.exp(a * np.log(rt_eval / r0))

#             out[t] += x_interp

#     return out

@njit(float64[:](float64[:], float64[:, :], float64[:, :]), fastmath=True, cache=True)
def sum_extensive_loglog(r0, r, x) -> np.ndarray:
    """
    Sum an edge-defined (extensive) quantity from all species onto a new edge grid r0,
    using log–log (piecewise power-law) interpolation between edges, with inward/outward
    extrapolation. Avoids log(0) by extrapolating from the first finite-radius interval.

    Parameters
    ----------
    r0 : (N0+1,) float64
        Target edge radii.
    r  : (s, N+1) float64
        Per-species edge radii (monotone nondecreasing per row; r[:,0] may be 0).
    x  : (s, N+1) float64
        Per-species values defined at the same edges as r. Should be > 0 for log–log.

    Returns
    -------
    out : (N0+1,) float64
        Summed quantity evaluated at r0.
    """
    s, Np1 = r.shape
    N = Np1 - 1
    M = r0.shape[0]

    out = np.zeros(M, dtype=np.float64)

    for j in range(s):
        rj = r[j]   # (N+1,)
        xj = x[j]   # (N+1,)

        # Precompute a flag for rj[0]==0 to avoid log(0) slopes
        has_zero_edge = (rj[0] == 0.0)

        for t in range(M):
            rt = r0[t]
            rt_eval = rt if rt > 0.0 else 1e-300  # guard for log(rt)

            # Binary search: find j1 such that rj[j1] <= rt < rj[j1+1]
            lo = 0
            hi = N
            while lo < hi:
                mid = (lo + hi) // 2
                if rj[mid] <= rt:
                    lo = mid + 1
                else:
                    hi = mid
            j1 = lo - 1  # clamp into [0, N-1]
            if j1 < 0:
                j1 = 0
            elif j1 > N - 1:
                j1 = N - 1

            # If the lower endpoint is r=0, avoid using that in the log-slope:
            # use the first finite-radius interval [1,2] for inward extrapolation.
            if j1 == 0 and has_zero_edge:
                # Need at least two finite-radius edges; assume N >= 2.
                r_lo = rj[1]
                r_hi = rj[2]
                x_lo = xj[1]
                x_hi = xj[2]
            else:
                # Normal interval (works for interior and outward extrapolation)
                # For the very last index j1==N-1, this is the outermost interval [N-1, N].
                r_lo = rj[j1]
                r_hi = rj[j1 + 1] if j1 < N else rj[N]
                x_lo = xj[j1]
                x_hi = xj[j1 + 1] if j1 < N else xj[N]

                # In the unlikely case r_lo==0 here (shouldn’t happen except j1==0),
                # fall back to the first finite interval as above.
                if r_lo == 0.0 and has_zero_edge:
                    r_lo = rj[1]
                    r_hi = rj[2]
                    x_lo = xj[1]
                    x_hi = xj[2]

            # Power-law interpolation/extrapolation in log–log space
            a = (np.log(x_hi) - np.log(x_lo)) / (np.log(r_hi) - np.log(r_lo))
            x_interp = x_lo * np.exp(a * np.log(rt_eval / r_lo))

            out[t] += x_interp

    return out

import numpy as np
from numba import njit, float64, int64

@njit(float64[:](float64[:], float64[:, :], float64[:, :], int64), fastmath=True, cache=True)
def interp_species_loglog(r0, r, x, k):
    """
    Log–log (piecewise power-law) interpolation/extrapolation of x[k] from r[k] onto r0.

    Parameters
    ----------
    r0 : (M,) float64
        Target edge radii.
    r  : (s, N+1) float64
        Per-species edge radii (monotone nondecreasing per row; r[:,0] may be 0).
    x  : (s, N+1) float64
        Per-species values defined at the same edges as r. Must be > 0 for log–log.
    k  : int
        Species index to interpolate.

    Returns
    -------
    out : (M,) float64
        Interpolated/extrapolated x[k] evaluated at r0.
    """
    # Pull the selected species
    rj = r[k]
    xj = x[k]

    Np1 = rj.shape[0]
    N = Np1 - 1
    M = r0.shape[0]

    out = np.empty(M, dtype=np.float64)

    has_zero_edge = (rj[0] == 0.0)

    for t in range(M):
        rt = r0[t]
        rt_eval = rt if rt > 0.0 else 1e-300  # guard for log(rt)

        # Binary search: find j1 such that rj[j1] <= rt < rj[j1+1]
        lo = 0
        hi = N
        while lo < hi:
            mid = (lo + hi) // 2
            if rj[mid] <= rt:
                lo = mid + 1
            else:
                hi = mid
        j1 = lo - 1
        if j1 < 0:
            j1 = 0
        elif j1 > N - 1:
            j1 = N - 1

        # Avoid log(0) for inward extrapolation when the innermost edge is 0
        if j1 == 0 and has_zero_edge:
            # assumes N >= 2
            r_lo = rj[1]
            r_hi = rj[2]
            x_lo = xj[1]
            x_hi = xj[2]
        else:
            r_lo = rj[j1]
            r_hi = rj[j1 + 1]  # j1 is clamped to <= N-1 so this is safe
            x_lo = xj[j1]
            x_hi = xj[j1 + 1]

            # extra safety if r_lo is still zero
            if r_lo == 0.0 and has_zero_edge:
                r_lo = rj[1]
                r_hi = rj[2]
                x_lo = xj[1]
                x_hi = xj[2]

        # Log–log power-law interpolation/extrapolation
        a = (np.log(x_hi) - np.log(x_lo)) / (np.log(r_hi) - np.log(r_lo))
        out[t] = x_lo * np.exp(a * np.log(rt_eval / r_lo))

    return out

@njit(float64(float64, float64[:, :], float64[:, :]), fastmath=True, cache=True)
def sum_intensive_loglog_single(rmid0, rmid, x) -> float:
    """
    Numba-optimized single-value wrapper for sum_intensive_loglog.
    Returns the summed intensive quantity at a single midpoint radius.
    """
    tmp = np.empty(1, dtype=np.float64)
    tmp[0] = rmid0
    out = sum_intensive_loglog(tmp, rmid, x)
    return out[0]

@njit(float64(float64, float64[:, :], float64[:, :]), fastmath=True, cache=True)
def sum_extensive_loglog_single(r0, r, x) -> float:
    """
    Numba-optimized single-value wrapper for sum_extensive_loglog.
    Returns the summed extensive quantity at a single edge radius.
    """
    tmp = np.empty(1, dtype=np.float64)
    tmp[0] = r0
    out = sum_extensive_loglog(tmp, r, x)
    return out[0]

@njit(float64(float64, float64[:, :], float64[:, :], int64), fastmath=True, cache=True)
def interp_species_loglog_single(r0, r, x, k) -> float:
    """
    Numba-optimized single-value wrapper for interp_species_loglog.
    Returns the summed extensive quantity at a single edge radius.
    """
    tmp = np.empty(1, dtype=np.float64)
    tmp[0] = r0
    out = interp_species_loglog(tmp, r, x, k)
    return out[0]

@njit(void(int64, float64[:, :], float64[:, :], float64[:]), fastmath=True, cache=True)
def interp_m_enc(k, r, m, m_tot_on_k):
    """
    Fill m_tot_on_k with total enclosed mass evaluated on species-k radial grid,
    using piecewise power-law (log-log) interpolation for other species and
    power-law extrapolation beyond their maximum radius.

    Parameters
    ----------
    k : int
        Species index (0 <= k < s).
    r : (s, N+1) float64
        Edge radii per species.
    m : (s, N+1) float64
        Enclosed mass per species at edges.
    m_tot_on_k : (N+1,) float64
        Preallocated output buffer. Filled in place.
    """
    s, Np1 = r.shape
    N = Np1 - 1

    rk = r[k]  # view (N+1,)

    # Start with species k's own contribution
    for t in range(Np1):
        m_tot_on_k[t] = m[k, t]
    m_tot_on_k[0] = 0.0

    # accumulate contributions from other species
    for j in range(s):
        if j == k:
            continue

        rj = r[j]  # (N+1,)
        mj = m[j]  # (N+1,)

        m_last = mj[N]
        rj_max = rj[N]

        for t in range(1, Np1):
            x = rk[t]

            # Constant tail beyond species j's maximum radius
            # if x >= rj_max:
            #     m_tot_on_k[t] += m_last
            #     continue

            # Power-law extrapolation beyond species j's maximum radius
            if x >= rj_max:
                r0 = rj[N-1]; r1 = rj[N]
                m0 = mj[N-1]; m1 = mj[N]
                # use last log–log slope; fall back to constant if unsafe
                if r0 > 0.0 and m0 > 0.0 and m1 > 0.0 and r1 > r0:
                    a = (np.log(m1) - np.log(m0)) / (np.log(r1) - np.log(r0))
                    m_ext = m1 * np.exp(a * np.log(x / r1))
                else:
                    m_ext = m_last  # fallback for zeros/degeneracies
                m_tot_on_k[t] += m_ext
                continue

            # Locate j1 such that rj[j1] <= x < rj[j1+1]
            # Clamp so j1 >= 1 and j1 <= N-1, avoiding the origin in log space
            lo = 1
            hi = N  # invariant: search in [lo, hi)
            while lo < hi:
                mid = (lo + hi) // 2
                if rj[mid] <= x:
                    lo = mid + 1
                else:
                    hi = mid

            j1 = lo - 1
            if j1 < 1:
                j1 = 1
            elif j1 > N - 1:
                j1 = N - 1

            r0 = rj[j1]
            r1 = rj[j1 + 1]
            m0 = mj[j1]
            m1 = mj[j1 + 1]

            # Interior piecewise power-law interpolation in log-log space
            # Fall back to constant if logs would be unsafe
            if r0 > 0.0 and m0 > 0.0 and m1 > 0.0 and r1 > r0:
                a = (np.log(m1) - np.log(m0)) / (np.log(r1) - np.log(r0))
                m_interp = m0 * np.exp(a * np.log(x / r0))
            else:
                m_interp = m0

            m_tot_on_k[t] += m_interp

    m_tot_on_k[0] = 0.0

@njit(void(int64, float64[:, :], float64[:, :], float64[:], float64[:]), fastmath=True, cache=True)
def interp_m_enc_and_K(k, r, m, m_tot_on_k, K_on_k):
    """
    Fill m_tot_on_k with total enclosed mass evaluated on species-k radial grid,
    using piecewise power-law (log-log) interpolation for other species and
    power-law extrapolation beyond their maximum radius.

    Fill K_on_k with dM_other/dr evaluated on the same grid. The derivative K_on_k 
    includes only the non-Lagrangian contribution from other species interpolated 
    onto species k's grid. Species k's own enclosed mass is copied into m_tot_on_k 
    but contributes nothing to K_on_k.

    Parameters
    ----------
    k : int
        Species index (0 <= k < s).
    r : (s, N+1) float64
        Edge radii per species.
    m : (s, N+1) float64
        Enclosed mass per species at edges.
    m_tot_on_k : (N+1,) float64
        Preallocated output buffer for total enclosed mass on species-k grid.
    K_on_k : (N+1,) float64
        Preallocated output buffer for dM_other/dr on species-k grid.
        Only interior values K_on_k[1:N] are physically used later.
    """
    s, Np1 = r.shape
    N = Np1 - 1

    rk = r[k]  # view (N+1,)

    # Start with species k's own contribution to m_enc
    for t in range(Np1):
        m_tot_on_k[t] = m[k, t]
        K_on_k[t] = 0.0         # Initialize this array
    
    m_tot_on_k[0]   = 0.0       # To ensure
    K_on_k[0]       = 0.0       # This will not be used

    # Accumulate contributions from other species
    for j in range(s):
        if j == k:
            continue

        rj = r[j]  # (N+1,)
        mj = m[j]  # (N+1,)

        m_last = mj[N]
        rj_max = rj[N]

        for t in range(1, Np1):
            x = rk[t]

            # Constant tail beyond species j's maximum radius
            # if x >= rj_max:
            #     m_tot_on_k[t] += m_last
            #     continue

            # Power-law extrapolation beyond species j's maximum radius
            if x >= rj_max:
                r0 = rj[N-1]; r1 = rj[N]
                m0 = mj[N-1]; m1 = mj[N]
                # use last log–log slope; fall back to constant if unsafe
                if r0 > 0.0 and m0 > 0.0 and m1 > 0.0 and r1 > r0 and x > 0.0:
                    a = (np.log(m1) - np.log(m0)) / (np.log(r1) - np.log(r0))   # Log slope
                    m_ext = m1 * np.exp(a * np.log(x / r1))
                    K_ext = a * m_ext / x                                       # Derivative of the power law
                else:
                    m_ext = m_last                                              # Fallback for zeros/degeneracies
                    K_ext = 0.0
                m_tot_on_k[t] += m_ext
                K_on_k[t] += K_ext
                continue

            # Locate j1 such that rj[j1] <= x < rj[j1+1]
            # Clamp so j1 >= 1 and j1 <= N-1, avoiding the origin in log space
            lo = 1
            hi = N  # Invariant: search in [lo, hi)
            while lo < hi:
                mid = (lo + hi) // 2
                if rj[mid] <= x:
                    lo = mid + 1
                else:
                    hi = mid

            j1 = lo - 1
            if j1 < 1:
                j1 = 1
            elif j1 > N - 1:
                j1 = N - 1

            r0 = rj[j1]
            r1 = rj[j1 + 1]
            m0 = mj[j1]
            m1 = mj[j1 + 1]

            # Interior piecewise power-law interpolation in log-log space
            # Fall back to constant if logs would be unsafe
            if r0 > 0.0 and m0 > 0.0 and m1 > 0.0 and r1 > r0 and x > 0.0:
                a = (np.log(m1) - np.log(m0)) / (np.log(r1) - np.log(r0))
                m_interp = m0 * np.exp(a * np.log(x / r0))
                K_interp = a * m_interp / x
            else:
                m_interp = m0
                K_interp = 0.0

            m_tot_on_k[t] += m_interp
            K_on_k[t] += K_interp

    K_on_k[N] = 0.0 # Not used