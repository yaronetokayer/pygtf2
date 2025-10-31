import numpy as np
from numba import njit, float64, types

@njit(types.void(
        float64[:], float64[:], float64[:], float64[:]
     ),
     cache=True, fastmath=True)
def _loglin_interp_clamped(x_log_src, y_src, x_log_tgt, y_out) -> None:
    """
    Linear interp in log-x, with constant extrapolation on both ends.
    x_log_src: (M,), strictly increasing (log10 of x)
    y_src:     (M,)
    x_log_tgt: (K,)
    y_out:     (K,) pre-allocated
    """
    M = x_log_src.size
    K = x_log_tgt.size

    # Two-pointer walk
    j = 0
    for i in range(K):
        xt = x_log_tgt[i]

        # Before leftmost: clamp
        if xt <= x_log_src[0]:
            y_out[i] = y_src[0]
            continue
        # After rightmost: clamp
        if xt >= x_log_src[M-1]:
            y_out[i] = y_src[M-1]
            continue

        # Advance j so that x_src[j] <= xt < x_src[j+1]
        while j+1 < M and x_log_src[j+1] <= xt:
            j += 1

        x0 = x_log_src[j]
        x1 = x_log_src[j+1]
        y0 = y_src[j]
        y1 = y_src[j+1]
        t  = (xt - x0) / (x1 - x0)
        y_out[i] = y0 + t * (y1 - y0)

@njit(types.void(
        float64[:], float64[:], float64[:],
        float64[:], float64[:], float64[:], float64[:]
     ),
     cache=True, fastmath=True)
def _realign_one_species(r_src_edges, rho_src, v2_src, r_hat_edges,
                         rho_hat_out, v2_hat_out, m_hat_out) -> None:
    """
    r_src_edges: (N+1,), rho_src,v2_src: (N,)
    r_hat_edges: (N+1,)
    Outputs filled in-place:
      rho_hat_out, v2_hat_out: (N,)
      m_hat_out: (N+1,)
    """
    N = r_src_edges.size - 1
    # midpoints
    rmid_src = 0.5 * (r_src_edges[1:] + r_src_edges[:-1])
    rmid_hat = 0.5 * (r_hat_edges[1:] + r_hat_edges[:-1])

    # log-x arrays
    x_src = np.log10(rmid_src)
    x_tgt = np.log10(rmid_hat)

    # interpolate rho, v2 (clamped constant extrapolation)
    _loglin_interp_clamped(x_src, rho_src, x_tgt, rho_hat_out)
    _loglin_interp_clamped(x_src, v2_src,  x_tgt, v2_hat_out)

    # recompute enclosed mass on the new edges
    m_hat_out[0] = 0.0
    for i in range(N):
        dr3 = r_hat_edges[i+1]**3 - r_hat_edges[i]**3
        m_hat_out[i+1] = m_hat_out[i] + (dr3 * rho_hat_out[i]) / 3.0

@njit(types.Tuple((float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:]))
      (float64[:,:], float64[:,:], float64[:,:]), cache=True, fastmath=True)
def realign(r, rho, v2) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Realign species on a common log-spaced edge grid.
    Follows Zhong & Shapiro (2025)
    Inputs:
      r   : (s, N+1), per-species edges after revirialization (misaligned)
      rho : (s, N)   per-species shell densities
      v2  : (s, N)   per-species shell velocity dispersion^2
    Returns:
      r_hat      : (N+1,)
      rho_hat    : (s, N)
      v2_hat     : (s, N)
      p_hat      : (s, N)
      m_hat      : (s, N+1)
      m_tot_hat  : (N+1,)
    """
    s, Np1 = r.shape
    N = Np1 - 1

    # Common log grid bounds
    r1_hat = r[:, 1].min()
    rN_hat = r[:, N].max()

    # Build edges: r_hat_1d[0]=0, r_hat_1d[1..N] log between r1_hat and rN_hat
    r_hat_1d = np.empty(N+1, dtype=np.float64)
    # if s == 1: # Single-species fast path: keep the original grid exactly
    #     r_hat_1d[:] = r[0]
    # else: # Multiple species
    r_hat_1d[0] = 0.0
    xlo = np.log10(r1_hat)
    xhi = np.log10(rN_hat)
    for j in range(N):
        r_hat_1d[j+1] = 10.0 ** (xlo + (xhi - xlo) * (j / (N - 1.0)))

    rho_hat = np.empty_like(rho)
    v2_hat  = np.empty_like(v2)
    p_hat   = np.empty_like(rho)
    m_hat   = np.empty_like(r)

    # Per-species pass
    for k in range(s):
        _realign_one_species(r[k], rho[k], v2[k], r_hat_1d,
                             rho_hat[k], v2_hat[k], m_hat[k])

    p_hat[:] = rho_hat * v2_hat

    # totals on the aligned grid
    m_tot_hat = m_hat.sum(axis=0)

    # Broadcast the common r_hat_1d to all species
    r_hat = np.broadcast_to(r_hat_1d, (s, N+1)).copy()

    return r_hat, rho_hat, v2_hat, p_hat, m_hat, m_tot_hat

@njit(types.Array(float64, 1, "C")(float64, float64, types.intp), fastmath=True, cache=True)
def geospace1d(start, stop, num):
    """
    Geometric spacing (1D) between start and stop inclusive, length=num.
    Equivalent to np.exp(np.linspace(np.log(start), np.log(stop), num)).
    """
    out = np.empty(num, dtype=np.float64)
    log_a = np.log(start)
    log_b = np.log(stop)
    if num == 1:
        out[0] = np.exp(log_a)  # degenerate case
        return out
    step = (log_b - log_a) / (num - 1)
    for i in range(num):
        out[i] = np.exp(log_a + step * i)
    return out

# Return type: (r_hat, rho_hat, u_hat, v2_hat, p_hat, m_hat_edges, m_tot_hat)
_ret_t = types.Tuple((
    float64[:, :],  # r_hat
    float64[:, :],  # rho_hat
    float64[:, :],  # u_hat
    float64[:, :],  # v2_hat
    float64[:, :],  # p_hat
    float64[:, :],  # m_hat_edges
    float64[:]      # m_tot_hat
))

@njit(_ret_t(float64[:, :], float64[:, :], float64[:, :]), fastmath=True, cache=True)
def realign_extensive(r, rho, v2):
    """
    Conservative realignment of species onto a common edge grid by remapping
    *extensives* (mass and thermal energy) via overlaps in q=r^3.

    Parameters
    ----------
    r   : (s, N+1) per-species edges (after revir; misaligned across species)
    rho : (s, N)   per-species shell densities
    v2  : (s, N)   per-species shell velocity-dispersion^2 (p/rho)

    Returns
    -------
    r_hat        : (s, N+1) common edges (row-broadcast of 1D grid)
    rho_hat      : (s, N)  per-species densities on r_hat
    u_hat        : (s, N)  per-species specific internal energy
    v2_hat       : (s, N)  per-species p/rho
    p_hat        : (s, N)  per-species pressure
    m_hat_edges  : (s, N+1) cumulative mass on r_hat
    m_tot_hat    : (N+1,)  total enclosed mass at edges (cumulative over species)
    """
    s, Np1 = r.shape
    N = Np1 - 1

    # --- Compute geometry (avoid np.geomspace/np.insert in nopython)
    # r_hat_min = r[:, 1].min(); r_hat_max = r[:, -1].max() but do via loops.
    r_hat_min = r[0, 1]
    r_hat_max = r[0, N]
    for k in range(s):
        if r[k, 1] < r_hat_min:
            r_hat_min = r[k, 1]
        if r[k, N] > r_hat_max:
            r_hat_max = r[k, N]

    r_hat_1d = np.empty(Np1, dtype=np.float64)
    r_hat_1d[0] = 0.0
    tmp = geospace1d(r_hat_min, r_hat_max, N)  # length N
    for j in range(N):
        r_hat_1d[j + 1] = tmp[j]

    # q-spaces
    q_old = r * r * r  # (s, N+1)
    q_hat = r_hat_1d * r_hat_1d * r_hat_1d  # (N+1,)
    V_hat = (q_hat[1:] - q_hat[:-1]) / 3.0  # (N,)

    # --- Overlap deposit
    m_hat = np.zeros((s, N), dtype=np.float64)
    E_hat = np.zeros((s, N), dtype=np.float64)

    for k in range(s):
        i = 0
        j = 0
        while i < N and j < N:
            left = q_old[k, i]
            if q_hat[j] > left:
                left = q_hat[j]
            right = q_old[k, i + 1]
            if q_hat[j + 1] < right:
                right = q_hat[j + 1]
            dq = right - left
            if dq > 0.0:
                dV = dq / 3.0
                m_hat[k, j] += rho[k, i] * dV
                E_hat[k, j] += rho[k, i] * 1.5 * v2[k, i] * dV

            # advance the interval that ends first
            if q_old[k, i + 1] <= q_hat[j + 1]:
                i += 1
            else:
                j += 1

    # --- Consistency rescale per species (manual isclose)
    # rtol=1e-12, atol=1e-14 to match original
    rtol = 1e-12
    atol = 1e-14
    for k in range(s):
        # m_in = sum(rho[k] * (q_old[k,1:]-q_old[k,:-1])/3)
        m_in = 0.0
        for i in range(N):
            m_in += rho[k, i] * (q_old[k, i + 1] - q_old[k, i]) / 3.0
        # m_out = np.sum(m_hat[k])
        m_out = 0.0
        for j in range(N):
            m_out += m_hat[k, j]

        if not (np.abs(m_in - m_out) <= atol + rtol * np.abs(m_in)):
            scale = 1.0
            if m_out != 0.0:
                scale = m_in / m_out
            # keep original behavior (rescale both mass and energy)
            for j in range(N):
                m_hat[k, j] *= scale
                E_hat[k, j] *= scale

    # --- Recover intensives
    rho_hat = np.empty_like(m_hat)
    u_hat   = np.empty_like(m_hat)
    v2_hat  = np.empty_like(m_hat)
    p_hat   = np.empty_like(m_hat)

    # guard against empty cells & compute outputs
    for k in range(s):
        for j in range(N):
            mj = m_hat[k, j]
            if mj <= 0.0:
                # match original error behavior
                raise RuntimeError("Bins with zero mass in realignment step")
            rho_hat[k, j] = mj / V_hat[j]
            uj = E_hat[k, j] / mj
            u_hat[k, j] = uj
            v2_hat[k, j] = 2.0 * uj / 3.0
            p_hat[k, j]  = 2.0 * rho_hat[k, j] * uj / 3.0

    # --- cumulative masses on edges and total
    m_hat_edges = np.zeros((s, Np1), dtype=np.float64)
    for k in range(s):
        c = 0.0
        for j in range(N):
            c += m_hat[k, j]
            m_hat_edges[k, j + 1] = c

    m_tot_hat = np.zeros(Np1, dtype=np.float64)
    for j in range(Np1):
        tot = 0.0
        for k in range(s):
            tot += m_hat_edges[k, j]
        m_tot_hat[j] = tot

    # broadcast r_hat_1d to (s, N+1) without np.broadcast_to
    r_hat = np.empty((s, Np1), dtype=np.float64)
    for k in range(s):
        for j in range(Np1):
            r_hat[k, j] = r_hat_1d[j]

    return r_hat, rho_hat, u_hat, v2_hat, p_hat, m_hat_edges, m_tot_hat

def realign_extensive_readable(r, rho, v2):
    """
    Conservative realignment of species onto a common edge grid by remapping
    *extensives* (mass and thermal energy) via overlaps in q=r^3.

    Parameters
    ----------
    r   : (s, N+1) per-species edges (after revir; misaligned across species)
    rho : (s, N)   per-species shell densities
    v2  : (s, N)   per-species shell velocity-dispersion^2 (p/rho)

    Returns
    -------
    r_hat      : (N+1,) common edges
    rho_hat    : (s, N)  per-species densities on r_hat
    u_hat      : (s, N)  per-species specific internal energy
    v2_hat     : (s, N)  per-species p/rho
    p_hat      : (s, N)  per-species pressure
    m_hat_edges : (s, N+1)  per-species cumulative mass on r_hat
    m_tot_hat  : (N+1,)  total enclosed mass at edges (cumulative over species)
    """
    s, Np1 = r.shape
    N = Np1 -1

    #--- Compute geometry
    r_hat_min = r[:, 1].min()
    r_hat_max = r[:, Np1-1].max()
    r_hat = np.geomspace(r_hat_min, r_hat_max, num=Np1-1)
    r_hat = np.insert(r_hat, 0, 0.0)

    q_old = r**3
    q_hat = r_hat**3

    V_hat = (q_hat[1:] - q_hat[:-1]) / 3.0

    #--- Overlap deposit
    # Compute overlap volumes for each species and shell
    # and update extensive quantities

    m_hat = np.zeros((s, N), dtype=np.float64)  # mass per cell, shape (s, N_hat)
    E_hat = np.zeros((s, N), dtype=np.float64)

    for k in range(s):
        i, j = 0, 0
        while i < N and j < N:
            left  = max(q_old[k, i],   q_hat[j])
            right = min(q_old[k, i+1], q_hat[j+1])
            dq = right - left
            if dq > 0.0:
                dV = dq / 3.0
                m_hat[k, j] += rho[k, i] * dV
                E_hat[k, j] += rho[k, i] * 1.5 * v2[k, i] * dV

            # advance the interval that ends first
            if q_old[k, i+1] <= q_hat[j+1]:
                i += 1
            else:
                j += 1
    
    #--- Check for consistency
    # For each species, compare input and output total mass, rescale if needed
    for k in range(s):
        m_in  = np.sum(rho[k] * (q_old[k,1:] - q_old[k,:-1]) / 3.0)
        m_out = np.sum(m_hat[k])
        if not np.isclose(m_in, m_out, rtol=1e-12, atol=1e-14):
            scale = m_in / m_out if m_out != 0 else 1.0
            print("WARNING: rescaling masses in realign step")
            m_hat[k] *= scale
            E_hat[k] *= scale

    #--- Recover intensives
    rho_hat = np.zeros_like(m_hat)
    u_hat   = np.zeros_like(m_hat)
    v2_hat  = np.zeros_like(m_hat)
    p_hat   = np.zeros_like(m_hat)
    
    # guard against empty cells
    if np.any(m_hat <= 0.0):
        raise RuntimeError("Bins with zero mass in realignment step")
    rho_hat = m_hat / V_hat[np.newaxis, :]
    u_hat   = E_hat / m_hat
    v2_hat  = 2.0 * u_hat / 3.0
    p_hat   = 2.0 * rho_hat * u_hat / 3.0

    # total enclosed mass at edges (sum species, cumulative)
    # Cumulative mass profile per species (on edges)
    m_hat_edges = np.zeros((s, Np1), dtype=np.float64)
    m_hat_edges[:, 1:] = np.cumsum(m_hat, axis=1)
    # Total cumulative mass profile (sum over species)
    m_tot_hat = np.sum(m_hat_edges, axis=0)

    r_hat = np.broadcast_to(r_hat, (s, Np1)).copy()

    return r_hat, rho_hat, u_hat, v2_hat, p_hat, m_hat_edges, m_tot_hat
