import numpy as np
from numba import njit, float64

### UNUSED METHODS ###

def revirialize_OLD(r, rho, p, m_tot) -> tuple[str, np.ndarray | None, np.ndarray | None, np.ndarray | None, float | None]:
    """
    Multi-species re-virialization.
    Solves for radius adjustments and updates physical quantities for all species.
    Assumes all species have aligned radial bins.

    Parameters
    ----------
    r : ndarray, shape (s, N+1)
        Edge radii per species.
    rho : ndarray, shape (s, N)
        Shell densities per species.
    p : ndarray, shape (s, N)
        Shell pressures per species.
    m_tot : ndarray, shape (N+1,)
        Total enclosed mass at edges (shared across species), computed on the
        aligned grid before calling this routine.

    Returns
    -------
    status : str
        'ok' if successful, 'shell_crossing' if any radii cross.
    r_new : ndarray or None, shape (s, N+1)
        Updated edge radii per species, or None if shell crossing.
    rho_new : ndarray or None, shape (s, N)
        Updated shell densities per species, or None if shell crossing.
    p_new : ndarray or None, shape (s, N)
        Updated shell pressures per species, or None if shell crossing.
    dr_max : float or None
        Global maximum |dr/r| across all species, or None if shell crossing.

    Notes
    -----
    This function solves a tridiagonal system to compute radius corrections for each species,
    then updates density and pressure accordingly. If any radii cross, the function returns
    'shell_crossing' and None for all outputs except status.
    """
    s, _ = r.shape
    r_new   = np.empty_like(r)
    rho_new = np.empty_like(rho)
    p_new   = np.empty_like(p)
    dr_max = 0.0

    for k in range(s):
        a, b, c, y = build_tridiag_system(r[k], rho[k], p[k], m_tot)
        # print(np.max(np.abs(y)), np.min(np.abs(y)))
        xk = solve_tridiagonal_frank(a, b, c, y)
        # xk *= alpha
        rk, pk, rhok = _update_r_p_rho(r[k], xk, p[k], rho[k])
        r_new[k]   = rk
        p_new[k]   = pk
        rho_new[k] = rhok
        dr_max = max(dr_max, float(np.max(np.abs(xk))))
    if np.any((r_new[:,1:] - r_new[:,:-1]) <= 0.0):
        return 'shell_crossing', r_new, None, None, dr_max

    return 'ok', r_new, rho_new, p_new, dr_max

def revirialize_drift_damp(r, rho, p, m_tot) -> tuple[str, np.ndarray | None, np.ndarray | None, np.ndarray | None, float | None]:
    """
    Multi-species re-virialization.
    Solves for radius adjustments and updates physical quantities for all species.
    Assumes all species have aligned radial bins.
    Uses a damping factor based on the relative change between species

    Parameters
    ----------
    r : ndarray, shape (s, N+1)
        Edge radii per species.
    rho : ndarray, shape (s, N)
        Shell densities per species.
    p : ndarray, shape (s, N)
        Shell pressures per species.
    m_tot : ndarray, shape (N+1,)
        Total enclosed mass at edges (shared across species), computed on the
        aligned grid before calling this routine.

    Returns
    -------
    status : str
        'ok' if successful, 'shell_crossing' if any radii cross.
    r_new : ndarray or None, shape (s, N+1)
        Updated edge radii per species, or None if shell crossing.
    rho_new : ndarray or None, shape (s, N)
        Updated shell densities per species, or None if shell crossing.
    p_new : ndarray or None, shape (s, N)
        Updated shell pressures per species, or None if shell crossing.
    dr_max : float or None
        Global maximum |dr/r| across all species, or None if shell crossing.

    Notes
    -----
    This function solves a tridiagonal system to compute radius corrections for each species,
    then updates density and pressure accordingly. If any radii cross, the function returns
    'shell_crossing' and None for all outputs except status.
    """
    s, n = r.shape
    r_new   = np.empty_like(r)
    rho_new = np.empty_like(rho)
    p_new   = np.empty_like(p)
    x = np.empty((s, n - 2), dtype=np.float64)

    for k in range(s):
        a, b, c, y = build_tridiag_system_log(r[k], rho[k], p[k], m_tot)
        ab = np.zeros((3, b.size), dtype=float)
        ab[0, 1:]   = c[:-1]        # c[-1] is outside of the matrix
        ab[1, :]    = b
        ab[2, :-1]  = a[1:]         # a[0] is outside of the matrix
        print(n, y.shape)
        xk = solve_banded((1,1), ab, y)
        # xk = solve_tridiagonal_frank(a, b, c, y)
        x[k] = xk

    # alpha = under_relax(x)            # Hill function based on relative difference between the two grids
    alpha = 1.0
    dr_max = float(np.max(np.abs(x)))
    # print(d, alpha, dr_max)

    for k in range(s):
        xk = alpha * x[k]
        rk, pk, rhok = _update_r_p_rho(r[k], xk, p[k], rho[k])
        r_new[k]   = rk
        p_new[k]   = pk
        rho_new[k] = rhok

    if np.any((r_new[:,1:] - r_new[:,:-1]) <= 0.0):
        return 'shell_crossing', r_new, None, None, dr_max

    return 'ok', r_new, rho_new, p_new, dr_max

def revirialize_res_damp(r, rho, p, m_tot, frac) -> tuple[str, np.ndarray | None, np.ndarray | None, np.ndarray | None, float | None]:
    """
    Multi-species re-virialization.
    Solves for radius adjustments and updates physical quantities for all species.
    Assumes all species have aligned radial bins.
    Uses a damping factor based on the residual norm.

    Parameters
    ----------
    r : ndarray, shape (s, N+1)
        Edge radii per species.
    rho : ndarray, shape (s, N)
        Shell densities per species.
    p : ndarray, shape (s, N)
        Shell pressures per species.
    m_tot : ndarray, shape (N+1,)
        Total enclosed mass at edges (shared across species), computed on the
        aligned grid before calling this routine.

    Returns
    -------
    status : str
        'ok' if successful, 'shell_crossing' if any radii cross.
    r_new : ndarray or None, shape (s, N+1)
        Updated edge radii per species, or None if shell crossing.
    rho_new : ndarray or None, shape (s, N)
        Updated shell densities per species, or None if shell crossing.
    p_new : ndarray or None, shape (s, N)
        Updated shell pressures per species, or None if shell crossing.
    dr_max : float or None
        Global maximum |dr/r| across all species, or None if shell crossing.

    Notes
    -----
    This function solves a tridiagonal system to compute radius corrections for each species,
    then updates density and pressure accordingly. If any radii cross, the function returns
    'shell_crossing' and None for all outputs except status.
    """
    s, n = r.shape
    print(n)
    r_new   = np.empty_like(r)
    rho_new = np.empty_like(rho)
    p_new   = np.empty_like(p)
    x = np.empty((s, n - 2), dtype=np.float64)

    tau = 0.0
    # Outer loop: controls strict diagonal dominance with tau
    while True:
        y_combined = np.zeros(s*(n-2), dtype=np.float64)
        for k in range(s):
            a, b, c, y = build_tridiag_system_log(r[k], rho[k], p[k], m_tot)
            b += tau * (1.0 + np.abs(a) + np.abs(b) + np.abs(c))
            y_combined[k*(n-2):(k+1)*(n-2)] = y[:]
            ab = np.zeros((3, b.size), dtype=float)
            ab[0, 1:]   = c[:-1]        # c[-1] is outside of the matrix
            ab[1, :]    = b
            ab[2, :-1]  = a[1:]         # a[0] is outside of the matrix
            xk = solve_banded((1,1), ab, y)
            # xk = solve_tridiagonal_frank(a, b, c, y)
            x[k] = xk

            # Check diagonal dominance
            # Finding that it's not...
            dom = 1e100
            for i in range(n-2):
                if i == 0:
                    check = b[i] - abs(c[i])
                elif i == n-3:
                    check = b[i] - abs(a[i])
                else:
                    check - b[i] - abs(a[i]) - abs(c[i])
                if check < dom:
                    dom = check
            print(dom)
        res_old = resid_norm(y_combined, m_tot, r, rho, frac)
        dr_max = float(np.max(np.abs(x)))

        alpha = 1.0
        # Inner loop: controls damping of the correction with alpha
        while True:
            y_combined = np.zeros(s*(n-2), dtype=np.float64)
            for k in range(s):
                xk = alpha * x[k]
                rk, pk, rhok = _update_r_p_rho(r[k], xk, p[k], rho[k])
                r_new[k]   = rk
                p_new[k]   = pk
                rho_new[k] = rhok
                y_combined[k*(n-2):(k+1)*(n-2)] = compute_res_log(rk, rhok, pk, m_tot)
            res_new = resid_norm(y_combined, m_tot, r_new, rho_new, frac)
            if res_new < res_old:
                print(f"got it, alpha={alpha}, tau={tau}")
                # exit both loops by returning (perform shell-crossing guard first)
                if np.any((r_new[:,1:] - r_new[:,:-1]) <= 0.0):
                    return 'shell_crossing', r_new, None, None, dr_max
                return 'ok', r_new, rho_new, p_new, dr_max
            if alpha <= 1e-3:   # Increase tau and go to top of outer loop
                print("alpha too small! increasing tau")
                if tau == 0.0:
                    tau = 1e-3
                elif tau > 1e6:
                    print('tau too big!')
                    if np.any((r_new[:,1:] - r_new[:,:-1]) <= 0.0):
                        return 'shell_crossing', r_new, None, None, dr_max
                    return 'ok', r_new, rho_new, p_new, dr_max
                else:
                    tau *= 10.0
                break
            alpha *= 0.5

    if np.any((r_new[:,1:] - r_new[:,:-1]) <= 0.0):
        return 'shell_crossing', r_new, None, None, dr_max

    return 'ok', r_new, rho_new, p_new, dr_max

def resid_norm(y_combined, m_tot, r, rho, frac) -> float:
    """
    L2 norm of stacked HE residuals on edges (interior edges only),
    scaled by rho_edge * |g_edge| and weighted by sqrt(mass fraction).
    
    y_combined: shape (s*(n-1),) if you include all interior edges per species.
    m_tot     : shape (n+1,), total enclosed mass at edges (m_tot[0]=0).
    r         : shape (s, n+1), per-species edge radii (r[:,0]=0).
    rho       : shape (s, n),   per-species cell-centered densities.
    frac      : shape (s,),     species mass fractions (sum≈1).
    """
    import numpy as np
    s, np1 = r.shape
    n = np1 - 1
    assert y_combined.shape[0] == s*(n-1)  # edges 1..n-1 per species

    floor = 1e-300  # practical floor to keep scaling finite
    scaling = np.empty_like(y_combined, dtype=np.float64)

    # Edge-centered gravity from total mass & species edge radii
    # (assumes m_tot corresponds to each species' edges; if not, interp m_tot to r[k])
    off = 0
    for k in range(s):
        r_edge = r[k, 1:-1]                 # edges 1..n-1, length n-1
        M_edge = m_tot[1:-1]                # same indices
        g_edge = M_edge / np.maximum(r_edge*r_edge, floor)  # |g| at edges

        # Edge-centered density = avg of adjacent cell densities
        # rho_edge[i] aligns with edge i (between cells i-1 and i)
        rho_edge = 0.5*(rho[k, :-1] + rho[k, 1:])          # length n-1

        denom = np.maximum(np.abs(g_edge)*rho_edge, floor)

        w = np.sqrt(frac[k])                # species mass-fraction weighting
        scaling[off:off+(n-1)] = w / denom
        off += (n-1)

    y_scaled = y_combined * scaling
    return float(np.linalg.norm(y_scaled))

@njit(float64[:]
      (float64[:], float64[:], float64[:], float64[:]), cache=True, fastmath=False)
def compute_res_log(r, rho, p, m_tot) -> tuple[np.ndarray]:
    """
    Compute the residual (y) of the tridiagonal system A·x = y for interior radial corrections.
    These are the coefficients for the log form of the HE equation.

    Arguments
    ---------
    r : ndarray
        Radial edge coordinates, length = n + 1.
    rho : ndarray
        Shell-centered densities, length = n.
    p : ndarray
        Shell-centered pressures, length = n.
    m_tot : ndarray
        Total enclosed mass at edge points, length = n + 1.

    Returns
    -------
    y : ndarray
    """

    lnr = np.empty_like(r)
    lnr[1:] = np.log(r[1:])             # Don't take ln0 - lnr[0] never used anyway
    lnr[0]  = lnr[1]                    # Arbitrary finite placeholder
    dlnr = 0.5 * ( lnr[2:] - lnr[:-2] ) # Central difference

    lnp = np.log(p)
    dlnp = lnp[1:] - lnp[:-1]           # Right-sided difference

    sr = rho[:-1] + rho[1:]
    sp = p[:-1] + p[1:]

    mr = m_tot[1:-1] / r[1:-1]

    # floors to avoid divide-by-zero/inf
    tiny = np.finfo(np.float64).tiny
    sp   = np.where(np.abs(sp)   < tiny, np.copysign(tiny, sp),   sp)
    dlnr   = np.where(np.abs(dlnr)   < tiny, np.copysign(tiny, dlnr),   dlnr)

    dpdr = 0.5 * dlnp / dlnr**2
    srsp = sr / sp

    # Terms in final expressions
    dfac = 2.0 * dpdr * dlnr

    y = -mr * srsp - dfac

    # Enforce dp/dr = 0 for i=1
    y[0] = -dlnp[0]

    return y

@njit(float64 (float64[:, :]), fastmath=True, cache=True)
def max_frac_diff(r) -> float:
    """
    Computes the maximum absolute fractional difference between
    the r arrays of any pair of species
    """
    n, m = r.shape
    max_frac = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(m):
                avg = 0.5 * (r[i, k] + r[j, k])
                if avg != 0.0:
                    frac_diff = abs(r[i, k] - r[j, k]) / avg
                else:
                    frac_diff = 0.0
                if frac_diff > max_frac:
                    max_frac = frac_diff
    return max_frac

@njit(float64(float64[:, :]), fastmath=True, cache=True)
def under_relax(x) -> float:
    """
    Computes the maximum absolute difference between the
    proposed fractional displacements (Δr/r) across species.
    Then computes a damping factor for the corrections based on a Hill function.
    """
    z = 1.0     # Sharpness
    d0 = 1e-5   # Reference value

    # Compute the maximum fractional difference bewteen species
    n, m = x.shape
    max_diff = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(m):
                diff = abs(x[i, k] - x[j, k])
                if diff > max_diff:
                    max_diff = diff

    alpha = 1.0 / (1 + (max_diff/d0)**z)

    return alpha
