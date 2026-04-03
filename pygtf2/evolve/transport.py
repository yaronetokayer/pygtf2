import numpy as np 
from numba import njit, float64, types, void
from pygtf2.util.interpolate import interp_linear_to_interfaces
from pygtf2.util.calc import solve_tridiagonal_thomas

@njit(
    float64[:, :](float64, float64[:, :], float64[:, :], float64[:, :], float64[:], float64[:,:]),
    cache=True, fastmath=True
)
def compute_luminosities(c2, r, u, rho, mrat, lnL) -> np.ndarray:
    """ 
    Compute luminosity of each shell interface based on temperature gradient and conductivity.
    e.g, Eq. (43) in Zhong and Shapiro (2025).

    Arguments
    ----------
    c2 : float
        Constant 'c2' in the luminosity formula.
    r : ndarray
        Radial grid points, including cell edges.
    u : ndarray
        Specific internal energy for each cell.
    rho : ndarray
        Density for each cell.
    mrat : ndarray
        Mass ratio of species (length = s)
    lnL : ndarray
        lnL array.

    Returns
    -------
    L: ndarray
        Luminosities at each shell boundary (same length as r).
    """
    s, Np1 = r.shape
    N = Np1 - 1
    L = np.zeros((s, Np1), np.float64) # Initialization takes care of boundary conditions

    for n in range(s):
        r_n     = r[n]          # (N+1,)
        u_n     = u[n]          # (N,)
        rho_n   = rho[n]        # (N,)

        # Centered interface values
        rhom = interp_linear_to_interfaces(r_n, rho_n)
        umed = interp_linear_to_interfaces(r_n, u_n)

        # dT/dr at interfaces
        dTdr = np.empty(N-1, dtype=np.float64)
        dTdr[0] = (u_n[1] - u_n[0]) / (r_n[2] - r_n[1])
        dTdr[1:] = 2.0 * (u_n[2:] - u_n[1:-1]) / (r_n[3:] - r_n[1:-2])

        # Prefactors
        fac = (r[n, 1:-1]**2) * (rhom / np.sqrt(umed))
        pref = (-c2) * (mrat[n] * lnL[n,n])

        L[n, 1:-1] = pref * fac * dTdr

    return L

@njit(types.Tuple((float64[:, :], float64[:, :], float64, float64,))
      (float64[:, :], float64[:, :], float64[:, :], float64[:, :],
        float64[:, :], float64[:], float64[:, :], float64, float64, float64),
    cache=True)
def conduct_heat(m, u, rho, lum, lnL, mrat, r, dt_prop, eps_du, c1) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Conduct heat and update internal energies (u) and pressures (p) for all species.

    This routine applies two contributions to du/dt:
      - Intra-species conduction: flux divergence of interface luminosities (lum).
      - Inter-species heat exchange: volume-overlap weighted exchange using lnL and mrat.

    PdV work is ignored and density is held fixed. An adaptive limiter is applied
    to ensure the maximum relative change in u does not exceed eps_du; if needed the
    proposed dt_prop is scaled down and dt_eff returned.

    Notes
    -----
    - Luminosities `lum` are defined at cell interfaces (length N+1 per species).
    - Pressure is recomputed as p = (2/3) * rho * u_new.
    - The function returns the updated pressure array, the observed maximum relative
      change in u (dumax), and the effective timestep used (dt_eff). The updated
      internal energy array (u_new) is not returned.

    Parameters
    ----------
    m : ndarray, shape (s, N+1)
        Enclosed mass at cell interfaces for each species.
    u : ndarray, shape (s, N)
        Specific internal energy per cell for each species.
    rho : ndarray, shape (s, N)
        Density per cell for each species (assumed fixed here).
    lum : ndarray, shape (s, N+1)
        Luminosity at cell interfaces for each species (used to compute flux divergence).
    lnL : ndarray, shape (s, s)
        Inter-species coupling coefficients (matrix of lnL[n,k]).
    mrat : ndarray, shape (s,)
        Mass-ratio-like coefficient for each species.
    r : ndarray, shape (s, N+1)
        Radial coordinates including cell interfaces for each species.
    dt_prop : float
        Proposed timestep to apply for this update.
    eps_du : float
        Maximum allowed relative change in u (used by adaptive limiter).
    c1 : float
        Multiplicative constant for inter-species exchange term.

    Returns
    -------
    p_new : ndarray, shape (s, N)
        Updated pressure array after applying du and recomputing p = (2/3) * rho * u_new.
    v2_new : ndarray, shape (s, N)
        Updated v2 array after applying du and recomputing v2 = (2/3) u_new.
    dumax : float
        Observed maximum relative change in u (after any scaling).
    dt_eff : float
        Effective timestep actually used (dt_prop scaled by limiter if needed).
    """

    s, Np1 = m.shape
    N = Np1 - 1

    # Outputs
    u_new   = np.empty_like(u)
    v2_new  = np.empty_like(u)
    p_new   = np.empty_like(u)  # same (s, N) shape
    du      = np.empty_like(u)

    # ---------- intra-species conduction (flux divergence in mass coord) ----------
    # dudt_cond[n,i] = - (L[n,i+1]-L[n,i]) / (M[n,i+1]-M[n,i])
    dudt_cond = np.empty_like(u)
    for n in range(s):
        for i in range(N):
            dm = m[n, i+1] - m[n, i]
            dudt_cond[n, i] = - (lum[n, i+1] - lum[n, i]) / dm

    # ---------- inter-species heat exchange with volume-overlap weighting ----------
    # For each (n,i), sweep all species k!=n and all their cells j
    #   overlap [max(r_n_i, r_k_j), min(r_n_ip1, r_k_jp1)]
    #   VolRat = overlap_volume / vol(cell n,i)
    #   contribution ~ lnL[n,k] * rho[k,j] * VolRat * (mrat[k]*u[k,j] - mrat[n]*u[n,i]) / (u[k,j]+u[n,i])^(3/2)
    dudt_hex = np.zeros_like(u)

    for n in range(s):
        mrn = mrat[n]

        # precompute r^3 arrays for n and reuse
        rn3 = np.empty(N+1, dtype=np.float64)
        for i in range(N+1):
            rr = r[n, i]
            rn3[i] = rr*rr*rr

        for k in range(s):
            if k == n:
                continue
            lnL_nk = lnL[n, k]
            mrk    = mrat[k]

            # precompute r^3 for species k too
            rk3 = np.empty(N+1, dtype=np.float64)
            for j in range(N+1):
                rr = r[k, j]
                rk3[j] = rr*rr*rr

            # two-pointer sweep over i (species n) and j (species k)
            i = 0
            j = 0
            while i < N and j < N:
                rn_lo = r[n, i]
                rn_hi = r[n, i+1]
                rk_lo = r[k, j]
                rk_hi = r[k, j+1]

                # compute overlap interval
                # if no overlap, advance the pointer with the smaller upper bound
                if rk_hi <= rn_lo:
                    j += 1
                    continue
                if rn_hi <= rk_lo:
                    i += 1
                    continue

                # they overlap: [max(lo), min(hi)]
                # rrmin = max(rn_lo, rk_lo), rrmax = min(rn_hi, rk_hi)
                rrmin = rn_lo if rn_lo >= rk_lo else rk_lo
                rrmax = rn_hi if rn_hi <= rk_hi else rk_hi

                rrmin3 = rrmin*rrmin*rrmin
                rrmax3 = rrmax*rrmax*rrmax

                # shell volume of (n,i) in r^3 units, and overlap volume
                vol_n_r3 = rn3[i+1] - rn3[i]
                if vol_n_r3 > 0.0:
                    dvol_overlap_r3 = rrmax3 - rrmin3
                    if dvol_overlap_r3 > 0.0:
                        vol_ratio = dvol_overlap_r3 / vol_n_r3

                        un = u[n, i]
                        uj = u[k, j]
                        denom_uj_un = uj + un
                        if denom_uj_un > 0.0:
                            root = np.sqrt(denom_uj_un)
                            inv_p32 = 1.0 / (denom_uj_un * root)
                            term = lnL_nk * rho[k, j] * vol_ratio * (mrk * uj - mrn * un) * inv_p32
                            dudt_hex[n, i] += term
                # advance the pointer whose cell ends first
                if rn_hi <= rk_hi:
                    i += 1
                else:
                    j += 1

    dudt_hex *= c1

    # ---------- combine and propose update ----------
    dudt = dudt_cond + dudt_hex
    du[:, :] = dudt * dt_prop

    # ---------- adaptive limiter on max relative change ----------
    floor = 1e-40
    dumax = 0.0
    for n in range(s):
        for i in range(N):
            denom = abs(u[n, i])
            if denom < floor:
                denom = floor
            rat = abs(du[n, i]) / denom
            if rat > dumax:
                dumax = rat

    if dumax > eps_du:
        scale = 0.95 * (eps_du / dumax)
        du *= scale
        dumax *= scale
        dt_eff = dt_prop * scale
    else:
        dt_eff = dt_prop

    # ---------- update u and p ----------
    for n in range(s):
        for i in range(N):
            u_new[n, i] = u[n, i] + du[n, i]
            v2_new[n, i] = (2.0 / 3.0) * u_new[n, i]
            p_new[n, i] = (2.0 / 3.0) * rho[n, i] * u_new[n, i]

    return p_new, v2_new, float(dumax), float(dt_eff)

### NEW IMPLICIT METHOD

@njit(
    types.Tuple((float64, float64))(
        float64[:, :],   # u        (in/out)
        float64[:, :],   # rho
        float64[:, :],   # lnL
        float64[:],      # mrat
        float64[:, :],   # r
        float64[:, :],   # du_work
        float64,         # dt_prop
        float64,         # eps_du
        float64          # c1
    ),
    cache=True
)
def heat_exchange(u, rho, lnL, mrat, r, du_work, dt_prop, eps_du, c1):
    """
    Apply only the inter-species heat exchange step, updating u in place.

    This routine:
      1. Computes the inter-species heat exchange contribution only
         (no intra-species conduction).
      2. Forms du = dudt_hex * dt_prop into du_work.
      3. Applies the adaptive limiter so that max(|du/u|) <= eps_du
         up to the 0.95 safety factor.
      4. Updates u in place.
      5. Returns (du_max, dt_eff).

    Parameters
    ----------
    u : ndarray, shape (s, N)
        Specific internal energy per cell for each species.
        Updated in place.
    rho : ndarray, shape (s, N)
        Density per cell for each species.
    lnL : ndarray, shape (s, s)
        Inter-species coupling coefficients.
    mrat : ndarray, shape (s,)
        Mass-ratio-like coefficient for each species.
    r : ndarray, shape (s, N+1)
        Radial cell interfaces for each species.
    du_work : ndarray, shape (s, N)
        Workspace array used to store du before updating u.
    dt_prop : float
        Proposed timestep.
    eps_du : float
        Maximum allowed relative change in u.
    c1 : float
        Multiplicative constant for inter-species exchange term.

    Returns
    -------
    du_max : float
        Realized maximum relative change in u after any limiter scaling.
    dt_eff : float
        Effective timestep after any limiter scaling.
    """

    s, N = u.shape

    # Zero workspace: this will hold du = dudt_hex * dt_prop
    for n in range(s):
        for i in range(N):
            du_work[n, i] = 0.0

    # ---------- inter-species heat exchange only ----------
    for n in range(s):
        mrn = mrat[n]

        # precompute r^3 for species n
        rn3 = np.empty(N + 1, dtype=np.float64)
        for i in range(N + 1):
            rr = r[n, i]
            rn3[i] = rr * rr * rr

        for k in range(s):
            if k == n:
                continue

            lnL_nk = lnL[n, k]
            mrk = mrat[k]

            # precompute r^3 for species k
            rk3 = np.empty(N + 1, dtype=np.float64)
            for j in range(N + 1):
                rr = r[k, j]
                rk3[j] = rr * rr * rr

            # two-pointer overlap sweep
            i = 0
            j = 0
            while i < N and j < N:
                rn_lo = r[n, i]
                rn_hi = r[n, i + 1]
                rk_lo = r[k, j]
                rk_hi = r[k, j + 1]

                if rk_hi <= rn_lo:
                    j += 1
                    continue
                if rn_hi <= rk_lo:
                    i += 1
                    continue

                rrmin = rn_lo if rn_lo >= rk_lo else rk_lo
                rrmax = rn_hi if rn_hi <= rk_hi else rk_hi

                rrmin3 = rrmin * rrmin * rrmin
                rrmax3 = rrmax * rrmax * rrmax

                vol_n_r3 = rn3[i + 1] - rn3[i]
                if vol_n_r3 > 0.0:
                    dvol_overlap_r3 = rrmax3 - rrmin3
                    if dvol_overlap_r3 > 0.0:
                        vol_ratio = dvol_overlap_r3 / vol_n_r3

                        un = u[n, i]
                        uk = u[k, j]
                        denom = uk + un
                        if denom > 0.0:
                            root = np.sqrt(denom)
                            inv_p32 = 1.0 / (denom * root)
                            du_work[n, i] += (
                                c1
                                * lnL_nk
                                * rho[k, j]
                                * vol_ratio
                                * (mrk * uk - mrn * un)
                                * inv_p32
                                * dt_prop
                            )

                if rn_hi <= rk_hi:
                    i += 1
                else:
                    j += 1

    # ---------- adaptive limiter on max relative change ----------
    floor = 1e-40
    du_max = 0.0

    for n in range(s):
        for i in range(N):
            denom = abs(u[n, i])
            if denom < floor:
                denom = floor
            rat = abs(du_work[n, i]) / denom
            if rat > du_max:
                du_max = rat

    dt_eff = dt_prop
    if du_max > eps_du:
        scale = 0.95 * (eps_du / du_max)
        for n in range(s):
            for i in range(N):
                du_work[n, i] *= scale
        du_max *= scale
        dt_eff = dt_prop * scale

    # ---------- update u in place ----------
    for n in range(s):
        for i in range(N):
            u[n, i] += du_work[n, i]

    return float(du_max), float(dt_eff)

@njit(
    void(
        float64[:],  # a
        float64[:],  # b
        float64[:],  # c
        float64[:],  # d
        float64[:],  # rk
        float64[:],  # mk
        float64[:],  # rhok_int
        float64[:],  # uk
        float64,     # pref
        float64      # dt
    ),
    cache=True,
    fastmath=True
)
def build_tridiag_system(a, b, c, d, rk, mk, rhok_int, uk, pref, dt):
    """
    Construct tridiagonal coefficients: a_i du_i-1 + b_i du_i + c_i du_i+1 = d_i
    a, b, c, and d are updated in place

    Arguments
    ---------
    a : ndarray, shape (N,)
        Subdiagonal coefficients (multiply du_{i-1}) for interior nodes j=1..N.
    b : ndarray, shape (N,)
        Main diagonal coefficients (multiply du_i) for interior nodes.
    c : ndarray, shape (N,)
        Superdiagonal coefficients (multiply d_{u+1}) for interior nodes.
    d : ndarray, shape (N,)
        Right-hand side vector for the interior nodes.
    rk : ndarray, shape (N+1,)
        Edge radii.
    mk : ndarray, shape (N+1,)
        Enclosed mass at edges.
    rhok_int : ndarray, shape (N-1,)
        Densities interpolated to shell edges
    uk : ndarray, shape (N,)
        Specific internal energy.
    pref : float
        prefactor for species k
    dt : float
        timestep
    """
    drc     = 0.5 * (rk[2:] - rk[:-2])      # (N-1,)
    delu    = uk[1:] - uk[:-1]              # (N-1,)
    su      = uk[1:] + uk[:-1]              # (N-1,)
    sqrt2   = 1.41421356237309

    # Interior cells
    facL    = rhok_int[:-1] * rk[1:-2]**2 / drc[:-1]
    facR    = rhok_int[1:] * rk[2:-1]**2 / drc[1:]
    su12L   = 1 / np.sqrt(su[:-1])
    su12R   = 1 / np.sqrt(su[1:])
    dusu32L = 0.5 * delu[:-1] / su[:-1]**(3.0/2.0)
    dusu32R = 0.5 * delu[1:] / su[1:]**(3.0/2.0)

    a[1:-1] = facL * ( su12L + dusu32L )
    b[1:-1] = -1 * (
        facR * ( su12R + dusu32R )
        + facL * ( su12L - dusu32L )
        + ( ( mk[2:-1] - mk[1:-2] ) / ( sqrt2 * pref * dt ) )
    )
    c[1:-1] = facR * ( su12R - dusu32R )
    d[1:-1] = (
        facL * delu[:-1] / np.sqrt(su[:-1])
        - facR * delu[1:] / np.sqrt(su[1:])
    )

    # i = 1
    a[0] = 0.0
    b[0] = -1 *  (
        su12L[0] + dusu32L[0]
        + ( mk[1] * drc[0] / ( rhok_int[0] * rk[1]**2 * pref * sqrt2 * dt ) )
    )
    c[0] = su12L[0] - dusu32L[0]
    d[0] = - delu[0] / np.sqrt(su[0])

    # i = N
    a[-1] = su12R[-1] + dusu32R[-1]
    b[-1] = (
        dusu32R[-1] - su12R[-1]
        - ( 
            (mk[-1] - mk[-2]) * drc[-1] 
            / ( rhok_int[-1] * rk[-2]**2 * pref * sqrt2 * dt )
        )
    )
    c[-1] = 0.0
    d[-1] = delu[-1] / np.sqrt(su[-1])

# @njit(
#     types.Tuple((float64, float64))(
#         float64[:, :],  # u
#         float64[:, :],  # rho
#         float64[:, :],  # r
#         float64[:, :],  # m
#         float64,        # c2
#         float64[:],     # mrat
#         float64[:, :],  # lnL
#         float64,        # dt
#         float64         # eps_du
#     ),
#     cache=True,
#     fastmath=True
# )
def conduct_implicit(u, rho, r, m, c2, mrat, lnL, dt, eps_du):
    """
    For each species, solve tridiagonal system to perform implicit
    conduction step.
    The tridiagonal system is defined by:
        a_i du_i-1 + b_i du_i + c_i du_i+1 = d_i
    u is updated in-place.
    """
    s, N = u.shape

    # Allocate a, b, c, d, du
    a  = np.empty(N, dtype=np.float64)
    b  = np.empty(N, dtype=np.float64)
    c  = np.empty(N, dtype=np.float64)
    d  = np.empty(N, dtype=np.float64)
    du = np.empty(N, dtype=np.float64)

    while True:
        dumax = 0.0
        for k in range(s):
            rk          = r[k]
            mk          = m[k]
            rhok        = rho[k]
            uk          = u[k]
            rhok_int    = interp_linear_to_interfaces(rk, rhok)

            pref        = (-c2) * (mrat[k] * lnL[k,k])

            build_tridiag_system(a, b, c, d, rk, mk, rhok_int, uk, pref, dt)
            solve_tridiagonal_thomas(a, b, c, d, du)

            u[k] += du

            dumaxk = np.max(np.abs(du / uk) )
            if dumaxk > dumax:
                dumax = dumaxk

        if dumax > eps_du:
            dt *= 0.1 * eps_du / dumax
            continue
        else:
            break

    return float(dumax), float(dt)