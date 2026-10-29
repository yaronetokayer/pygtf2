import numpy as np 
from numba import njit, float64, types, void
import math
from multigtf.util.interpolate import interp_linear_to_interfaces
from multigtf.util.calc import solve_tridiagonal_thomas

HEX_FIRST       = 0
COND_FIRST      = 1
STRANG_SPLIT    = 2

@njit(void(float64[:, :], float64[:, :],float64[:, :], float64[:], float64[:, :],float64[:, :], float64, float64), cache=True, fastmath = True)
def hex_explicit(v2, rho, lnL, mrat, r, v2_hex_src_work, dt, c1):
    """
    Apply one explicit forward-Euler inter-species heat-exchange step
    directly to v2.

    Parameters
    ----------
    v2 : ndarray, shape (s, N)
        Velocity dispersion squared. Updated in place.
    rho : ndarray, shape (s, N)
        Density per cell for each species.
    lnL : ndarray, shape (s, s)
        Inter-species coupling coefficients.
    mrat : ndarray, shape (s,)
        Mass-ratio-like coefficient for each species.
    r : ndarray, shape (s, N+1)
        Radial cell interfaces for each species.
    v2_hex_src_work : ndarray, shape (s, N)
        Workspace used to hold the unchanged input state for this
        explicit substep. It must not alias v2.
    dt : float
        Explicit heat-exchange timestep.
    c1 : float
        Multiplicative constant for inter-species exchange term.
    """
    s, N = v2.shape

    # Zero workspace
    for n in range(s):
        for i in range(N):
            v2_hex_src_work[n, i] = v2[n, i]

    # Original exchange term was written in u.
    # Converting to v2 = (2/3)u gives:
    # prefac = c1 * ((2/3) / sqrt(1.5))
    prefac_dt = c1 * ((2.0 / 3.0) / np.sqrt(1.5)) * dt

    for n in range(s):
        mrn = mrat[n]

        for k in range(s):
            if k == n:
                continue

            pair_prefac = prefac_dt * lnL[n, k]
            mrk = mrat[k]

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

                rn_lo3 = rn_lo * rn_lo * rn_lo
                rn_hi3 = rn_hi * rn_hi * rn_hi
                vol_n_r3 = rn_hi3 - rn_lo3

                if vol_n_r3 > 0.0:
                    rrmin3 = rrmin * rrmin * rrmin
                    rrmax3 = rrmax * rrmax * rrmax
                    dvol_overlap_r3 = rrmax3 - rrmin3

                    if dvol_overlap_r3 > 0.0:
                        vol_ratio = dvol_overlap_r3 / vol_n_r3

                        # Always read both species from the unchanged original state.
                        v2n = v2_hex_src_work[n, i]
                        v2k = v2_hex_src_work[k, j]
                        denom = v2k + v2n

                        if denom > 0.0:
                            root = np.sqrt(denom)
                            inv_p32 = 1.0 / (denom * root)

                            v2[n, i] += pair_prefac * rho[k, j] * vol_ratio * (mrk * v2k - mrn * v2n) * inv_p32

                if rn_hi <= rk_hi:
                    i += 1
                else:
                    j += 1

@njit(void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64, float64), cache=True, fastmath=True)
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
        Superdiagonal coefficients (multiply du_{i+1}) for interior nodes.
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
    N = uk.shape[0]
    sqrt2       = 1.41421356237309
    inv_sqrt2_pref_dt  = 1.0 / (sqrt2 * pref * dt)

    ### First cell ###
    rL  = rk[0]
    rR  = rk[1]
    rRR = rk[2]

    drcR = 0.5 * ( rRR - rL )
    rR2  = rR * rR
    coefR = rhok_int[0] * rR2 / drcR

    uC = uk[0]
    uR = uk[1]

    duR         = uR - uC
    suR         = uR + uC
    inv_sqrt_suR    = 1.0 / math.sqrt(suR)
    inv_suR32       = inv_sqrt_suR / suR
    common_p_R = inv_sqrt_suR + 0.5 * duR * inv_suR32
    common_m_R = inv_sqrt_suR - 0.5 * duR * inv_suR32

    a[0] = 0.0
    b[0] = -common_p_R - mk[1] * inv_sqrt2_pref_dt / coefR
    c[0] = common_m_R
    d[0] = - duR * inv_sqrt_suR

    ### Interior cells ###
    # Initilize sliding window
    rL  = rR
    rR  = rRR
    rRR = rk[3]

    drcR    = 0.5 * ( rRR - rL )
    rR2     = rR * rR
    coefL   = coefR
    coefR   = rhok_int[1] * rR2 / drcR

    uC = uR
    uR = uk[2]

    duL             = duR
    duR             = uR - uC
    suR             = uR + uC
    inv_sqrt_suL    = inv_sqrt_suR
    inv_sqrt_suR    = 1.0 / math.sqrt(suR)
    inv_suR32       = inv_sqrt_suR / suR

    common_p_L = common_p_R
    common_m_L = common_m_R
    common_p_R = inv_sqrt_suR + 0.5 * duR * inv_suR32
    common_m_R = inv_sqrt_suR - 0.5 * duR * inv_suR32

    for j in range(1, N - 1):
        a[j] = coefL * common_p_L
        b[j] = - coefR * common_p_R - coefL * common_m_L - inv_sqrt2_pref_dt * ( mk[j + 1] - mk[j] )
        c[j] = coefR * common_m_R
        d[j] = coefL * duL * inv_sqrt_suL - coefR * duR * inv_sqrt_suR

        # Advance sliding window
        if j < N - 2:
            rL  = rR
            rR  = rRR
            rRR = rk[j + 3]

            drcR    = 0.5 * ( rRR - rL )
            rR2     = rR * rR
            coefL   = coefR
            coefR   = rhok_int[j + 1] * rR2 / drcR

            uC = uR
            uR = uk[j + 2]

            duL             = duR
            duR             = uR - uC
            suR             = uR + uC
            inv_sqrt_suL    = inv_sqrt_suR
            inv_sqrt_suR    = 1.0 / math.sqrt(suR)
            inv_suR32       = inv_sqrt_suR / suR

            common_p_L = common_p_R
            common_m_L = common_m_R
            common_p_R = inv_sqrt_suR + 0.5 * duR * inv_suR32
            common_m_R = inv_sqrt_suR - 0.5 * duR * inv_suR32

    ### Last cell ###
    a[N - 1] = common_p_R
    b[N - 1] = - common_m_R - inv_sqrt2_pref_dt * ( mk[N] - mk[N - 1] ) / coefR
    c[N - 1] = 0.0
    d[N - 1] = duR * inv_sqrt_suR

@njit(void(float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64[:], float64[:], float64[:], float64[:], float64, float64[:], float64[:, :], float64[:, :], float64), cache=True, fastmath=True)
def conduct_implicit(v2, rho, r, m, 
                     a, b, c, d, 
                     c2, mrat, lnL, du_trial, dt,):
    """
    Apply implicit intra-species conduction at the supplied fixed timestep.

    Solve a tridiagonal system for each species' specific-energy increment
    and update v2 in place using u = 1.5*v2. This routine has no acceptance
    limiter and returns None; any timestep rejection is the caller's job.
    The supplied du_trial workspace is overwritten with the increments.
    """
    s, N = v2.shape

    for k in range(s):
        rk       = r[k]
        mk       = m[k]
        rhok     = rho[k]
        uk       = 1.5 * v2[k]
        rhok_int = interp_linear_to_interfaces(rk, rhok)

        pref = c2 * (mrat[k] * lnL[k, k])

        build_tridiag_system(a, b, c, d, rk, mk, rhok_int, uk, pref, dt)
        solve_tridiagonal_thomas(a, b, c, d, du_trial[k])

    for k in range(s):
        for i in range(N):
            v2[k, i] += (2.0 / 3.0) * du_trial[k, i]

@njit(
    void(
        float64[:, :],  # v2
        float64[:, :],  # rho
        float64[:, :],  # r
        float64[:, :],  # m
        float64,         # c1
        float64,         # c2
        float64[:],      # mrat
        float64[:, :],  # lnL
        float64[:, :],  # v2_work
        float64[:],      # a_work
        float64[:],      # b_work
        float64[:],      # c_work
        float64[:],      # d_work
        float64,         # dt
        types.int64,     # order
    ),
    cache=True, fastmath=True,
)
def conduct_imex_once(
    v2, rho, r, m,
    c1, c2, mrat, lnL,
    v2_work, a_work, b_work, c_work, d_work,
    dt, order,
):
    """
    Apply one fixed-timestep split conduction update to v2 in place.

    HEX_FIRST applies explicit inter-species exchange followed by implicit
    intra-species conduction; COND_FIRST reverses that order. STRANG_SPLIT
    uses exchange(dt/2), conduction(dt), then exchange(dt/2).
    The work arrays are overwritten. There is no timestep limiter or return
    value; the caller must measure the change and accept or reject it.
    An unrecognized order raises ValueError.
    """
    if order == HEX_FIRST:
        hex_explicit(v2, rho, lnL, mrat, r, v2_work, dt, c1,)
        conduct_implicit(v2, rho, r, m, a_work, b_work, c_work, d_work, c2, mrat, lnL, v2_work, dt)

    elif order == COND_FIRST:
        conduct_implicit(v2, rho, r, m, a_work, b_work, c_work, d_work, c2, mrat, lnL, v2_work, dt)
        hex_explicit(v2, rho, lnL, mrat, r, v2_work, dt, c1,)

    elif order == STRANG_SPLIT:
        half_dt = 0.5 * dt

        # First half-step hex on initial state determines dt_eff
        hex_explicit(v2, rho, lnL, mrat, r, v2_work, half_dt, c1,)

        conduct_implicit(v2, rho, r, m, a_work, b_work, c_work, d_work, c2, mrat, lnL, v2_work, dt,)

        # Recompute the second half-step on the updated state.
        hex_explicit(v2, rho, lnL, mrat, r, v2_work, half_dt, c1,)

    else:
        raise ValueError("unrecognized order code in conduction step")
