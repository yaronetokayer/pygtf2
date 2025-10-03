import numpy as np
from numba import njit, float64, types

def compute_f_fick_test(r_edges, rho, v2, c_d) -> np.ndarray:
    v2_avg = 0.5 * (v2[:,:-1] + v2[:,:-1])
    rho_avg = 0.5 * (rho[:,:-1] + rho[:,:-1])
    dr = 0.5 * (r_edges[0,:-2] - r_edges[0,2:])
    mass_frac = 0.5 * (rho[:]) / rho.sum(axis=0)

@njit(
    types.Tuple((float64[:, :], float64[:], float64[:]))(
        float64[:, :],  # r_edges: (s, N+1) -- shared geometry, rows identical
        float64[:, :],  # rho:     (s, N)
        float64[:, :],  # v2:      (s, N)
        float64,        # c_d:     scalar prefactor
        types.boolean,  # use_species_Dk: if True, use per-species D_k; else D_mix
        types.boolean,  # use_harmonic_mix: only used when use_species_Dk=False
    ),
    cache=True, fastmath=True
)
def compute_f_fick(r_edges, rho, v2, c_d, use_species_Dk, use_harmonic_mix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute species-wise Fick flux at interior edges with zero-net-flux projection.

    Returns
    -------
    F_if    : (s, N-1)  species flux at interior **edges** j=1..N-1 (outward +)
    dr_if   : (N-1,)    center-to-center spacings for gradients
    Dmax_if : (N-1,)    max_k D_k at each interface (for CFL control)
    """
    s, nedge = r_edges.shape
    N = nedge - 1
    r0 = r_edges[0]
    tiny = np.finfo(np.float64).tiny

    # Cell centers and center-to-center spacing
    rc    = 0.5 * (r0[1:] + r0[:-1])      # (N,)
    dr_if = rc[1:] - rc[:-1]              # (N-1,)

    # Totals and mass fractions
    rho_tot = np.zeros(N, dtype=np.float64)
    for i in range(N):
        acc = 0.0
        for k in range(s):
            acc += rho[k, i]
        rho_tot[i] = acc if acc > tiny else tiny

    X = np.zeros((s, N), dtype=np.float64)
    for i in range(N):
        inv = 1.0 / rho_tot[i]
        for k in range(s):
            X[k, i] = rho[k, i] * inv

    # Interface averages
    rho_if = np.zeros((s, N-1), dtype=np.float64)
    v2_if  = np.zeros((s, N-1), dtype=np.float64)
    X_if   = np.zeros((s, N-1), dtype=np.float64)
    rho_tot_if = np.zeros(N-1, dtype=np.float64)

    for i in range(N-1):
        rt = 0.0
        for k in range(s):
            rij = 0.5 * (rho[k, i] + rho[k, i+1])
            vij = 0.5 * (v2[k, i]  + v2[k, i+1])
            rho_if[k, i] = rij if rij > tiny else tiny
            v2_if[k, i]  = vij if vij > tiny else tiny
            X_if[k, i]   = 0.5 * (X[k, i] + X[k, i+1])
            rt += rho_if[k, i]
        rho_tot_if[i] = rt if rt > tiny else tiny

    # Per-species interface diffusivities from your trelax proxy:
    # trelax_k ~ 1/(sqrt(v2_k)*rho_k)  =>  D_k ~ c_d * v2_k * trelax_k ~ c_d * sqrt(v2_k)/rho_k
    Dk_if = np.zeros((s, N-1), dtype=np.float64)
    Dmax_if = np.zeros(N-1, dtype=np.float64)
    for i in range(N-1):
        dmax = 0.0
        for k in range(s):
            d = c_d * np.sqrt(v2_if[k, i]) / rho_if[k, i]
            Dk_if[k, i] = d
            if d > dmax:
                dmax = d
        Dmax_if[i] = dmax if dmax > tiny else tiny

    # Gradient of X_k using center-to-center spacing
    gradX = np.zeros((s, N-1), dtype=np.float64)
    for i in range(N-1):
        inv_dr = 1.0 / (dr_if[i] if dr_if[i] > tiny else tiny)
        for k in range(s):
            gradX[k, i] = (X[k, i+1] - X[k, i]) * inv_dr

    # Raw Fick fluxes
    f_raw = np.zeros((s, N-1), dtype=np.float64)
    if use_species_Dk:
        # Level 2: per-species D_k
        for i in range(N-1):
            coeff = - rho_tot_if[i]
            for k in range(s):
                f_raw[k, i] = coeff * Dk_if[k, i] * gradX[k, i]
    else:
        # Level 1: single mixture D_mix
        Dmix_if = np.zeros(N-1, dtype=np.float64)
        if not use_harmonic_mix:
            # arithmetic: sum_k X_k D_k
            for i in range(N-1):
                acc = 0.0
                for k in range(s):
                    acc += X_if[k, i] * Dk_if[k, i]
                Dmix_if[i] = acc
        else:
            # harmonic: 1 / sum_k X_k / D_k
            for i in range(N-1):
                acc = 0.0
                for k in range(s):
                    d = Dk_if[k, i]
                    if d < tiny:
                        d = tiny
                    acc += X_if[k, i] / d
                if acc < tiny:
                    acc = tiny
                Dmix_if[i] = 1.0 / acc

        for i in range(N-1):
            coeff = - rho_tot_if[i] * Dmix_if[i]
            for k in range(s):
                f_raw[k, i] = coeff * gradX[k, i]

    # Zero-net-flux projection at each interface: enforce sum_k F_k = 0
    F_if = np.zeros((s, N-1), dtype=np.float64)
    for i in range(N-1):
        sumn = 0.0
        for k in range(s):
            sumn += f_raw[k, i]
        for k in range(s):
            F_if[k, i] = f_raw[k, i] - X_if[k, i] * sumn

    return F_if, dr_if, Dmax_if

@njit(
    types.Tuple((float64[:, :], float64, float64))(
        float64[:, :],  # F_if: (s, N-1)
        float64[:, :],  # m_encl_edges: (s, N+1)
        float64[:, :],  # r_edges: (s, N+1) -- shared geometry
        float64,        # dt_prop
        float64[:],     # dr_if: (N-1,)
        float64[:],     # Dmax_if: (N-1,)  max_k D_k at each interface
        float64         # cfl_coeff (e.g. 0.4)
    ),
    cache=True, fastmath=True
)
def update_m(F_if, m_encl_edges, r_edges, dt_prop, dr_if, Dmax_if, cfl_coeff) -> tuple[np.ndarray, float, float]:
    """
    Conservative update of per-species ENCLOSED mass at edges using interface fluxes,
    with a CFL check for diffusion and automatic dt reduction if necessary.

    Returns
    -------
    m_encl_new : (s, N+1)  updated enclosed mass
    dt_used    : float      min(dt_prop, dt_cfl)
    dt_cfl     : float      global CFL bound computed this step
    """
    s, nedge = r_edges.shape
    N = nedge - 1
    r0 = r_edges[0]

    tiny = np.finfo(np.float64).tiny
    # Global CFL: dt_cfl = cfl_coeff * min_i dr_i^2 / Dmax_i
    dt_cfl = 1.0e300
    for i in range(N-1):
        d = Dmax_if[i]
        if d < tiny:
            continue
        bound = cfl_coeff * (dr_if[i] * dr_if[i]) / d
        if bound < dt_cfl:
            dt_cfl = bound
    if dt_cfl == 1.0e300:
        dt_cfl = dt_prop  # effectively no constraint if all D were tiny

    dt_used = dt_prop if dt_prop <= dt_cfl else dt_cfl

    # Prepare output and copy boundaries unchanged (zero-flux BCs)
    m_new = np.empty_like(m_encl_edges)
    for k in range(s):
        m_new[k, 0] = m_encl_edges[k, 0]
        m_new[k, N] = m_encl_edges[k, N]

    # Update interior edges j=1..N-1; interface index i_if=j-1
    for j in range(1, N):
        A = r0[j] * r0[j]
        i_if = j - 1
        for k in range(s):
            m_new[k, j] = m_encl_edges[k, j] - F_if[k, i_if] * A * dt_used

    return m_new, dt_used, dt_cfl


@njit(
    types.Tuple((float64[:, :], float64[:, :]))(
        float64[:, :],  # m_encl_edges_new (s, N+1)
        float64[:, :],  # u (s, N)  specific internal energies
        float64[:]      # r_edges (N+1,) shared
    ),
    cache=True, fastmath=True
)
def update_rho_p(m_encl_edges_new, u, r_edges) -> tuple[np.ndarray, np.ndarray]:
    s, nedge = m_encl_edges_new.shape
    N = nedge - 1

    rho_new = np.empty((s, N), dtype=np.float64)
    p_new   = np.empty((s, N), dtype=np.float64)

    # shell volumes
    V = (1.0/3.0) * (r_edges[1:]**3 - r_edges[:-1]**3)

    for k in range(s):
        for i in range(N):
            m_shell = m_encl_edges_new[k, i+1] - m_encl_edges_new[k, i]
            rho_new[k, i] = m_shell / V[i]
            p_new[k, i]   = rho_new[k, i] * u[k, i]

    return rho_new, p_new