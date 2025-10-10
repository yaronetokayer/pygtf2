import numpy as np
from numba import njit
from numba.types import float64, Tuple

def burgers_transport(m_in, m_tot_in, r_in, rho_in, v2_in, u_in, p_in, dt_in):
    """
    Compute new arrays due to mass diffusion from Burgers momentum equation

    Arguments
    ---------
    m_in : ndarray, shape (s, N+1)
        Enclosed mass per species.
    m_tot_in : ndarray, shape (N+1,)
        Total enclosed mass profile.
    r_in : ndarray, shape (s, N+1)
        Edge radii per species.
        (in future version, this will be shape (N+1,), since it's the same for all species).
    rho_in : ndarray, shape (s, N)
        Shell densities per species.
    v2_in : ndarray, shape (s, N)
        Shell v2 per species.
    u_in : ndarray, shape (s, N)
        Shell specific internal energy per species.
    p_in : ndarray, shape (s, N)
        Shell pressure per species.
    dt_in : float
        Current proposed time step

    Returns
    -------
    status, m_out, m_tot_out, rho_out, u_out, p_out, v2_out, dt_prop, cfl_lim
    """

    _, Np1 = m_in.shape
    N = Np1 - 1

    #--- Input validation
    assert np.allclose(m_tot_in, np.sum(m_in, axis=0))
    assert np.allclose(1.5 * v2_in, u_in)

    shell_volumes = (r_in[1:]**3 - r_in[:-1]**3) / 3.0  # shape (N,)
    expected_rho = m_in[:,1:] - m_in[:,:-1]
    expected_rho = expected_rho / shell_volumes  # shape (s, N)
    assert np.allclose(rho_in, expected_rho)

    #--- Find w and flux per species at each internal interface
    tiny = np.finfo(np.float64).tiny
    F_if,dt_cfl = flux_serial(r_in, rho_in, v2_in, p_in, m_tot_in, tiny)

    # Check for mass conservation
    if not np.allclose(F_if.sum(axis=0), 0.0):
        print("WARNING: mass may not conserved in Burger's step")

    #--- Check CFL condition is met with current dt_in
    if dt_in > dt_cfl:
        return 'reduce_dt', None, None, None, None, None, None, 0.95 * dt_cfl, None
    else:
        cfl_lim = dt_in / dt_cfl
    
    #--- Update extensive and intensive variables
    m_out, m_tot_out, rho_out, u_out, p_out, v2_out = update_state(r_in, m_in, F_if, dt_in, u_in)
    
    return ('ok', m_out, m_tot_out, rho_out, u_out, p_out, v2_out, dt_in, cfl_lim)

@njit(
    Tuple((float64[:, :], float64))(
        float64[:], float64[:, :], float64[:, :], float64[:, :], float64[:], float64
    ),
    fastmath=True, cache=True
)
def flux_serial(r_in, rho_in, v2_in, p_in, m_tot_in, tiny):
    s = rho_in.shape[0]
    N = rho_in.shape[1]

    F_if = np.zeros((s, N-1), dtype=np.float64)
    dt_cfl = 1.0e300

    A = np.empty((s, s), dtype=np.float64)
    b = np.empty(s, dtype=np.float64)
    w = np.empty(s, dtype=np.float64)
    rho_if = np.empty(s, dtype=np.float64)
    v_if = np.empty(s, dtype=np.float64)
    dpdr = np.empty(s, dtype=np.float64)
    grav = np.empty(s, dtype=np.float64)
    nu_if = np.empty(s, dtype=np.float64)

    for i in range(1, N):
        r_if = float(r_in[i])
        dr_if = 0.5 * (r_in[i+1] - r_in[i-1])
        if dr_if < tiny:
            dr_if = tiny

        for k in range(s):
            rho_if[k] = 0.5 * (rho_in[k, i-1] + rho_in[k, i])
            v_if[k]   = np.sqrt(0.5 * (v2_in[k, i-1] + v2_in[k, i]))

        p_L = p_in[:, i-1]
        p_R = p_in[:, i]

        m_tot_if = float(m_tot_in[i])
        rho_tot_if = 0.0
        for k in range(s):
            rho_tot_if += rho_if[k]
        if rho_tot_if < tiny:
            rho_tot_if = tiny

        for k in range(s):
            dpdr[k] = (p_R[k] - p_L[k]) / dr_if
            grav[k] = rho_if[k] * m_tot_if / max(r_if * r_if, tiny)
            b[k] = dpdr[k] + grav[k]

        for k in range(s):
            val = v_if[k] * rho_if[k]
            if val < tiny:
                val = tiny
            nu_if[k] = val

        for k in range(s):
            for j in range(s):
                A[k, j] = 0.0

        for k in range(s):
            diag = 0.0
            for j in range(s):
                if j != k:
                    denom = nu_if[k] + nu_if[j]
                    if denom < tiny:
                        denom = tiny
                    K_kj = (rho_if[k] * rho_if[j] / rho_tot_if) * (2.0 * nu_if[k] * nu_if[j] / denom)
                    A[k, j] = K_kj
                    diag -= K_kj
            A[k, k] = diag

        # zero net flux constraint on last row: sum_k rho_k * w_k = 0
        b[s-1] = 0.0
        for j in range(s):
            A[s-1, j] = rho_if[j]

        w[:] = np.linalg.solve(A, b)

        w_abs_max = tiny
        for k in range(s):
            val = abs(w[k])
            if val > w_abs_max:
                w_abs_max = val
        dt_candidate = 0.5 * dr_if / w_abs_max
        if dt_candidate < dt_cfl:
            dt_cfl = dt_candidate

        for k in range(s):
            F_if[k, i-1] = rho_if[k] * w[k]

    return F_if, dt_cfl

@njit(
    Tuple((
        float64[:, :],  # m_out     (s, N+1)
        float64[:],     # m_tot_out (N+1,)
        float64[:, :],  # rho_out   (s, N)
        float64[:, :],  # u_out     (s, N)
        float64[:, :],  # p_out     (s, N)
        float64[:, :],  # v2_out    (s, N)
    ))(
        float64[:],     # r_in   (N+1,)
        float64[:, :],  # m_in   (s, N+1)
        float64[:, :],  # F_if   (s, N-1)
        float64,        # dt_in  (scalar)
        float64[:, :],  # u_in   (s, N)
    ),
    fastmath=True, cache=True
)
def update_state(r_in, m_in, F_if, dt_in, u_in):
    """
    Conservative update of mass, density, internal energy, pressure and v^2.

    Shapes:
        r_in:    (N+1,)
        m_in:    (s, N+1)  -- shell-edge masses per species
        F_if:    (s, N-1)  -- interface mass fluxes per species at internal faces
        dt_in:   scalar
        u_in:    (s, N)    -- cell-centered internal energy per mass

    Returns:
        m_out:     (s, N+1)
        m_tot_out: (N+1,)
        rho_out:   (s, N)
        u_out:     (s, N)
        p_out:     (s, N)
        v2_out:    (s, N)
    """
    s = m_in.shape[0]
    Np1 = m_in.shape[1]
    N = Np1 - 1

    # --- Geometry
    # A_if on internal interfaces: j = 1..N-1 (store at col=j-1)
    A_if = np.empty(max(0, N - 1), dtype=np.float64)
    for j in range(1, N):
        rj = r_in[j]
        A_if[j - 1] = rj * rj

    # Cell volumes: j = 0..N-1
    V_cell = np.empty(N, dtype=np.float64)
    for j in range(N):
        rp = r_in[j + 1]
        rl = r_in[j]
        V_cell[j] = (rp * rp * rp - rl * rl * rl) / 3.0

    # --- Mass update
    m_out = m_in.copy()
    # interior edges j = 1..N-1 correspond to F_if[:, j-1]
    for j in range(1, N):
        area_dt = A_if[j - 1] * dt_in
        col = j - 1
        for k in range(s):
            m_out[k, j] = m_in[k, j] - F_if[k, col] * area_dt

    # Total mass per edge
    m_tot_out = np.empty(Np1, dtype=np.float64)
    for i in range(Np1):
        acc = 0.0
        for k in range(s):
            acc += m_out[k, i]
        m_tot_out[i] = acc

    # Cell masses and density
    m_cell_out = np.empty((s, N), dtype=np.float64)
    rho_out = np.empty((s, N), dtype=np.float64)
    for k in range(s):
        for j in range(N):
            mc = m_out[k, j + 1] - m_out[k, j]
            m_cell_out[k, j] = mc
            rho_out[k, j] = mc / V_cell[j]

    # --- Energy advection
    # E_in = (m_in[:, 1:] - m_in[:, :-1]) * u_in
    E_in = np.empty((s, N), dtype=np.float64)
    for k in range(s):
        for j in range(N):
            mc_in = m_in[k, j + 1] - m_in[k, j]
            E_in[k, j] = mc_in * u_in[k, j]

    # Upwind face energies Phi[:, col], col = j-1 for interface j in [1..N-1]
    Phi = np.empty((s, max(0, N - 1)), dtype=np.float64)
    for j in range(1, N):
        col = j - 1
        for k in range(s):
            fij = F_if[k, col]
            # donor: left (j-1) if F>0, else right (j)
            u_face = u_in[k, j - 1] if fij >= 0.0 else u_in[k, j]
            Phi[k, col] = fij * u_face

    # E_out = E_in with flux exchange across internal faces
    E_out = E_in.copy()
    for col in range(N - 1):
        area_dt = A_if[col] * dt_in
        for k in range(s):
            dE = Phi[k, col] * area_dt
            E_out[k, col]     -= dE      # left cell (j-1)
            E_out[k, col + 1] += dE      # right cell (j)

    # --- Intensive updates: u_out, v2_out, p_out
    u_out = np.empty((s, N), dtype=np.float64)
    v2_out = np.empty((s, N), dtype=np.float64)
    p_out = np.empty((s, N), dtype=np.float64)
    for k in range(s):
        for j in range(N):
            mc = m_cell_out[k, j]
            if mc > 0.0:
                uval = E_out[k, j] / mc
            else:
                uval = 0.0
            u_out[k, j] = uval
            v2_out[k, j] = uval / 1.5
            p_out[k, j] = (2.0 / 3.0) * rho_out[k, j] * uval

    return m_out, m_tot_out, rho_out, u_out, p_out, v2_out

def burgers_transport_readable(m_in, m_tot_in, r_in, rho_in, v2_in, u_in, p_in, dt_in):
    """
    Compute new arrays due to mass diffusion from Burgers momentum equation

    Arguments
    ---------
    m_in : ndarray, shape (s, N+1)
        Enclosed mass per species.
    m_tot_in : ndarray, shape (N+1,)
        Total enclosed mass profile.
    r_in : ndarray, shape (s, N+1)
        Edge radii per species.
        (in future version, this will be shape (N+1,), since it's the same for all species).
    rho_in : ndarray, shape (s, N)
        Shell densities per species.
    v2_in : ndarray, shape (s, N)
        Shell v2 per species.
    u_in : ndarray, shape (s, N)
        Shell specific internal energy per species.
    p_in : ndarray, shape (s, N)
        Shell pressure per species.
    dt_in : float
        Current proposed time step

    Returns
    -------
    status, m_out, m_tot_out, rho_out, u_out, p_out, v2_out, dt_prop, cfl_lim
    """

    s, Np1 = m_in.shape
    N = Np1 - 1
    tiny = np.finfo(np.float64).tiny

    #--- Input validation
    assert np.allclose(m_tot_in, np.sum(m_in, axis=0))
    assert np.allclose(1.5 * v2_in, u_in)

    shell_volumes = (r_in[1:]**3 - r_in[:-1]**3) / 3.0  # shape (N,)
    expected_rho = m_in[:,1:] - m_in[:,:-1]
    expected_rho = expected_rho / shell_volumes  # shape (s, N)
    assert np.allclose(rho_in, expected_rho)

    #--- Find w and flux per species at each internal interface
    F_if = np.zeros((s, N-1))        # to store fluxes per interface
    dt_cfl = 1e300

    for i in range(1, N):
        # Interface values
        r_if = float(r_in[i])
        dr_if = float(0.5 * (r_in[i+1] - r_in[i-1]))
        if dr_if < tiny: dr_if = tiny

        rho_if = 0.5 * (rho_in[:,i-1] + rho_in[:,i])        # shape (s,)
        v_if = np.sqrt(0.5 * (v2_in[:,i-1] + v2_in[:,i]))   # shape (s,)
        p_L = p_in[:,i-1]
        p_R = p_in[:,i]

        m_tot_if = float(m_tot_in[i])
        rho_tot_if = float(np.sum(rho_if)); rho_tot_if = max(rho_tot_if, tiny)

        # b_k = d/dr(p_k) + rho_k*g
        dpdr = (p_R - p_L) / dr_if
        grav = rho_if * m_tot_if / max(r_if * r_if, tiny)
        b = dpdr + grav

        # A_kj = (rho_k * rho_j / rho_tot) * nu_kj
        # A_kj = Kjk; K_kk = -\sum_{j\neq k} K_kj
        A = np.zeros((s,s))
        nu_if = np.maximum(tiny, v_if * rho_if) # nu ~ 1/trelax = rho*sqrt(v2)
        for k in range(s):
            for j in range(s):
                if j != k:
                    K_kj = (rho_if[k] * rho_if[j] / rho_tot_if)
                    K_kj *= 2 * nu_if[k] * nu_if[j] / (nu_if[k] + nu_if[j]) # Harmonic mean of friction terms
                
                    A[k,j] = K_kj
                    A[k,k] -= K_kj

        # Replace bottom row of matrix equation with zero net flux constraint
        b[-1] = 0.0
        A[-1,:] = rho_if[:]

        # Solve Aw = b
        w = np.linalg.solve(A,b)

        # Update dt_cfl
        w_abs_max = max(tiny, np.max(np.abs(w)))
        dt_cfl = min(dt_cfl, 0.5 * dr_if / w_abs_max) # 0.5 safety factor

        # Mass flux at this interface: F_k = rho_k * w_k
        F_if[:, i-1] = rho_if * w

    # Check for mass conservation
    if not np.allclose(F_if.sum(axis=0), 0.0):
        print("WARNING: mass may not conserved in Burger's step")

    #--- Check CFL condition is met with current dt_in
    if dt_in > dt_cfl:
        return 'reduce_dt', None, None, None, None, None, None, 0.95 * dt_cfl, None
    else:
        cfl_lim = dt_in / dt_cfl
    
    #--- Update extensive and intensive variables
    A_if = r_in[1:-1]**2
    V_cell = (r_in[1:]**3 - r_in[:-1]**3) / 3.0

    # Mass and density
    m_out = m_in.copy()
    m_out[:, 1:-1] = m_in[:, 1:-1] - F_if * A_if[np.newaxis, :] * dt_in
    m_tot_out = m_out.sum(axis=0)

    m_cell_out = m_out[:, 1:] - m_out[:, :-1]     # (s, N)
    rho_out = m_cell_out / V_cell[np.newaxis, :]

    # Energy advection
    E_in  = (m_in[:, 1:] - m_in[:, :-1]) * u_in

    # Build upwind face energies: Phi[:, j-1] uses edge j (1..N-1)
    Phi = np.empty_like(F_if)                    # (s, N-1)
    for j in range(1, N):
        col = j - 1
        # upwind donor: if F>0, donor is left shell (j-1); else right shell (j)
        u_face = np.where(F_if[:, col] >= 0.0, u_in[:, j-1], u_in[:, j])
        Phi[:, col] = F_if[:, col] * u_face

    E_out = E_in.copy()
    E_out[:,:-1]    -= Phi * A_if[np.newaxis, :] * dt_in
    E_out[:,1:]     += Phi * A_if[np.newaxis, :] * dt_in

    # Other intensive quantities
    with np.errstate(divide='ignore', invalid='ignore'):
        u_out = np.where(m_cell_out > 0.0, E_out / m_cell_out, 0.0)
    v2_out = u_out / 1.5
    p_out  = (2.0/3.0) * rho_out * u_out

    return ('ok', m_out, m_tot_out, rho_out, u_out, p_out, v2_out, dt_in, cfl_lim)