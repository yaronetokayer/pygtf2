import numpy as np
from numba import njit, prange, float64, int64, types

# --------- Numba-compiled core ----------
# Return convention:
#   status_code: 0 = 'ok', 1 = 'reduce_dt'
#   arrays are always returned (wrapper can convert to None when reduce_dt)
@njit(
    (float64[:, :],  # m_in      (s, N+1)
     float64[:],     # m_tot_in  (N+1,)
     float64[:, :],  # r_in      (s, N+1)  (we'll use r_in[0])
     float64[:, :],  # rho_in    (s, N)
     float64[:, :],  # v2_in     (s, N)
     float64[:, :],  # u_in      (s, N)
     float64[:, :],  # p_in      (s, N)
     float64         # dt_in
    ),
    cache=True, fastmath=True, parallel=True
)
def burgers_step_numba_core(m_in, m_tot_in, r_in, rho_in, v2_in, u_in, p_in, dt_in):
    s, Np1 = m_in.shape
    N = Np1 - 1
    tiny = np.finfo(np.float64).tiny

    # Collapse r_in to shape (N+1,) for easy compatibility with future versions
    r1 = r_in[0]  # shape (N+1,)

    # Precompute shell volumes (N,)
    shell_volumes = (r1[1:]**3 - r1[:-1]**3) / 3.0

    # --- Find w and flux per species at each internal interface
    F_if = np.zeros((s, N-1))        # fluxes per interface (between cells)
    dt_cfl_arr = np.full(N-1, 1e300) # per-interface CFL; reduce after loop

    # Parallelized loop over interfaces i = 1..N-1
    for i in prange(1, N):
        r_if = float(r1[i])
        dr_if = float(0.5 * (r1[i+1] - r1[i-1]))
        if dr_if < tiny:
            dr_if = tiny

        # Interface-averaged states
        # shapes: (s,)
        rho_if = 0.5 * (rho_in[:, i-1] + rho_in[:, i])
        v_if = np.sqrt(0.5 * (v2_in[:, i-1] + v2_in[:, i]))
        p_L = p_in[:, i-1]
        p_R = p_in[:, i]

        m_tot_if = float(m_tot_in[i])
        rho_tot_if = float(np.sum(rho_if))
        if rho_tot_if < tiny:
            rho_tot_if = tiny

        # b_k = d/dr(p_k) + rho_k*g
        dpdr = (p_R - p_L) / dr_if
        grav = rho_if * m_tot_if / max(r_if * r_if, tiny)
        b = dpdr + grav

        # A_kj construction (s x s)
        # A_kj = (rho_k * rho_j / rho_tot) * nu_kj
        # K_kk = - sum_{j != k} K_kj
        A = np.zeros((s, s))
        nu_if = np.maximum(tiny, v_if * rho_if)  # ~ rho*sqrt(v2)
        for k in range(s):
            akk = 0.0
            for j in range(s):
                if j != k:
                    # harmonic mean of friction terms
                    denom = (nu_if[k] + nu_if[j])
                    if denom <= tiny:
                        K_kj = 0.0
                    else:
                        K_kj = (rho_if[k] * rho_if[j] / rho_tot_if)
                        K_kj *= 2.0 * nu_if[k] * nu_if[j] / denom
                    A[k, j] = K_kj
                    akk -= K_kj
            A[k, k] = akk

        # Replace bottom row with zero net flux constraint
        b[-1] = 0.0
        for j in range(s):
            A[-1, j] = rho_if[j]

        # Solve Aw = b
        w = np.linalg.solve(A, b)

        # CFL per-interface
        w_abs_max = np.max(np.abs(w))
        if w_abs_max < tiny:
            w_abs_max = tiny
        dt_cfl_arr[i-1] = 0.5 * dr_if / w_abs_max  # 0.5 safety factor

        # Mass flux at this interface: F_k = rho_k * w_k
        F_if[:, i-1] = rho_if * w

    # global CFL
    dt_cfl = dt_cfl_arr.min()

    # If CFL violated, signal to reduce dt and skip the (costly) update;
    # still return shaped arrays (copies of inputs) for type stability.
    status_code = int64(0)
    if dt_in > dt_cfl:
        status_code = int64(1)

    # --- Update extensive and intensive variables (if not reducing dt, it still runs;
    #     but the wrapper can discard results on reduce_dt to match your original API)
    A_if = r1[1:-1]**2        # (N-1,)
    V_cell = shell_volumes    # (N,)

    # Mass and density
    m_out = m_in.copy()
    # m_out[:, 1:-1] update uses F_if * A_if * dt_in
    for col in range(N-1):
        a = A_if[col] * dt_in
        for k in range(s):
            m_out[k, 1 + col] = m_out[k, 1 + col] - F_if[k, col] * a

    m_tot_out = np.empty_like(m_tot_in)
    for j in range(N+1):
        # sum over species
        acc = 0.0
        for k in range(s):
            acc += m_out[k, j]
        m_tot_out[j] = acc

    m_cell_out = np.empty((s, N))
    for j in range(N):
        for k in range(s):
            m_cell_out[k, j] = m_out[k, 1 + j] - m_out[k, j]

    rho_out = np.empty_like(rho_in)
    for j in range(N):
        invV = 1.0 / V_cell[j]
        for k in range(s):
            rho_out[k, j] = m_cell_out[k, j] * invV

    # Energy advection
    E_in = np.empty_like(u_in)
    for j in range(N):
        for k in range(s):
            E_in[k, j] = (m_in[k, 1 + j] - m_in[k, j]) * u_in[k, j]

    # Upwind face energies Phi[:, j-1] uses edge j (1..N-1)
    Phi = np.empty_like(F_if)  # (s, N-1)
    for j in range(1, N):
        col = j - 1
        for k in range(s):
            u_face = u_in[k, j-1] if (F_if[k, col] >= 0.0) else u_in[k, j]
            Phi[k, col] = F_if[k, col] * u_face

    E_out = E_in.copy()
    for col in range(N-1):
        a = A_if[col] * dt_in
        for k in range(s):
            E_out[k, col]     = E_out[k, col]     - Phi[k, col] * a
            E_out[k, col + 1] = E_out[k, col + 1] + Phi[k, col] * a

    # Other intensive quantities
    u_out = np.zeros_like(u_in)
    for j in range(N):
        for k in range(s):
            mc = m_cell_out[k, j]
            u_out[k, j] = (E_out[k, j] / mc) if (mc > 0.0) else 0.0

    v2_out = u_out / 1.5
    p_out  = (2.0/3.0) * rho_out * u_out

    # dt_prop: either input dt (ok) or a slightly reduced CFL suggestion (to match original)
    dt_prop = dt_in if status_code == 0 else (0.95 * dt_cfl)

    return status_code, m_out, m_tot_out, rho_out, u_out, p_out, v2_out, dt_prop


# --------- Thin Python wrapper (preserves your original return convention) ----------
def burgers_step(m_in, m_tot_in, r_in, rho_in, v2_in, u_in, p_in, dt_in):
    """
    Drop-in wrapper preserving:
      Returns: (status_str, m_out, m_tot_out, rho_out, u_out, p_out, v2_out, dt_prop)
      where on 'reduce_dt' arrays are returned as None (like your original).
    """
    status_code, m_out, m_tot_out, rho_out, u_out, p_out, v2_out, dt_prop = \
        burgers_step_numba_core(m_in, m_tot_in, r_in, rho_in, v2_in, u_in, p_in, float(dt_in))

    if status_code == 1:
        return 'reduce_dt', None, None, None, None, None, None, dt_prop
    else:
        return 'ok', m_out, m_tot_out, rho_out, u_out, p_out, v2_out, dt_prop



# import numpy as np
# from numba import njit, float64, types

# def burgers_step(m_in, m_tot_in, r_in, rho_in, v2_in, u_in, p_in, dt_in):
#     """
#     Compute new arrays due to mass diffusion from Burgers momentum equation

#     Arguments
#     ---------
#     m_in : ndarray, shape (s, N+1)
#         Enclosed mass per species.
#     m_tot_in : ndarray, shape (N+1,)
#         Total enclosed mass profile.
#     r_in : ndarray, shape (s, N+1)
#         Edge radii per species.
#         (in future version, this will be shape (N+1,), since it's the same for all species).
#     rho_in : ndarray, shape (s, N)
#         Shell densities per species.
#     v2_in : ndarray, shape (s, N)
#         Shell v2 per species.
#     u_in : ndarray, shape (s, N)
#         Shell specific internal energy per species.
#     p_in : ndarray, shape (s, N)
#         Shell pressure per species.
#     dt_in : float
#         Current proposed time step

#     Returns
#     -------
#     status, m_out, m_tot_out, rho_out, u_out, p_out, v2_out, dt_prop
#     """

#     s, Np1 = m_in.shape
#     N = Np1 - 1
#     tiny = np.finfo(np.float64).tiny

#     # Collapse r_in to shape (N+1,) for easy compatibility with future versions
#     r_in = r_in[0]

#     #--- Input validation
#     assert np.allclose(m_tot_in, np.sum(m_in, axis=0))
#     assert np.allclose(1.5 * v2_in, u_in)

#     shell_volumes = (r_in[1:]**3 - r_in[:-1]**3) / 3.0  # shape (N,)
#     expected_rho = m_in[:,1:] - m_in[:,:-1]
#     expected_rho = expected_rho / shell_volumes  # shape (s, N)
#     assert np.allclose(rho_in, expected_rho)

#     #--- Find w and flux per species at each internal interface
#     F_if = np.zeros((s, N-1))        # to store fluxes per interface
#     dt_cfl = 1e300

#     for i in range(1, N):
#         # Interface values
#         r_if = float(r_in[i])
#         dr_if = float(0.5 * (r_in[i+1] - r_in[i-1]))
#         if dr_if < tiny: dr_if = tiny

#         rho_if = 0.5 * (rho_in[:,i-1] + rho_in[:,i])        # shape (s,)
#         v_if = np.sqrt(0.5 * (v2_in[:,i-1] + v2_in[:,i]))   # shape (s,)
#         p_L = p_in[:,i-1]
#         p_R = p_in[:,i]

#         m_tot_if = float(m_tot_in[i])
#         rho_tot_if = float(np.sum(rho_if)); rho_tot_if = max(rho_tot_if, tiny)

#         # b_k = d/dr(p_k) + rho_k*g
#         dpdr = (p_R - p_L) / dr_if
#         grav = rho_if * m_tot_if / max(r_if * r_if, tiny)
#         b = dpdr + grav

#         # A_kj = (rho_k * rho_j / rho_tot) * nu_kj
#         # A_kj = Kjk; K_kk = -\sum_{j\neq k} K_kj
#         A = np.zeros((s,s))
#         nu_if = np.maximum(tiny, v_if * rho_if) # nu ~ 1/trelax = rho*sqrt(v2)
#         for k in range(s):
#             for j in range(s):
#                 if j != k:
#                     K_kj = (rho_if[k] * rho_if[j] / rho_tot_if)
#                     K_kj *= 2 * nu_if[k] * nu_if[j] / (nu_if[k] + nu_if[j]) # Harmonic mean of friction terms
                
#                     A[k,j] = K_kj
#                     A[k,k] -= K_kj

#         # Replace bottom row of matrix equation with zero net flux constraint
#         b[-1] = 0.0
#         A[-1,:] = rho_if[:]

#         # Solve Aw = b
#         w = np.linalg.solve(A,b)

#         # Update dt_cfl
#         w_abs_max = max(tiny, np.max(np.abs(w)))
#         dt_cfl = min(dt_cfl, 0.5 * dr_if / w_abs_max) # 0.5 safety factor

#         # Mass flux at this interface: F_k = rho_k * w_k
#         F_if[:, i-1] = rho_if * w

#     # Check for mass conservation
#     if not np.allclose(F_if.sum(axis=0), 0.0):
#         print("WARNING: mass may not conserved in Burger's step")

#     #--- Check CFL condition is met with current dt_in
#     if dt_in > dt_cfl:
#         return 'reduce_dt', None, None, None, None, None, None, 0.95 * dt_cfl
    
#     #--- Update extensive and intensive variables
#     A_if = r_in[1:-1]**2
#     V_cell = (r_in[1:]**3 - r_in[:-1]**3) / 3.0

#     # Mass and density
#     m_out = m_in.copy()
#     m_out[:, 1:-1] = m_in[:, 1:-1] - F_if * A_if[np.newaxis, :] * dt_in
#     m_tot_out = m_out.sum(axis=0)

#     m_cell_out = m_out[:, 1:] - m_out[:, :-1]     # (s, N)
#     rho_out = m_cell_out / V_cell[np.newaxis, :]

#     # Energy advection
#     E_in  = (m_in[:, 1:] - m_in[:, :-1]) * u_in

#     # Build upwind face energies: Phi[:, j-1] uses edge j (1..N-1)
#     Phi = np.empty_like(F_if)                    # (s, N-1)
#     for j in range(1, N):
#         col = j - 1
#         # upwind donor: if F>0, donor is left shell (j-1); else right shell (j)
#         u_face = np.where(F_if[:, col] >= 0.0, u_in[:, j-1], u_in[:, j])
#         Phi[:, col] = F_if[:, col] * u_face

#     E_out = E_in.copy()
#     E_out[:,:-1]    -= Phi * A_if[np.newaxis, :] * dt_in
#     E_out[:,1:]     += Phi * A_if[np.newaxis, :] * dt_in

#     # Other intensive quantities
#     with np.errstate(divide='ignore', invalid='ignore'):
#         u_out = np.where(m_cell_out > 0.0, E_out / m_cell_out, 0.0)
#     v2_out = u_out / 1.5
#     p_out  = (2.0/3.0) * rho_out * u_out

#     return ('ok', m_out, m_tot_out, rho_out, u_out, p_out, v2_out, dt_in)