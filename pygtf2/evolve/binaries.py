import numpy as np
from numba import njit, float64, int64, types

@njit(
    types.void(
        float64[:, :],   # rho
        float64[:, :],   # v2
        float64[:, :],   # r
        float64,         # dt
        float64[:],      # m_part
        int64,           # n_particles
    ),
    cache=True,
    fastmath=True,
)
def binary_heating(
    rho: np.ndarray,
    v2: np.ndarray,
    r: np.ndarray,
    dt: float,
    m_part: np.ndarray,
    n_particles: int,
) -> None:
    """
    Apply heating from three-body binary formation using the
    multicomponent prescription of Lee, Fahlman & Richer (1991).

    The total local binary heating rate is evaluated on the common
    refinement of the individual species grids.  The generated energy
    is distributed among species in proportion to their local mass
    densities, following Lee et al., and is then conservatively mapped
    back onto each species' Lagrangian grid.

    Absolute particle masses in code units are determined from
    the total particle number and the component masses implied by rho
    and r.

    Modifies v2 in place.

    Arguments
    ---------
    rho : ndarray
        Species mass densities, with shape (n_species, n_cell).
    v2 : ndarray
        Species one-dimensional velocity dispersions squared, with
        shape (n_species, n_cell).  Updated in place.
    r : ndarray
        Species radial cell edges, with shape
        (n_species, n_cell + 1).
    dt : float
        Current timestep.
    m_part : ndarray
        Relative particle masses m_k / m_1, with shape (n_species,).
    n_particles : int
        Total number of particles represented by the system.

    Returns
    -------
    None
        The v2 array is updated in place.
    """

    # Lee et al. binary-heating coefficient in the code's
    # 4*pi-absorbed density convention.
    C_bin = 0.57

    n_species, n_cell = rho.shape

    # ------------------------------------------------------------------
    # Determine the absolute particle masses in code units.
    #
    # m_1 = sum_k (M_k / mu_k) / N.
    # ------------------------------------------------------------------

    mass_over_mu = 0.0

    for k in range(n_species):

        mass_k = 0.0

        for j in range(n_cell):

            dr3 = r[k, j + 1]**3 - r[k, j]**3

            # rho uses the 4*pi-absorbed density convention.
            mass_k += rho[k, j] * dr3 / 3.0

        mass_over_mu += mass_k / m_part[k]

    m1_abs = mass_over_mu / n_particles

    m_abs = np.empty(n_species, dtype=np.float64)

    for k in range(n_species):
        m_abs[k] = m_part[k] * m1_abs

    # ------------------------------------------------------------------
    # Single-species case.
    #
    # For one component, no overlap grid is needed.  The Lee et al.
    # prescription reduces directly to
    #
    #     du/dt = C_bin * v2_c * m^3 * rho^2 / v2^(9/2),
    #
    # where v2_c is the central one-dimensional velocity dispersion
    # squared.  Since u = (3/2) v2,
    #
    #     dv2/dt = (2/3) du/dt.
    #
    # The coefficient C_bin = 0.57 already accounts for the code's
    # 4*pi-absorbed density convention.
    # ------------------------------------------------------------------

    if n_species == 1:

        # With a single species, its absolute particle mass is simply
        # the total component mass divided by the particle number.
        m = m_abs[0]

        # Store the central dispersion before modifying v2.  The zeroth
        # cell extends from r = 0 to the first radial edge.
        v2_c = v2[0, 0]

        if v2_c <= 0.0:
            raise ValueError(
                "Central v2 must be positive for binary heating."
            )

        for j in range(n_cell):

            rho_j = rho[0, j]
            v2_j = v2[0, j]

            # Empty cells contribute no binary heating.
            if rho_j <= 0.0:
                continue

            if v2_j <= 0.0:
                raise ValueError(
                    "v2 must be positive wherever rho is positive."
                )

            # Lee et al. three-body binary heating rate per unit mass:
            #
            #     du/dt = C_bin * v2_c
            #             * m^3 * rho^2 / v2^(9/2).
            #
            # Writing the denominator this way avoids a fractional
            # power operation:
            #
            #     v2^(9/2) = v2^4 * sqrt(v2).
            du_dt = (
                C_bin
                * v2_c
                * m**3
                * rho_j**2
                / (v2_j**4 * np.sqrt(v2_j))
            )

            # u = (3/2) v2.
            v2[0, j] += (2.0 / 3.0) * dt * du_dt

        return

    # ------------------------------------------------------------------
    # Compute the mass-density-weighted central 1D velocity dispersion.
    # ------------------------------------------------------------------

    rho_c = 0.0
    rho_v2_c = 0.0

    for k in range(n_species):
        rho_c += rho[k, 0]
        rho_v2_c += rho[k, 0] * v2[k, 0]

    sigma_c2 = rho_v2_c / rho_c

    # ------------------------------------------------------------------
    # Construct the common refinement of all species grids.
    #
    # We collect all radial cell edges and sort them.  Adjacent distinct
    # edges then define intervals over which rho and v2 are constant for
    # every species under the cell-centered representation used here.
    # ------------------------------------------------------------------

    n_edge = n_species * (n_cell + 1)

    edges = np.empty(n_edge, dtype=np.float64)

    q = 0

    for k in range(n_species):
        for j in range(n_cell + 1):
            edges[q] = r[k, j]
            q += 1

    edges.sort()

    # For each species, keep track of the cell containing the current
    # common-grid interval.  Because we sweep outward in radius, these
    # indices only ever increase.
    j_parent = np.zeros(n_species, dtype=np.int64)

    # Accumulate the binary energy-generation rate deposited into each
    # original Lagrangian shell.  This has units of energy / time.
    heat_rate_shell = np.zeros_like(v2)

    # ------------------------------------------------------------------
    # Lee et al. write the three-body heating rate using physical
    # density.  In terms of 1D velocity dispersions,
    #
    #     Edot_3b = C_b G^5 sigma_c^2
    #               [sum_k rho_phys,k m_k / sigma_k^3]^3,
    #
    # with C_b = 90.
    #
    # Our stored density is rho = 4*pi*rho_phys.  After dividing the
    # total heating rate by rho_phys,tot to obtain the specific heating
    # rate, this changes the coefficient by a factor (4*pi)^(-2).
    #
    # G = 1 in code units.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Evaluate the binary heating field on the common refinement.
    # ------------------------------------------------------------------

    for q in range(n_edge - 1):

        r_lo = edges[q]
        r_hi = edges[q + 1]

        # Duplicate grid edges produce zero-width intervals.
        if r_hi <= r_lo:
            continue

        rho_tot = 0.0
        heating_sum = 0.0

        # --------------------------------------------------------------
        # Evaluate
        #
        #     sum_k rho_k m_k / sigma_k^3
        #
        # and the total density on this overlap interval.
        # --------------------------------------------------------------

        for k in range(n_species):

            # Advance to the cell containing r_lo.
            while (
                j_parent[k] < n_cell
                and r[k, j_parent[k] + 1] <= r_lo
            ):
                j_parent[k] += 1

            j = j_parent[k]

            # This species does not extend to the current interval.
            if j >= n_cell:
                continue

            # This also allows species grids with different radial
            # extents.  A species contributes only where it is defined.
            if r[k, j] > r_lo or r[k, j + 1] < r_hi:
                continue

            rho_k = rho[k, j]
            v2_k = v2[k, j]

            # Zero-density cells make no contribution.
            if rho_k <= 0.0:
                continue

            # A populated cell must have a positive velocity dispersion.
            if v2_k <= 0.0:
                raise ValueError(
                    "v2 must be positive wherever rho is positive."
                )

            rho_tot += rho_k

            # sigma^3 = (v2)^(3/2).
            heating_sum += (
                rho_k
                * m_abs[k]
                / (v2_k * np.sqrt(v2_k))
            )

        # There is nothing to heat outside the system.
        if rho_tot <= 0.0:
            continue

        # --------------------------------------------------------------
        # Lee et al. distribute the generated energy among components
        # in proportion to rho_k.  Consequently every component has the
        # same local specific heating rate,
        #
        #     h_3b = Edot_3b / rho_tot.
        #
        # Here rho is our 4*pi-absorbed density, hence C_bin above.
        # --------------------------------------------------------------

        h_3b = (
            C_bin
            * sigma_c2
            * heating_sum**3
            / rho_tot
        )

        # In the code density convention,
        #
        #     dm = rho * d(r^3) / 3.
        #
        # The same overlap volume applies to every species present in
        # this interval.
        dvol_code = (r_hi**3 - r_lo**3) / 3.0

        # --------------------------------------------------------------
        # Deposit the heating into every species shell intersecting this
        # overlap interval.
        #
        # Since h_3b is energy / mass / time,
        #
        #     dEdot_k = dm_k * h_3b.
        #
        # Accumulating the extensive energy rate first makes the mapping
        # back onto the separate Lagrangian grids conservative.
        # --------------------------------------------------------------

        for k in range(n_species):

            j = j_parent[k]

            if j >= n_cell:
                continue

            if r[k, j] > r_lo or r[k, j + 1] < r_hi:
                continue

            rho_k = rho[k, j]

            if rho_k <= 0.0:
                continue

            dm = rho_k * dvol_code

            heat_rate_shell[k, j] += dm * h_3b

    # ------------------------------------------------------------------
    # Convert the accumulated shell heating into a specific-energy
    # change and update v2 in place.
    #
    # For an isotropic system,
    #
    #     u = (3/2) v2,
    #
    # so
    #
    #     dv2 = (2/3) du.
    # ------------------------------------------------------------------

    for k in range(n_species):

        for j in range(n_cell):

            dr3 = r[k, j + 1]**3 - r[k, j]**3

            mass_shell = rho[k, j] * dr3 / 3.0

            if mass_shell <= 0.0:
                continue

            du_dt = heat_rate_shell[k, j] / mass_shell

            v2[k, j] += (2.0 / 3.0) * dt * du_dt

# import numpy as np
# from pygtf2.util.interpolate import sum_intensive_loglog
# from numba import njit, types, float64

# @njit(types.Tuple((float64[:,:], float64[:,:], float64))(float64[:,:], float64[:,:], float64[:,:], float64),
#     fastmath=True,cache=True)
# def binaries_heating(rmid, rho, v2, dt):
#     s, N = rmid.shape
    
#     # Constants, see e.g., Bettwieser (1985MNRAS.215..499B)
#     pow = 2.0
#     c = 1.0e-9 # arbitrary right now
#     l = 1.0

#     v2_new  = np.zeros_like(v2, dtype=np.float64)
#     p_new   = np.zeros_like(v2, dtype=np.float64)
#     eps_max = 0.0

#     # per species update
#     for k in range(s):
#         if s == 1:
#             rho_tot = rho[k]
#             v2_tot  = v2[k]
#         else:
#             rmidk   = rmid[k]
#             rho_tot = sum_intensive_loglog(rmidk, rmid, rho)
#             v2_tot  = sum_intensive_loglog(rmidk, rmid, rho*v2) / rho_tot   # Mass-weighted
#         eps         = c * rho_tot**pow / v2_tot**(l/2.0)
#         v2k_new     = v2[k] + eps * dt
#         v2_new[k,:] = v2k_new
#         p_new[k,:]  = rho[k] * v2k_new
#         for e in eps:
#             if e > eps_max:
#                 eps_max = e
    
#     return v2_new, p_new, eps_max