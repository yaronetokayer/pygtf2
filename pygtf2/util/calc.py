import numpy as np 
from numba import njit, types, float64, void
from pygtf2.util.interpolate import sum_intensive_loglog_single, interp_intensive_loglog
from pygtf2.profiles.bkg_pot import hernq_static_scalar

CORE_HALF_RHO0 = 0
CORE_SPITZER87 = 1

@njit(void(float64[:], float64[:], float64[:], float64[:], float64[:]), cache=True)
def solve_tridiagonal_thomas(a, b, c, y, x):
    """
    Solve a tridiagonal system Ax = y using the Thomas algorithm.
    This follows the Numerical Recipes convention:
        a[i] * x[i-1] + b[i] * x[i] + c[i] * x[i+1] = y[i]

    For an N x N system, all coefficient arrays have length N.
    The value a[0] is unused, and c[N-1] is unused.

    Parameters
    ----------
    a : ndarray
        Subdiagonal coefficients.
    b : ndarray
        Main diagonal coefficients.
    c : ndarray
        Superdiagonal coefficients.
    y : ndarray
        Right-hand side vector.
    x : ndarray
        Output solution vector, updated in place.
    """
    n = b.size
    gam = np.empty(n, dtype=np.float64)
    gam[0] = 0.0

    bet = b[0]
    x[0] = y[0] / bet

    for i in range(1, n):
        gam[i] = c[i-1] / bet
        bet = b[i] - a[i] * gam[i]
        x[i] = (y[i] - a[i] * x[i-1]) / bet

    for i in range(n - 2, -1, -1):
        x[i] -= gam[i+1] * x[i+1]

@njit((float64[:], float64[:, :], float64[:], float64[:]), fastmath=True, cache=True)
def compute_eta_multi(masses, sigmas, eta_out, err_out):
    """
    Fit sigma proportional to particle_mass**(-eta) at each radius.

    Parameters
    ----------
    masses : ndarray, shape (s,)
        Positive particle masses, not total species masses.
    sigmas : ndarray, shape (s, N)
        Positive one-dimensional velocity dispersions (not their squares).
    eta_out : ndarray, shape (N,)
        Preallocated fitted exponents, overwritten in place. Equal particle
        masses give zero slope.
    err_out : ndarray, shape (N,)
        Preallocated RMS residuals in natural-log space, overwritten in place.

    Returns
    -------
    None
        Results are written to the supplied buffers.
    """
    s, N = sigmas.shape

    # log masses
    x = np.empty(s)
    for i in range(s):
        x[i] = np.log(masses[i])
    x_mean = 0.0
    for i in range(s):
        x_mean += x[i]
    x_mean /= s

    # log sigmas
    y = np.empty((s, N))
    for i in range(s):
        for j in range(N):
            y[i, j] = np.log(sigmas[i, j])
    y_mean = np.zeros(N)
    for j in range(N):
        temp = 0.0
        for i in range(s):
            temp += y[i, j]
        y_mean[j] = temp / s

    # variance of masses
    var = 0.0
    for i in range(s):
        dx = x[i] - x_mean
        var += dx * dx

    # covariance in each radial bin
    for j in range(N):
        cov = 0.0
        for i in range(s):
            cov += (x[i] - x_mean) * (y[i, j] - y_mean[j])

        # Protect against division by zero
        if var > 0.0:
            eta_out[j] = -cov / var
        else:
            eta_out[j] = 0.0

    # compute RMS error
    for j in range(N):
        c = y_mean[j] + eta_out[j] * x_mean
        acc = 0.0
        for i in range(s):
            r = y[i, j] + eta_out[j] * x[i] - c
            acc += r * r
        err_out[j] = np.sqrt(acc / s)

def compute_eta(masses, sigmas):
    """
    Return equipartition exponents and natural-log RMS fit residuals.

    masses contains positive particle masses. sigmas contains positive
    one-dimensional velocity dispersions, shaped (s, N), not v2. Multiple
    species return two arrays of length N. A one-dimensional sigmas input
    or a single species returns the scalar pair (0.0, 0.0).
    """
    masses = np.asarray(masses)
    sigmas = np.asarray(sigmas)

    # Single species case (s = 1)
    if sigmas.ndim == 1:
        # shape is (N,) → single species, one radial array
        return 0.0, 0.0

    # Multi-species case
    s, N = sigmas.shape
    if s == 1:
        # Another possibility: sigmas is (1, N)
        return 0.0, 0.0

    # Prepare output arrays
    eta = np.empty(N, dtype=np.float64)
    err = np.empty(N, dtype=np.float64)

    compute_eta_multi(masses, sigmas, eta, err)
    return eta, err

@njit((float64[:], float64[:, :], float64[:, :],float64,), fastmath=True, cache=True)
def compute_eta_interp(masses, rmid, v2, rmax=0.0):
    r"""
    Compute eta profile for arbitrary number of species.
    Defines a shared grid, computes the interpolated v2,
    and then computes eta for each radial bin.

    eta defined by sigma \propto m^-eta

    Arguments
    ---------
    masses : array-like, shape (s,)
        Particle mass of each species.
    rmid : array-like, shape (s, N)
        Midpoints of radial grid points per species, where v2 is evaluated.
    v2 : ndarray, shape (s, N)
        Square of velocity dispersion for each species.
    rmax : float, optional
        Maximum shared midpoint radius. Nonpositive values use rmid.max().
        A positive value rescales the output point count to int(N*rmax/rmid.max());
        choose a value that leaves at least one point.

    Returns
    -------
    rmid_shared : ndarray, shape (N_shared,)
    eta : ndarray, shape (N_shared,)
    """
    s, N = rmid.shape

    rmin = rmid.min()
    if rmax <= 0.0:
        rmax = rmid.max()
    else:
        N = int(N * rmax / rmid.max())
    rmid_shared = np.empty(N, dtype=np.float64)

    # geometric spacing factor
    if N > 1:
        log_rmin = np.log(rmin)
        log_rmax = np.log(rmax)
        dlog = (log_rmax - log_rmin) / (N - 1)

        for i in range(N):
            rmid_shared[i] = np.exp(log_rmin + i * dlog)
    else:
        # degenerately single-point grid
        rmid_shared[0] = rmin
    
    eta_out = np.zeros(N, dtype=np.float64)
    err_out = np.zeros(N, dtype=np.float64)

    if s == 1:
        return rmid_shared, eta_out

    v2_interp = interp_intensive_loglog(rmid_shared, rmid, v2)

    # square root into temporary array
    sigma_interp = np.empty((s, N), dtype=np.float64)
    for i in range(s):
        for j in range(N):
            sigma_interp[i, j] = np.sqrt(v2_interp[i, j])

    # compute eta
    compute_eta_multi(masses, sigma_interp, eta_out, err_out)

    return rmid_shared, eta_out

@njit(float64(float64, float64[:]), fastmath=True, cache=True)
def add_bkg_pot_scalar(r, bkg_param):
    """
    Return the background enclosed mass at a single radius.

    Parameters
    ----------
    r : float
        Radius at which to evaluate the enclosed background mass.

    bkg_param : ndarray, shape (4,)
        Background parameters:
        (prof, m_par, r_par, x_par).

    Returns
    -------
    float
        Background enclosed mass at r.
    """
    prof = int(bkg_param[0])
    m_par = bkg_param[1]
    r_par = bkg_param[2]
    # x_par = bkg_param[3]  # For future profiles

    if prof == 0:
        # Static Hernquist profile:
        return hernq_static_scalar(r, m_par, r_par)

    return 0.0

@njit(void(float64[:], float64[:], float64[:]), fastmath=True, cache=True)
def add_bkg_pot(r, bkg_param, m_enc):
    """
    Add the background enclosed-mass contribution to m_enc in place.

    Parameters
    ----------
    r : ndarray, shape (N+1,)
        Radii at which to evaluate the enclosed background mass.

    bkg_param : ndarray, shape (4,)
        Background parameters:
        (prof, m_par, r_par, x_par).

    m_enc : ndarray, shape (N+1,)
        Enclosed-mass array. The background contribution is added
        to this array in place.
    """
    for i in range(r.shape[0]):
        m_enc[i] += add_bkg_pot_scalar(r[i], bkg_param)

@njit(void(float64[:], float64[:], float64[:]), fastmath=True, cache=True,)
def add_bkg_K(r, bkg_param, K_on_k):
    """
    Add dM_bkg/dr to K_on_k using a centered finite difference.

    Parameters
    ----------
    r : ndarray, shape (N+1,)
        Edge radii of the current species.

    bkg_param : ndarray, shape (4,)
        Background-potential parameters.

    K_on_k : ndarray, shape (N+1,)
        Existing dM_other/dr evaluated on the current species grid.
        The background contribution is added in place.

    Notes
    -----
    Only interior values K_on_k[1:-1] are used by the
    re-virialization Jacobian.
    """
    fd_frac = 1.0e-5
    Np1 = r.shape[0]

    for i in range(1, Np1 - 1):
        ri = r[i]

        if ri > 0.0:
            rp = ri * (1.0 + fd_frac)
            rm = ri * (1.0 - fd_frac)

            mp = add_bkg_pot_scalar(rp, bkg_param)
            mm = add_bkg_pot_scalar(rm, bkg_param)

            K_on_k[i] += (mp - mm) / (2.0 * fd_frac * ri)

@njit(float64(float64, float64), fastmath=True, cache=True,)
def calc_r_c_spitzer87(rho_0, v2_0):
    """
    Spitzer-like core radius.

    In current code units:
        r_core = sqrt(v2_0 / rho_0)
    """
    return np.sqrt(v2_0 / rho_0)

@njit(float64(float64[:, :], float64[:, :]), fastmath=True, cache=True)
def calc_r_c_half_rho0(rmid, rho):
    """
    Find the first radius where the total density falls to half of
    its value at the innermost radial point:

        rho_tot(r_core) = 0.5 * rho_0

    The total density at arbitrary radius is evaluated using the same
    log-log interpolation used elsewhere in the code.
    """
    s, N = rmid.shape

    r_0 = np.min(rmid[:, 0])
    rho_0 = sum_intensive_loglog_single(r_0, rmid, rho)
    rho_target = 0.5 * rho_0

    # Collect all species midpoint radii to obtain a robust set of
    # locations at which to search for the first crossing.
    radii = np.empty(s * N, dtype=np.float64)

    k = 0
    for j in range(s):
        for i in range(N):
            radii[k] = rmid[j, i]
            k += 1

    radii.sort()

    # Find the first bracket containing the half-density crossing.
    r_lo = r_0
    rho_lo = rho_0

    found = False
    r_hi = r_lo

    for k in range(radii.size):
        r = radii[k]

        if r <= r_lo:
            continue

        rho_r = sum_intensive_loglog_single(r, rmid, rho)

        if rho_r <= rho_target:
            r_hi = r
            found = True
            break

        r_lo = r
        rho_lo = rho_r

    if not found:
        return np.nan

    # Solve within the bracket. Geometric midpoint is natural for
    # a logarithmic radial grid.
    for _ in range(50):
        r_mid = np.sqrt(r_lo * r_hi)

        rho_mid = sum_intensive_loglog_single(
            r_mid, rmid, rho
        )

        if rho_mid > rho_target:
            r_lo = r_mid
        else:
            r_hi = r_mid

    return np.sqrt(r_lo * r_hi)

@njit(float64(float64[:, :], float64[:, :], float64[:, :], types.int64,), cache=True,)
def calc_r_c(rmid, rho, v2, core_def):
    """
    Return a dimensionless core radius for the selected definition.

    rmid, rho, and v2 have shape (s, N). CORE_HALF_RHO0 finds the radius
    where total density first falls to half its innermost value.
    CORE_SPITZER87 uses sqrt(v20/rho0), with total innermost density and
    density-weighted innermost v2. Other core_def values raise ValueError.
    """
    if core_def == CORE_HALF_RHO0:

        return calc_r_c_half_rho0(rmid, rho)

    elif core_def == CORE_SPITZER87:
        r0      = np.min(rmid[:,0])
        rho0    = sum_intensive_loglog_single(r0, rmid, rho)
        v20     = sum_intensive_loglog_single(r0, rmid, rho*v2) / rho0 # Mass-weighted average
        return calc_r_c_spitzer87(rho0, v20)

    else:
        raise ValueError("Unrecognized core-radius definition")

@njit(float64(float64, float64[:, :], float64[:, :], float64[:, :]), fastmath=True, cache=True)
def calc_v2_c(r_c, r, m, v2):
    """
    Mass-weighted 1D velocity dispersion squared inside r_c,
    summed over all species.

    Parameters
    ----------
    r_c : float
        Core radius in simulation units.
    r : (s, N+1)
        Shell-interface radii.
    m : (s, N+1)
        Enclosed mass for each species.
    v2 : (s, N)
        Shell 1D velocity dispersion squared.
    """
    s, N = v2.shape

    mass_c = 0.0
    mv2_c = 0.0

    for j in range(s):
        for i in range(N):

            r_in = r[j, i]
            r_out = r[j, i + 1]

            if r_in >= r_c:
                break

            dm = m[j, i + 1] - m[j, i]

            if r_out <= r_c:
                # Entire shell lies inside the core
                dm_c = dm

            else:
                # r_c cuts through this shell.
                # Assuming uniform shell density, mass fraction scales as r^3.
                frac = (
                    (r_c**3 - r_in**3)
                    / (r_out**3 - r_in**3)
                )
                dm_c = frac * dm

            mass_c += dm_c
            mv2_c += dm_c * v2[j, i]

            if r_out > r_c:
                break

    return mv2_c / mass_c

@njit(types.Tuple((float64[:], float64[:]))(float64, float64[:, :], float64[:, :], float64[:, :]), cache=True, fastmath=True,)
def calc_core_species_quantities(r_c, r, m, v2):
    """
    Compute core mass and mass-weighted v2 for each species.

    For a shell intersected by r_c, the enclosed fraction of the shell
    mass is computed assuming constant density within the shell.

    Parameters
    ----------
    r_c : float
        Core radius.
    r : ndarray, shape (s, N+1)
        Shell-interface radii for each species.
    m : ndarray, shape (s, N+1)
        Enclosed mass for each species.
    v2 : ndarray, shape (s, N)
        Shell velocity dispersion squared for each species.

    Returns
    -------
    m_c_species : ndarray, shape (s,)
        Mass of each species enclosed within r_c.
    v2_c_species : ndarray, shape (s,)
        Mass-weighted velocity dispersion squared of each species
        within r_c.
    """
    s, N = v2.shape

    m_c_species = np.zeros(s, dtype=np.float64)
    v2_c_species = np.zeros(s, dtype=np.float64)

    for k in range(s):
        m_c_k = 0.0
        mv2_c_k = 0.0

        for i in range(N):
            r_in = r[k, i]
            r_out = r[k, i + 1]

            if r_in >= r_c:
                break

            dm = m[k, i + 1] - m[k, i]

            # If r_c cuts through this shell, include only the
            # corresponding fraction of its volume/mass.
            if r_out > r_c:
                frac = (
                    (r_c**3 - r_in**3)
                    / (r_out**3 - r_in**3)
                )
                dm *= frac

            m_c_k += dm
            mv2_c_k += dm * v2[k, i]

            if r_out >= r_c:
                break

        m_c_species[k] = m_c_k

        if m_c_k > 0.0:
            v2_c_species[k] = mv2_c_k / m_c_k
        else:
            v2_c_species[k] = np.nan

    return m_c_species, v2_c_species

@njit(float64[:](float64[:], float64[:], float64[:]), fastmath=True, cache=True)
def mass_fraction_radii(r_edges, m_edges, fracs):
    """
    Interpolate radii enclosing sorted fractions of the final enclosed mass.

    Parameters
    ----------
    r_edges, m_edges : ndarray, shape (N,)
        Corresponding increasing radial edges and nondecreasing enclosed
        masses. Supply equal-length arrays with at least two entries.
    fracs : ndarray, shape (M,)
        Fractions in nondecreasing order. The search only moves forward, so
        unsorted fractions are not supported. Inputs are not validated.

    Returns
    -------
    ndarray, shape (M,)
        Radii found by linear interpolation in enclosed mass. Nonpositive
        final mass yields NaNs. Targets above the final mass use the outer
        radius; targets below the first mass extrapolate the first interval.
        A flat mass interval uses its left radius.

    Examples
    --------
    >>> import numpy as np
    >>> r = np.array([0.0, 1.0, 2.0])
    >>> m = np.array([0.0, 10.0, 20.0])
    >>> mass_fraction_radii(r, m, np.array([0.25, 0.5, 1.0]))
    array([0.5, 1.0, 2.0])
    """
    # m_edges should be enclosed mass at edges, with m_edges[-1] = species total
    m_tot = m_edges[-1]
    if m_tot <= 0:
        return np.full(fracs.size, np.nan)
    target = fracs * m_tot
    out = np.empty(fracs.size)
    # linear-in-radius search & interpolation on edges
    j = 0
    for i, mt in enumerate(target):
        while j+1 < m_edges.size and m_edges[j+1] < mt:
            j += 1
        if j+1 == m_edges.size:
            out[i] = r_edges[-1]
        else:
            m0, m1 = m_edges[j], m_edges[j+1]
            r0, r1 = r_edges[j], r_edges[j+1]
            t = 0.0 if m1 == m0 else (mt - m0) / (m1 - m0)
            out[i] = r0 + t * (r1 - r0)
    return out

@njit(float64(float64[:, :], float64[:, :], float64[:, :]), fastmath=True, cache=True)
def calc_r50_spread(r, m, r50evo):
    """
    Compute a simple segregation metric:
        spread = (max(r50) - min(r50)) / mean(r50)

    Also update S_k = r_{50,k}(t)/r_{50,k}(0), in place.

    Arguments
    ---------
    r : array-like, shape (s, N+1)
        Radii arrays per species
    m : array-like, shape (s, N+1)
        Mass arrays per species
    r50evo : array-like, shape (s, 2)
        [k,0] is initial r_50 for species k and [k,1] is the S_k value

    Returns
    -------
    spread : float
        Dimensionless measure of segregation (0 = none)
    """
    s, _ = r.shape

    if s < 2:
        return 0.0

    frac = np.array([0.5])

    r50_min = 1.0e308
    r50_max = -1.0e308
    r50_sum = 0.0

    for k in range(s):
        r_50_k      = mass_fraction_radii(r[k], m[k], frac)[0]
        r50_sum += r_50_k

        if r_50_k < r50_min:
            r50_min = r_50_k
        if r_50_k > r50_max:
            r50_max = r_50_k

        r50evo[k,1]    = r_50_k/r50evo[k,0]
        
    r50_mean = r50_sum / s

    return (r50_max - r50_min) / r50_mean
