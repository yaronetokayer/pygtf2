import numpy as np 
from numba import njit, types, float64
from pygtf2.util.interpolate import sum_intensive_loglog_single, interp_intensive_loglog, sum_extensive_loglog_single, interp_species_loglog_single
from pygtf2.profiles.bkg_pot import hernq_static
 
@njit((float64[:], float64[:, :], float64[:], float64[:]), fastmath=True, cache=True)
def compute_eta_multi(masses, sigmas, eta_out, err_out):
    """
    Compute eta for arbitrary number of species using log-log regression.
    
    Arguments
    ---------
    masses : array-like, shape (s,)
        Mass of each species.
    sigmas : array-like, shape (s, N)
        Velocity dispersions for each species.
    Return arrays are preallocated
        
    Returns
    -------
    eta_out : float or ndarray of shape (N,)
        Best-fit equipartition exponent(s).
    error_out : float or ndarray of shape (N,)
        RMS residual in log-log space; measure of deviation from power law.
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
    User-facing function.
    If only one species: eta = 0, error = 0.
    Otherwise: call numba backend.
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
    and the computes eta for each radial bin.

    eta defined by sigma \propto m^-eta
    
    Arguments
    ---------
    masses : array-like, shape (s,)
        Mass of each species.
    rmid : array-like, shape (s, N)
        Midpoints of radial grid points per species, where v2 is evaluated.
    v2 : array-like, shape (N,) or (s, N)
        Square of velocity dispersion for each species.
    rmidmax : float
        Maximum value for interpolated rmid

    Returns
    -------
    rmid_shared : ndarray, shape (N,)
    eta : ndarray, shape (N,)
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

@njit(float64[:](float64[:], float64[:]), fastmath=True, cache=True)
def add_bkg_pot(r, bkg_param):
    """
    Add a background potential to the enclosed-mass array.

    Parameters
    ----------
    r : array_like, shape (N+1,)
        Edge radii.
    bkg_param : sequence of length 4
        Background parameters: (prof, m_par, r_par, x_par). Only prof==0 is implemented.

    Returns
    -------
    m_add : ndarray, shape (N+1,)
        The the background potential contribution to be added.
    """
    prof = int(bkg_param[0])
    m_par = bkg_param[1]
    r_par = bkg_param[2]
    # x_par = bkg_param[3] For future profiles

    if prof == 0:
        m_add = hernq_static(r, m_par, r_par)

    else:
        m_add = np.zeros(r.shape, dtype=np.float64)
    
    return m_add

@njit(float64(float64, float64[:]), fastmath=True, cache=True)
def add_bkg_pot_scalar(r, bkg_param):
    """
    Calls add_bkg_pot on a scalar value
    """
    r_arr = np.array([r], dtype=np.float64)
    return add_bkg_pot(r_arr, bkg_param)[0]

@njit(types.Tuple((float64, float64, float64))(float64[:, :], float64[:, :], float64[:, :]), fastmath=True, cache=True)
def calc_rho_v2_r_c(rmid, rho, v2):
    """
    Computes central values of system at smallest non-zero radial point.
    rho, v2, and core radius.
    Use core radius definition of Spitzer (1987)
    r_c^2 = 3*v2_c / (4 * pi * G * rho_c)

    Arguments
    ---------
    rmid : ndarray, shape (s, N)
        Midpoint radii per species.
    rho : ndarray, shape (s, N)
        Shell densities per species.
    v2 : ndarray, shape (s, N)
        Shell velocity dispersion squared per species.

    Returns
    -------
    rho_c : float
        Central density
    v2_c : float
        Central square of velocity dispersion
    r_c : float
        Core radius
    """
    r0 = np.min(rmid[:,0])

    rho_c = sum_intensive_loglog_single(r0, rmid, rho)

    v2_c = sum_intensive_loglog_single(r0, rmid, rho*v2) / rho_c # Mass-weighted average

    # Should be good estimate within order unity, based on exact result from a King profile
    r_c = np.sqrt( v2_c / rho_c )  

    return float(rho_c), float(v2_c), float(r_c)

@njit(float64[:](float64[:], float64[:], float64[:]), fastmath=True, cache=True)
def mass_fraction_radii(r_edges, m_edges, fracs):
    """
    Compute radii at which a given set of mass fractions of the total enclosed mass are reached.

    Parameters
    ----------
    r_edges : array_like, shape (N,)
        Radii of the radial edges (grid points). Expected to be monotonic (typically increasing).
    m_edges : array_like, shape (N,)
        Cumulative (enclosed) mass evaluated at each radius in ``r_edges``.
        The final element ``m_edges[-1]`` is interpreted as the total mass of the species.
    fracs : array_like, shape (M,)
        Mass fractions of the total mass at which the radius is requested (typically in [0, 1]).
        Can be any real values; see Notes for behavior outside the nominal range.

    Returns
    -------
    numpy.ndarray, shape (M,)
        Array of radii corresponding to each requested mass fraction in ``fracs``.
        If the total mass (``m_edges[-1]``) is <= 0, an array of NaNs with the same shape as ``fracs`` is returned.

    Raises
    ------
    ValueError
        If the input arrays have incompatible lengths or if ``m_edges`` and ``r_edges`` do not have at least two points.
        (The implementation assumes at least two edges for meaningful interpolation.)

    Notes
    -----
    - The function computes target enclosed masses as ``fracs * m_tot`` where ``m_tot = m_edges[-1]`` and finds the interval
      in ``m_edges`` that brackets each target. It then linearly interpolates in radius between the two bracketing edges.
    - The search advances monotonically through the mass-edge array for successive targets (amortized O(N + M) cost),
      so performance is best when ``fracs`` are provided in non-decreasing order. The function still works for unsorted ``fracs``.
    - The routine assumes ``m_edges`` is non-decreasing and corresponds to the same ordering as ``r_edges``. No internal sorting is performed.
    - Behavior for targets outside the range of ``m_edges``:
      - Targets >= total mass map to ``r_edges[-1]``.
      - Targets less than the first edge mass are handled by linear interpolation using the first interval and may produce radii
        smaller than ``r_edges[0]`` (i.e., backward extrapolation).
    - Exact matches to an edge mass return the corresponding edge radius.

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
    s_k : array-like, shape (s, 2)
        [k,0] is initial r_50 for species k and [k,1] is the the S_k value

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

@njit((float64[:, :], float64[:, :], float64, float64[:]), fastmath=True, cache=True)
def compute_rc_frac(r, m, r_c, rc_frac):
    """
    Compute f_k = M_k(<r_c)/Mtot(<r_c) for each species, in place

    Arguments
    ---------
    r : array-like, shape (s, N+1)
        Radii arrays per species
    m : array-like, shape (s, N+1)
        Mass arrays per species
    r_c : float
        Core radius
    f_k : array-like, shape (s,)
        f_k for each species
    """
    s, _ = r.shape

    denom = sum_extensive_loglog_single(r_c, r, m)
    for k in range(s):
        rc_frac[k] = interp_species_loglog_single(r_c, r, m, k) / denom