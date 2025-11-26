import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import interp1d
from numba import njit
from pygtf2.util.calc import add_bkg_pot

"""
Code units in King profile:
r_s : core radius (r0)
mtot : total mass
rho_s : mtot / (4 * pi * r_s^3)

x = r / r_s; W(x) = Psi(r) / sigma^2; nu(W) = rho(Psi) / rho_0

In King, r0^2 = 9 sigma^2 / (4 pi rho_0); rho_0 = central density

Throughout, we assume sigma parameter of King profile is 1.0
"""

@njit
def df(e):
    """
    Full (unnormalized) distribution function f(e) for the King profile.
    Assumes velocity parameter sigma = 1.0

    Arguments
    ---------
    e : float
        relative energy per unit mass (E = psi - v^2/2); bound if e > 0

    Returns
    -------
    float
        Value of the distribution function.
    """
    if e <= 0.0:
        return 0.0

    return np.exp(e) - 1.0

@njit
def integrand_for_rho(v, Psi):
    e = Psi - 0.5*v**2
    return df(e)*v**2

def rho(Psi, prec):
    """
    Dimensionless density nu(w) = rho(Psi) / rho0; rho0 = central density
    w = Psi / sigma^2, dimensionless relative potential
    This gives rho(Psi)

    Arguments
    ---------
    Psi : float
        Dimensionless relative potential.  With sigma = 1.0, w = Psi
    prec : PrecisionParams
        The simulation PrecisionParams object
    """
    epsabs = float(prec.epsabs)
    epsrel = float(prec.epsrel)

    vmax = np.sqrt(2.0 * Psi)

    rho_int, _ = quad(integrand_for_rho, 0.0, vmax, 
                      args=(float(Psi)), epsabs=epsabs, epsrel=epsrel)
    
    return float(4.0 * np.pi * rho_int)

def generate_nu_lookup(init, prec, chatter, n_points=10000, w_min=1e-10):
    """
    Dimensionless density nu(w) = rho(Psi) / rho0; rho0 = central density
    w = Psi / sigma^2, dimensionless relative potential

    Arguments
    ---------
    init: InitParams
        Initial profile parameters object.
    prec : PrecisionParams
        The simulation PrecisionParams object
    chatter: bool
        Chatter flag.
    n_points : int, optional
        Number of points used to generate the interpolated function.
    w_min : float, optional
        Minimum potential value to consider.
    """
    if chatter:
        INDENT = ' ' * 8
        print(f"{INDENT}Generating lookup for nu(W)...")

    W0 = init.W0
    w_max = W0 * (1.1) # Some buffer

    # Central density
    rho0 = rho(W0, prec)

    # Precompute rho over a grid of w
    w_grid = np.linspace(float(w_min), w_max, n_points, dtype=np.float64)  # choose phi_max ~ initial phi0
    nu_grid = np.array([rho(w, prec) for w in w_grid], dtype=np.float64) / rho0
    nu_interp = interp1d(w_grid, nu_grid, bounds_error=False, fill_value=0.0)
    
    return nu_interp

def integrate_W_king(init, grid, chatter, nu_interp, deltaW_frac=-0.05):
    """
    Integrate the dimensionless King potential W(x) outward using

        W''(x) + (2/x) W'(x) = -9 * nu(W(x))

    until W crosses zero (tidal radius).
    x = r/r_s (for us, r_s is the core radius)

    Computes the potential profile for a King halo by integrating outward
    until the potential drops to zero. Also determines the outer truncation radius.
    
    Arguments
    ---------
    init: InitParams
        Initial profile parameters object.
    grid : GridParams
        The simulation GridParams object.
    chatter: bool
        Chatter flag.
    nu_interp : interp1d
        Interpolated function for rho(phi).
    deltaW_frac : float, optional
        Target fractional change in W per step, e.g. -0.05.

    Returns
    -------
    rcut : float
        Truncation radius (in units of r_s).
    rmax_new : float
        Updated value of grid.rmax (ensures rmax < rcut).
    W_interp : interp1d
        Cubic interpolant W(x).
    x_arr : ndarray
        Radii where W was computed (dimensionless).
    W_arr : ndarray
        Corresponding W(x) values. 
    """

    if chatter:
        INDENT = ' ' * 8
        print(f"{INDENT}Integrating dimensionless King Poisson equation...")

    W0 = float(init.W0)

    # Start slightly inside the first grid radius, in units of r0
    x_min = float(grid.rmin) / 10.0
    Nstep = 10

    # Taylor expansion at the centre:
    # W''(0) = -3 * nu(W0), with nu(W0)=1 in your normalization.
    nu0 = float(nu_interp(W0))
    W2_0 = -3.0 * nu0

    W_init = W0 + 0.5 * W2_0 * x_min**2   # W(x_min)
    dWdx_init = W2_0 * x_min              # W'(x_min)

    # State vector y = [W, dW/dx] at x1 = x_min
    x1 = x_min
    y = np.array([W_init, dWdx_init], dtype=np.float64)

    # Target change in W per step; W' < 0, so deltaW < 0 ⇒ dx > 0
    deltaW = deltaW_frac * W0
    if y[1] == 0.0:
        raise RuntimeError("Initial dW/dx is zero; cannot set step size.")
    dx = deltaW / y[1]
    dx = abs(dx)
    dx = min(dx, 0.001)
    x2 = x1 + dx

    x_list = [float(x1)]
    W_list = [float(y[0])]
    x_last_print = 0.0

    # RHS of ODE
    def dW_dx(x, y):
        W, dWdx = y
        if W <= 0.0:
            # Past the tidal radius; hold the solution.
            return [0.0, 0.0]
        # nu(W) = rho/ rho0; nu_interp is defined for 0 <= W <= W0.
        nu_val = float(nu_interp(W))
        ddWdx = -9.0 * nu_val - 2.0 * dWdx / x
        return [dWdx, ddWdx]
    
    # Integrate until W crosses zero
    while y[0] > 0.0 and x1 < float(grid.rmax) * 10.0:

        if chatter and (len(x_list) == 1 or abs(x1 - x_last_print) >= 0.1):
            print(f"\r{INDENT}Integrating Poisson equation outward: x = {x1:.6f}, W = {y[0]:.6f}", end='', flush=True)
            x_last_print = x1

        step_size = (x2 - x1) / Nstep

        sol = solve_ivp(
            dW_dx,
            (float(x1), float(x2)),
            y,
            method='RK45',
            t_eval=[float(x2)],
            max_step=float(step_size),
            rtol=1e-5,
            atol=1e-8
        )

        y = sol.y[:, -1].astype(np.float64, copy=False)
        x1 = float(x2)
        x_list.append(x1)
        W_list.append(float(y[0]))

        if y[0] <= 0.0:
            break

        # New step size: keep |ΔW| ≈ |deltaW|
        if y[1] == 0.0:
            break
        dx = deltaW / y[1]
        dx = abs(dx)
        dx = min(dx, 0.005)
        x2 = x1 + dx

    if chatter:
        print(f"\r{INDENT}Integrating Poisson equation outward: x = {x1:.6f}, W = {y[0]:.6f}")

    xt = float(x1)  # tidal radius in units of r0
    xmax_new = float(min(grid.rmax, 0.99 * xt))

    x_arr = np.asarray(x_list, dtype=np.float64)
    W_arr = np.asarray(W_list, dtype=np.float64)
    W_interp = interp1d(x_arr, W_arr, kind='cubic',
                        fill_value=0.0, bounds_error=False)

    return xt, xmax_new, W_interp, x_arr, W_arr

def _nu_times_x2_king(r, w_interp, nu_interp):
    """
    Integrand for computing enclosed mass in King profile:
    nu(W(x)) * x^2, where nu is determined 
    from interpolation of the potential and evaluation of the DF.

    Parameters
    ----------
    r : float
        Radius at which to evaluate the integrand (in units of r_s).
    w_interp : scipy interp1d
        Interpolated numerically integrated potential function.
    nu_interp : scipy interp1d
        Interpolated numerically integrated density function.

    Returns
    -------
    float
        Value of the integrand rho(r) * r^2.
    """
    r = float(r)
    pot = float(w_interp(r))
    density = float(nu_interp(pot))

    return density * r**2

def menc_king(r, prec, chatter=True, rt=10.0, w_interp=None, nu_interp=None, **_):
    """
    Enclosed mass for a King profile computed via numerical integration.

    Parameters
    ----------
    r : float or ndarray
        Radius in units of r_s.
    rt : float
        Truncation radius in units of r_s.
    prec : PrecisionParams
        The simulation PrecisionParams object.
    chatter: bool
        Chatter flag.
    pot_interp : scipy interp1d
        Interpolated numerically integrated potential function.
    rho_interp : scipy interp1d
        Interpolated numerically integrated density function.

    Returns
    -------
    M_enc : float or ndarray
        Enclosed mass in units of Mvir.
    """
    r = np.atleast_1d(np.asarray(r, dtype=np.float64))
    epsabs = float(prec.epsabs)
    epsrel = float(prec.epsrel)

    rmax = rt*1.05
    norm, _ = quad(             # Compute Mtot, which is the mass unit
        _nu_times_x2_king,
        0.0,
        float(rmax),
        args=(w_interp, nu_interp,),
        epsabs=epsabs,
        epsrel=epsrel,
        limit=500
        )

    out = np.empty(r.shape, dtype=np.float64)

    for i, ri in enumerate(r):
        val, _ = quad(
            _nu_times_x2_king,
            0.0,
            float(ri),
            args=(w_interp, nu_interp,),
            epsabs=epsabs,
            epsrel=epsrel,
            limit=500
        )
        out[i] = val / norm
        if chatter:
            INDENT = ' ' * 8
            print(f"\r{INDENT}Computing Menc: r = {ri:.3f}, m = {out[i]:.3f}", end='', flush=True)
    if chatter:
        print("")  # Finalize output line

    return out if out.size > 1 else float(out[0])

def generate_sigr_integrand_lookup(
        prec,
        grid,
        chatter, 
        rt,
        w_interp,
        nu_interp,
        bkg_param,
        n_points=1000
        ):
    """
    Generate an interpolated function for the velocity dispersion squared
    as a function of radius for the truncated NFW profile.

    Parameters
    ----------
    prec : PrecisionParams
        The simulation PrecisionParams object.
    grid : GridParams
        The simulation GridParams object.
    chatter: bool
        Chatter flag.
    rt : float
        Numerically calculated truncation radius for King potential.
    w_interp : scipy interp1d
        Interpolated numerically integrated potential function.
    nu_interp : scipy interp1d
        Interpolated numerically integrated density function.
    bkg_param : np.ndarray
        Parameters for background potential.
    n_points : int
        Number of points to use in the lookup table.

    Returns
    -------
    sigr_interp : interp1d
        Interpolated function for velocity dispersion squared.
    """
    if chatter:
        INDENT = ' ' * 8
        print(f"{INDENT}Generating lookup for v2 integrand...")
    
    r_lo = (float(grid.rmin) / 2.0) * 0.9
    rgrid = np.geomspace(r_lo, float(rt), int(n_points), dtype=np.float64)

    pot = w_interp(rgrid).astype(np.float64, copy=False)
    density = nu_interp(pot).astype(np.float64, copy=False)
    menc = menc_king(
        rgrid, prec, chatter=False, rt=rt, w_interp=w_interp, nu_interp=nu_interp
        ).astype(np.float64, copy=False)
    if bkg_param[0] != -1:
        menc += add_bkg_pot(rgrid, bkg_param)
    integrand_vals = menc * density / rgrid**2

    f_interp = interp1d(rgrid, integrand_vals, bounds_error=False, fill_value=0.0)

    return f_interp

def sigr_king(
        r,
        prec,
        bkg_param,
        chatter=True,
        grid=None,
        rt=None,
        w_interp=None,
        nu_interp=None,
        **_,
        ):
    """ 
    v^2 profile for King halo.

    Parameters
    ----------
    r : float or ndarray
        Radius in units of r_s.
    prec : PrecisionParams
        The simulation PrecisionParams object.
    bkg_param : np.ndarray
        Parameters for background potential.
    chatter: bool
        Chatter flag.
    grid : GridParams
        The simulation GridParams object.
    rt : float
        Numerically calculated truncation radius for Kingpotential.
    w_interp : scipy interp1d
        Interpolated numerically integrated potential function.
    nu_interp : scipy interp1d
        Interpolated numerically integrated density function.

    Returns
    -------
    float or ndarray
        Velocity dispersion squared at radius r.
    """
    r = np.asarray(r, dtype=np.float64)
    epsabs = prec.epsabs
    epsrel = prec.epsrel
    out = np.empty(r.shape, dtype=np.float64)

    integrand = generate_sigr_integrand_lookup(
        prec, grid, chatter, rt, w_interp, nu_interp, bkg_param
        )
    
    for i, ri in enumerate(r):
        if ri > float(rt):
            out[i] = 0.0
            continue

        pot = float(w_interp(ri))
        density = float(nu_interp(pot))
        integral, _ = quad(
            integrand,
            float(ri),
            float(rt),
            epsabs=float(epsabs),
            epsrel=float(epsrel),
            limit=200
        )

        out[i] = integral / density
        if chatter:
            INDENT = ' ' * 8
            print(f"\r{INDENT}Computing v2: r = {ri:.3f}, v2 = {out[i]:.3f}", end='', flush=True)
    
    if chatter:
        print("")  # Finalize output line

    return out if out.size > 1 else float(out[0])