import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import interp1d
from numba import njit

@njit
def df(e):
    """
    Full (untruncated) distribution function f(e) for the NFW profile.

    Parameters
    ----------
    e : float or ndarray
        Energy variable (0 < e < 1)

    Returns
    -------
    float or ndarray
        Value of the distribution function.
    """
    F0 = 9.1968e-2
    q = -2.7419
    p1, p2, p3, p4 = 0.3620, -0.5639, -0.0859, -0.4912

    e2 = 1.0 - e
    pp = p1 * e + p2 * e**2 + p3 * e**3 + p4 * e**4

    fac1 = e**1.5 / e2**2.5
    fac2 = (-np.log(e) / e2)**q
    fac3 = np.exp(pp)

    return F0 * fac1 * fac2 * fac3

@njit
def integrand_for_rho(z, phi, Zt, Ft):
    if z > phi:
        return 0.0
    e = z + Zt
    return (df(e) - Ft) * np.sqrt(2.0 * (phi - z))

def rho(phi, init, prec):
    """
    Compute density rho(phi) using Eddington-like inversion of the DF.

    Parameters
    ----------
    phi : float
        Gravitational potential at a radius.
    init: InitParams
        Initial profile parameters object.
    prec : PrecisionParams
        The simulation PrecisionParams object

    Returns
    -------
    rho : float
        Density corresponding to the given potential.
    """

    Zt = float(init.Zt)
    Ft = df(Zt)

    epsabs = float(prec.epsabs)
    epsrel = float(prec.epsrel)

    result, _ = quad(integrand_for_rho, 0.0, float(phi), 
                     args=(float(phi), Zt, float(Ft)), epsabs=epsabs, epsrel=epsrel)
    
    return float(4.0 * np.pi * result)

def generate_rho_lookup(init, prec, chatter, n_points=10000, phi_min=1e-7):
    """
    Generate an interpolated function for rho(phi) for use in later functions.

    Parameters
    ----------
    init: InitParams
        Initial profile parameters object.
    prec : PrecisionParams
        The simulation PrecisionParams object
    chatter: bool
        Chatter flag.
    n_points : int, optional
        Number of points used to generate the interpolated function.
    phi_min : float, optional
        Maximum potential value to consider.

    Returns
    -------
    rho_interp : interp1d
        Interpolated function for rho(phi).
    """
    if chatter:
        INDENT = ' ' * 8
        print(f"{INDENT}Generating lookup for rho(phi)...")

    phi_max = float(1.0 - init.Zt - 1e-4)

    # Precompute rho over a grid of phi
    phi_grid = np.linspace(float(phi_min), phi_max, n_points, dtype=np.float64)  # choose phi_max ~ initial phi0
    rho_grid = np.array([rho(phi, init, prec) for phi in phi_grid], dtype=np.float64)
    rho_interp = interp1d(phi_grid, rho_grid, bounds_error=False, fill_value=0.0)

    return rho_interp

@njit
def potential(r, Zt):
    """
    Gravitational potential for an NFW profile.

    Parameters
    ----------
    r : float or ndarray
        Radius in units of r_s.
    Zt : float
        Truncation energy parameter.

    Returns
    -------
    phi : float or ndarray
        Potential at radius r (dimensionless).
    """
    return np.log(1.0 + r) / r - Zt

def integrate_potential(init, grid, chatter, rho_interp):
    """
    Computes the potential profile for a truncated NFW halo by integrating outward
    until the potential drops to zero. Also determines the outer truncation radius.

    Parameters
    ----------
    init: InitParams
        Initial profile parameters object.
    grid : GridParams
        The simulation GridParams object.
    chatter: bool
        Chatter flag.
    rho_interp : interp1d
        Interpolated function for rho(phi).

    Returns
    -------
    rcut : float
        Truncation radius (in units of r_s).
    rmax_new : float
        Updated value of grid.rmax (ensures rmax < rcut).
    pot_interp : interp1d
        Interpolated potential function.
    rad : ndarray
        Radial grid points where potential was computed.
    pot_vals : ndarray
        Corresponding potential values at those radial points.
    """

    if chatter:
        INDENT = ' ' * 8
        print(f"{INDENT}Computing potential profile for truncated NFW halo...")
    r_min = float(grid.rmin) / 2.0
    eps = 1e-6  # A small number used for finite differences
    Zt = float(init.Zt)
    deltaP = -50.0 * float(init.deltaP)
    Nstep = 10

    # Step 1: Compute initial log-derivative of potential
    r_lo = (1.0 - eps) * r_min
    r_hi = (1.0 + eps) * r_min
    pot_lo = potential(r_lo, Zt)
    pot_hi = potential(r_hi, Zt)
    pot_init = potential(r_min, Zt)
    dpot_dlogr_init = (pot_hi - pot_lo) / (r_hi - r_lo)

    # Step 2: Prepare initial values for ODE integration, assuming linearity
    y = np.array([pot_init, dpot_dlogr_init], dtype=np.float64)  # y = [phi, dphi/dlogr]
    dr = deltaP / y[1]
    r1 = r_min
    r2 = r1 + dr

    rad = [float(r1)]
    pot_vals = [float(y[0])]

    r_last_print = 0.0

    # Step 3: Integrate until potential crosses zero
    while y[0] > 0.0:
        # Only print if r has changed by at least 0.5 since last print
        if chatter and (len(rad) == 1 or abs(r1 - r_last_print) >= 1.5):
                print(f"\r{INDENT}Integrating Poisson equation outward: r = {r1:.6f}, phi = {y[0]:.6f}", end='', flush=True)
                r_last_print = r1
        step_size = (r2 - r1) / Nstep

        def dphi_dr(r, y):
            phi, dphi_dr = y
            Q = float(rho_interp(phi))
            return [dphi_dr, -Q - 2.0 * dphi_dr / r]
        
        sol = solve_ivp(
            dphi_dr,
            (float(r1), float(r2)),
            y,
            method='RK45',
            t_eval=[float(r2)],
            max_step=float(step_size),
            rtol=1e-5,
            atol=1e-8
        )

        y = sol.y[:, -1].astype(np.float64, copy=False)
        r1 = float(r2)
        rad.append(r2)
        pot_vals.append(float(y[0]))

        dr = deltaP / y[1]
        if dr < 0.0:
            raise RuntimeError("dr became negative during integration.")

        dr = min(dr, 0.01)
        r2 = r1 + dr

    if chatter:
        print(f"\r{INDENT}Integrating Poisson equation outward: r = {r1:.6f}, phi = {y[0]:.6f}")

    # Step 4: truncate and return values
    rcut = float(r1)
    rmax_new = float(min(grid.rmax, 0.99 * rcut))

    rad_arr = np.asarray(rad, dtype=np.float64)
    pot_arr = np.asarray(pot_vals, dtype=np.float64)
    pot_interp = interp1d(rad, pot_vals, kind='cubic', fill_value=0.0, bounds_error=False)

    return rcut, rmax_new, pot_interp, rad_arr, pot_arr

def _density_times_r2_trunc(r, pot_rad, pot_interp, rho_interp):
    """
    Integrand for computing enclosed mass in truncated NFW profile:
    rho(r) * r^2, where rho is determined either analytically (inner region)
    or from interpolation of the potential and evaluation of the DF (outer region).

    Parameters
    ----------
    r : float
        Radius at which to evaluate the integrand (in units of r_s).
    pot_rad : ndarray
        Radii from numerical integration of the potential.
    pot_interp : scipy interp1d
        Interpolated numerically integrated potential function.
    rho_interp : scipy interp1d
        Interpolated numerically integrated density function.

    Returns
    -------
    float
        Value of the integrand rho(r) * r^2.
    """
    r = float(r)
    if r < float(pot_rad[0]): # the interpolated potential is not defined below the first radial point
        density = 1.0 / (r * (1.0 + r)**2)
    else:
        pot = float(pot_interp(r))
        density = float(rho_interp(pot))

    return density * r**2

def menc_trunc(r, prec, chatter=True, pot_rad=None, pot_interp=None, rho_interp=None): 
    """
    Enclosed mass for a truncated NFW profile computed via numerical integration.

    Parameters
    ----------
    r : float or ndarray
        Radius in units of r_s.
    prec : PrecisionParams
        The simulation PrecisionParams object.
    chatter: bool
        Chatter flag.
    pot_rad : ndarray
        Radii from numerical integration of the potential.
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

    out = np.empty(r.shape, dtype=np.float64)

    for i, ri in enumerate(r):
        val, _ = quad(
            _density_times_r2_trunc,
            0.0,
            float(ri),
            args=(pot_rad, pot_interp, rho_interp,),
            epsabs=epsabs,
            epsrel=epsrel,
            limit=200
        )
        out[i] = val
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
        rcut,
        pot_rad,
        pot_interp,
        rho_interp,
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
    rcut : float
        Numerically calculated cutoff radius for truncated NFW potential.
    pot_rad : ndarray
        Radii from numerical integration of the potential.
    pot_interp : scipy interp1d
        Interpolated numerically integrated potential function.
    rho_interp : scipy interp1d
        Interpolated numerically integrated density function.
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
    
    r_lo = float(grid.rmin) / 2.0 - 1e-4
    rgrid = np.geomspace(r_lo, float(rcut), int(n_points), dtype=np.float64)
    
    pot = pot_interp(rgrid).astype(np.float64, copy=False)
    mask = rgrid < float(pot_rad[0])
    density = np.zeros_like(rgrid, dtype=np.float64)
    density[mask]  = 1.0 / (rgrid[mask] * (1.0 + rgrid[mask])**2)
    density[~mask] = rho_interp(pot[~mask]).astype(np.float64, copy=False)
    menc = menc_trunc(
        rgrid, prec, chatter=False, pot_rad=pot_rad, pot_interp=pot_interp, rho_interp=rho_interp
        ).astype(np.float64, copy=False)
    integrand_vals = menc * density / rgrid**2
    
    f_interp = interp1d(rgrid, integrand_vals, bounds_error=False, fill_value=0.0)
    
    return f_interp

def sigr_trunc(
        r,
        prec,
        chatter=True,
        grid=None,
        rcut=None,
        pot_rad=None,
        pot_interp=None,
        rho_interp=None,
        ):
    """ 
    v^2 profile for truncated NFW halo.

    Parameters
    ----------
    r : float or ndarray
        Radius in units of r_s.
    prec : PrecisionParams
        The simulation PrecisionParams object.
    chatter: bool
        Chatter flag.
    grid : GridParams
        The simulation GridParams object.
    rcut : float
        Numerically calculated cutoff radius for truncated NFW potential.
    pot_rad : ndarray
        Radii from numerical integration of the potential.
    pot_interp : scipy interp1d
        Interpolated numerically integrated potential function.
    rho_interp : scipy interp1d
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
        prec, grid, chatter, rcut, pot_rad, pot_interp, rho_interp
        )

    for i, ri in enumerate(r):
        if ri > float(rcut):
            out[i] = 0.0
            continue
        else:
            if ri < float(pot_rad[0]): # the interpolated potential is not defined below the first radial point
                density = 1.0 / (ri * (1.0 + ri)**2)
            else:
                pot = float(pot_interp(ri))
                density = float(rho_interp(pot))
            if ri < 1.0:
                local_epsabs, local_epsrel = 1e-5, 1e-3
            else:
                local_epsabs, local_epsrel = epsabs, epsrel
            integral, _ = quad(
                integrand,
                float(ri),
                float(rcut),
                epsabs=float(local_epsabs),
                epsrel=float(local_epsrel),
                limit=200
            )

            out[i] = integral / density
        if chatter:
            INDENT = ' ' * 8
            print(f"\r{INDENT}Computing v2: r = {ri:.3f}, v2 = {out[i]:.3f}", end='', flush=True)
    if chatter:
        print("")  # Finalize output line

    return out if out.size > 1 else float(out[0])

# NOT USED IN CURRENT VERSION
# @njit
# def df_trunc(e, Zt, Ft):
#     """
#     Truncated distribution function f_trunc(e) = f(e + Zt) - Ft.

#     Parameters
#     ----------
#     e : float or ndarray
#         Energy variable for integration, in [0, phi]
#     Zt : float
#         Energy shift defining the truncation threshold.
#     Ft : float
#         Distribution function floor used in truncation.

#     Returns
#     -------
#     float or ndarray
#         Truncated distribution function.
#     """
#     return df(e + Zt) - Ft