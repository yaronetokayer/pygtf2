from scipy.integrate import quad
import numpy as np
from pygtf2.util.calc import add_bkg_pot_scalar

def chi(prec, init):
    """
    Computes the chi parameter used in the ABG profile normalization.

    Parameters
    ----------
    prec : PrecisionParams
        The simulation PrecisionParams object
    init: InitParams
        Initial profile parameters object for the most massive component.  Must use an ABG profile.

    Returns
    -------
    float
        The value of chi.
    """
    alpha = float(init.alpha)
    beta = float(init.beta)
    gamma = float(init.gamma)
    expo = (beta - gamma) / alpha
    epsabs = float(prec.epsabs)
    epsrel = float(prec.epsrel)

    def chi_integrand(x):
        return x**(2.0 - gamma) / (1.0 + x**alpha)**expo

    result, _ = quad(chi_integrand, 0.0, 1e4, epsabs=epsabs, epsrel=epsrel)
    return float(result)

def _abg_jeans_mass_integrand(x, alpha, beta, gamma):
    """
    Mass integrand in the spherical Jeans equation for ABG profile.
    """
    return x**(2.0 - gamma) / (1.0 + x**alpha)**((beta - gamma) / alpha)

def _abg_velocity_integrand(x, alpha, beta, gamma, epsabs, epsrel, bkg_param):
    """
    Integrand for the velocity dispersion from the Jeans equation.
    """
    chi_integrand = lambda y: y**(2.0 - gamma) / (1.0 + y**alpha)**((beta - gamma) / alpha)
    chi_x, _ = quad(chi_integrand, 0.0, float(x), epsabs=epsabs, epsrel=epsrel)
    rho_x = x**(-gamma) / (1.0 + x**alpha)**((beta - gamma) / alpha)
    if bkg_param[0] != -1:
        chi_x += add_bkg_pot_scalar(x, bkg_param)
    return rho_x * chi_x / x**2

def menc_abg(r, init, prec):
    """
    Enclosed mass profile for the alpha-beta-gamma profile.

    Parameters
    ----------
    r : float or ndarray
        Radius in units of r_s.
    prec : PrecisionParams
        The simulation PrecisionParams object
    init: InitParams
        Initial profile parameters object.

    Returns
    -------
    M_enc : float or ndarray
        Enclosed mass in units of Mvir.
    """
    alpha = float(init.alpha)
    beta  = float(init.beta)
    gamma = float(init.gamma)
    epsabs = float(prec.epsabs)
    epsrel = float(prec.epsrel)

    r = np.asarray(r, dtype=np.float64)
    out = np.empty(r.shape, dtype=np.float64)

    for i, ri in enumerate(r):
        integral, _ = quad(_abg_jeans_mass_integrand, 0.0, float(ri), 
                           args=(alpha, beta, gamma), epsabs=epsabs, epsrel=epsrel)
        out[i] = integral

    return out if out.size > 1 else float(out[0])

def sigr_abg(r, init, prec, bkg_param):
    """
    Velocity dispersion squared for alpha-beta-gamma profile at radius r.

    Parameters
    ----------
    r : float or ndarray
        Radius in units of r_s.
    init: InitParams
        Initial profile parameters object.
    prec : PrecisionParams
        The simulation PrecisionParams object.
    bkg_param : np.ndarray
        Parameters for background potential.

    Returns
    -------
    v2 : float or ndarray
        Velocity dispersion squared.
    """
    alpha = float(init.alpha)
    beta  = float(init.beta)
    gamma = float(init.gamma)

    epsabs = float(prec.epsabs)
    epsrel = float(prec.epsrel)

    r = np.asarray(r, dtype=np.float64)
    out = np.empty(r.shape, dtype=np.float64)

    for i, ri in enumerate(r):
        integrand = lambda x: _abg_velocity_integrand(x, alpha, beta, gamma, epsabs, epsrel, bkg_param=bkg_param)
        integral, _ = quad(integrand, float(ri), np.inf, epsabs=epsabs, epsrel=epsrel)
        rho_ri = ri**(-gamma) / (1.0 + ri**alpha)**((beta - gamma) / alpha)
        out[i] = integral / rho_ri

    return out if out.size > 1 else float(out[0])