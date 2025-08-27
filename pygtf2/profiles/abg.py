from scipy.integrate import quad
import numpy as np

def chi(config):
    """
    Computes the chi parameter used in the ABG profile normalization.

    Parameters
    ----------
    config : Config
        The simulation configuration object. Must use an ABG profile.

    Returns
    -------
    float
        The value of chi.
    """
    alpha = float(config.init.alpha)
    beta = float(config.init.beta)
    gamma = float(config.init.gamma)
    expo = (beta - gamma) / alpha
    epsabs = float(config.prec.epsabs)
    epsrel = float(config.prec.epsrel)

    def chi_integrand(x):
        return x**(2.0 - gamma) / (1.0 + x**alpha)**expo

    result, _ = quad(chi_integrand, 0.0, 1e4, epsabs=epsabs, epsrel=epsrel)
    return float(result)

def _abg_jeans_mass_integrand(x, alpha, beta, gamma):
    """
    Mass integrand in the spherical Jeans equation for ABG profile.
    """
    return x**(2.0 - gamma) / (1.0 + x**alpha)**((beta - gamma) / alpha)

def _abg_velocity_integrand(x, alpha, beta, gamma, epsabs, epsrel):
    """
    Integrand for the velocity dispersion from the Jeans equation.
    """
    chi_integrand = lambda y: y**(2.0 - gamma) / (1.0 + y**alpha)**((beta - gamma) / alpha)
    chi_x, _ = quad(chi_integrand, 0.0, float(x), epsabs=epsabs, epsrel=epsrel)
    rho_x = x**(-gamma) / (1.0 + x**alpha)**((beta - gamma) / alpha)
    return rho_x * chi_x / x**2

def menc_abg(r, config):
    """
    Enclosed mass profile for the alpha-beta-gamma profile.

    Parameters
    ----------
    r : float or ndarray
        Radius in units of r_s.
    config : Config
        The simulation configuration object (must use ABG profile).

    Returns
    -------
    M_enc : float or ndarray
        Enclosed mass in units of Mvir.
    """
    alpha = float(config.init.alpha)
    beta  = float(config.init.beta)
    gamma = float(config.init.gamma)
    epsabs = float(config.prec.epsabs)
    epsrel = float(config.prec.epsrel)

    r = np.asarray(r, dtype=np.float64)
    out = np.empty(r.shape, dtype=np.float64)

    for i, ri in enumerate(r):
        integral, _ = quad(_abg_jeans_mass_integrand, 0.0, float(ri), 
                           args=(alpha, beta, gamma), epsabs=epsabs, epsrel=epsrel)
        out[i] = integral

    return out if out.size > 1 else float(out[0])

def sigr_abg(r, config):
    """
    Velocity dispersion squared for alpha-beta-gamma profile at radius r.

    Parameters
    ----------
    r : float or ndarray
        Radius in units of r_s.
    config : Config
        The simulation configuration object (must use ABG profile).

    Returns
    -------
    v2 : float or ndarray
        Velocity dispersion squared.
    """
    alpha = float(config.init.alpha)
    beta  = float(config.init.beta)
    gamma = float(config.init.gamma)

    epsabs = float(config.prec.epsabs)
    epsrel = float(config.prec.epsrel)

    r = np.asarray(r, dtype=np.float64)
    out = np.empty(r.shape, dtype=np.float64)

    for i, ri in enumerate(r):
        integrand = lambda x: _abg_velocity_integrand(x, alpha, beta, gamma, epsabs, epsrel)
        integral, _ = quad(integrand, float(ri), np.inf, epsabs=epsabs, epsrel=epsrel)
        rho_ri = ri**(-gamma) / (1.0 + ri**alpha)**((beta - gamma) / alpha)
        out[i] = integral / rho_ri

    return out if out.size > 1 else float(out[0])