import numpy as np
from scipy.integrate import quad

def fNFW(r):
    """
    Analytic NFW mass profile: M(<r) / Mvir

    Parameters
    ----------
    r : float or ndarray
        Radius in units of r_s.

    Returns
    -------
    M_enc : float or ndarray
        Enclosed mass as a fraction of Mvir.
    """
    
    r = np.asarray(r, dtype=np.float64)

    return np.log(1.0 + r) - r / (1.0 + r)

def _nfw_velocity_integrand(x):
    fac = np.log(1.0 + x) - x / (1.0 + x)
    return fac / (x**3 * (1.0 + x)**2)

def menc_nfw(r):
    return fNFW(r)

def sigr_nfw(r, prec):
    """
    Velocity dispersion squared at radius r (in units of v0^2).

    Parameters
    ----------
    r : float or ndarray
        Radius in units of r_s.
    prec : PrecisionParams
        The simulation PrecisionParams object

    Returns
    -------
    v2 : float or ndarray
        Velocity dispersion squared.
    """
    epsabs = float(prec.epsabs)
    epsrel = float(prec.epsrel)

    r = np.asarray(r, dtype=np.float64)
    out = np.empty(r.shape, dtype=np.float64)

    for i, ri in enumerate(r):
        ri_f = float(ri)
        integral, _ = quad(_nfw_velocity_integrand, ri_f, np.inf, epsabs=epsabs, epsrel=epsrel)
        out[i] = ri * (1.0 + ri)**2 * integral

    return out if out.size > 1 else float(out[0])
