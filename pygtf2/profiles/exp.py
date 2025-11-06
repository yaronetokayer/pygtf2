import numpy as np
from scipy.integrate import quad

def _exp_velocity_integrand(x):
    fac = 0.5 * ( 2.0 - np.exp(-x) * (x**2 + 2.0 * x + 2.0) )
    return 0.5 * np.exp(-x) * fac / x**2

def menc_exp(r):
    
    return 0.5 * ( 2.0 - np.exp(-r) * (r**2 + 2.0 * r + 2.0) )

def sigr_exp(r, prec):
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
        integral, _ = quad(_exp_velocity_integrand, ri_f, np.inf, epsabs=epsabs, epsrel=epsrel)
        out[i] = 2.0 * np.exp(ri_f) * integral

    return out if out.size > 1 else float(out[0])
