import numpy as np
from scipy.integrate import quad
from pygtf2.util.calc import add_bkg_pot_scalar

def _exp_velocity_integrand(x, bkg_param):
    """
    rho(x) * M(x) / x^2
    rho(x) = (Mtot / 8*pi*r_s^3) * e^{-r/r_s} 
    M(r) = (Mtot / 2) * (2 - e^{-r/r_s} * ((r/r_s)^2 + 2*r/r_s + s))
    Mtot / 4*pi*r_s^3 = rho_s is set to 1
    Mtot is set to 1
    r_s is set to 1
    """
    fac = 0.5 * ( 2.0 - np.exp(-x) * (x**2 + 2.0 * x + 2.0) )
    if bkg_param[0] != -1:
        fac += add_bkg_pot_scalar(x, bkg_param)
    return 0.5 * np.exp(-x) * fac / x**2

def menc_exp(r):
    """
    M(r) = (Mtot / 2) * (2 - e^{-r/r_s} * ((r/r_s)^2 + 2*r/r_s + s))
    Mtot and r_s are set to 1
    """
    return 0.5 * ( 2.0 - np.exp(-r) * (r**2 + 2.0 * r + 2.0) )

def sigr_exp(r, prec, bkg_param):
    """
    Velocity dispersion squared at radius r (in units of v0^2).

    Parameters
    ----------
    r : float or ndarray
        Radius in units of r_s.
    prec : PrecisionParams
        The simulation PrecisionParams object
    bkg_param : np.ndarray
        Parameters for background potential.

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
        integral, _ = quad(_exp_velocity_integrand, ri_f, np.inf, args=(bkg_param,), epsabs=epsabs, epsrel=epsrel)
        out[i] = 2.0 * np.exp(ri_f) * integral

    return out if out.size > 1 else float(out[0])
