import numpy as np
from scipy.integrate import quad
from pygtf2.util.calc import add_bkg_pot_scalar

def _etnfw_velocity_integrand(x, bkg_param):
    """
    rho(x) * M(x) / x^2

    rho(r) = (Mtot / 4*pi*r_s^3) * (r_s / r) * e^{-r/r_s} 

    e.g., https://doi.org/10.1093/mnras/stab1215

    Units:
        Mtot = 1
        r_s = 1
        rho_unit = Mtot / (4*pi*r_s^3)

    rho(x) = exp(-x) / x
    M(x)   = 1 - (1 + x) exp(-x)  
    """
    fac = 1.0 - (1.0 + x) * np.exp(-x)

    if bkg_param[0] != -1:
        fac += add_bkg_pot_scalar(x, bkg_param)

    return np.exp(-x) * fac / x**3

def menc_etnfw(r):
    """
    M(r) = 1 - (1 + r) exp(-r)

    Dimensionless enclosed mass, with Mtot = 1 and r_s = 1.
    """
    return 1.0 - (1.0 + r) * np.exp(-r)

def sigr_etnfw(r, prec, bkg_param):
    """
    Velocity dispersion squared at radius r in units of v0^2 = G Mtot / r_s.

    Solves the isotropic Jeans equation:

        sigma_r^2(r) = 1/rho(r) * integral_r^inf rho(x) M(x) / x^2 dx

    Parameters
    ----------
    r : float or ndarray
        Radius in units of r_s.
    prec : PrecisionParams
        The simulation PrecisionParams object.
    bkg_param : np.ndarray
        Parameters for background potential.

    Returns
    -------
    v2 : float or ndarray
        Radial velocity dispersion squared.
    """
    epsabs = float(prec.epsabs)
    epsrel = float(prec.epsrel)

    r = np.asarray(r, dtype=np.float64)
    out = np.empty(r.shape, dtype=np.float64)

    for i, ri in enumerate(r):
        ri_f = float(ri)
        integral, _ = quad(
            _etnfw_velocity_integrand, 
            ri_f, 
            np.inf, 
            args=(bkg_param,), 
            epsabs=epsabs, 
            epsrel=epsrel
            )
        
        out[i] = ri_f * np.exp(ri_f) * integral

    return out if out.size > 1 else float(out[0])
