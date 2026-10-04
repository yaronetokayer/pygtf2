import numpy as np
from pygtf2.profiles.nfw import menc_nfw, sigr_nfw
from pygtf2.profiles.abg import menc_abg, sigr_abg
from pygtf2.profiles.truncated_nfw import menc_trunc, sigr_trunc
from pygtf2.profiles.exp import menc_exp, sigr_exp
from pygtf2.profiles.king import menc_king, sigr_king
from pygtf2.profiles.etnfw import menc_etnfw, sigr_etnfw

def _as_f64(x):
    """
    Helper function to ensure double precision for all input values

    """
    a = np.asarray(x, dtype=np.float64)
    return a if a.ndim else float(a)

def menc(r, init, prec, **kwargs):
    """
    Compute enclosed mass at radius r, in units of ms.

    Parameters
    ----------
    r : one-dimensional array-like
        Radii in units of scale radius (r / r_s). Some profile backends do
        not accept scalar input; a one-dimensional array works across profiles.
    prec : PrecisionParams
        The simulation PrecisionParams object
    init: InitParams
        Initial profile parameters object.

    **kwargs
        Profile-specific lookup tables and options forwarded to King or
        truncated-NFW routines (including grid for sigr).

    Returns
    -------
    float or ndarray
        Enclosed mass at r, in the selected profile's dimensionless mass normalization
        (m_s for State initialization), before multiplying by species fraction.
    """
    r = _as_f64(r)
    profile = init.prof
    if profile == "nfw":
        return menc_nfw(r)
    elif profile == "truncated_nfw":
        return menc_trunc(r, prec, **kwargs)
    elif profile == "abg":
        return menc_abg(r, init, prec)
    elif profile == "exp":
        return menc_exp(r)
    elif profile == "king":
        return menc_king(r, prec, **kwargs)
    elif profile == "etnfw":
        return menc_etnfw(r)
    else:
        raise ValueError(f"Unsupported profile type: {profile}")

def sigr(r, init, prec, bkg_param, **kwargs):
    """
    Compute radial velocity dispersion squared v^2(r).

    Parameters
    ----------
    r : one-dimensional array-like
        Radii in units of scale radius (r / r_s). Some profile backends do
        not accept scalar input; a one-dimensional array works across profiles.
    prec : PrecisionParams
        The simulation PrecisionParams object
    init: InitParams
        Initial profile parameters object.
    bkg_param : np.ndarray
        Parameters for background potential.

    **kwargs
        Profile-specific lookup tables and options forwarded to King or
        truncated-NFW routines (including grid for sigr).

    Returns
    -------
    float or ndarray
        Velocity dispersion squared.
    """
    r = _as_f64(r)
    profile = init.prof
    if profile == "nfw":
        return sigr_nfw(r, prec, bkg_param)
    elif profile == "truncated_nfw":
        return sigr_trunc(r, prec, bkg_param, **kwargs)
    elif profile == "abg":
        return sigr_abg(r, init, prec, bkg_param)
    elif profile == "exp":
        return sigr_exp(r, prec, bkg_param)
    elif profile == "king":
        return sigr_king(r, prec, bkg_param, **kwargs)
    elif profile == "etnfw":
        return sigr_etnfw(r, prec, bkg_param)
    else:
        raise ValueError(f"Unsupported profile type: {profile}")
