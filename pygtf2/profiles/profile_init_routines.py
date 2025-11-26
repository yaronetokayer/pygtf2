import numpy as np
from pygtf2.profiles.nfw import menc_nfw, sigr_nfw
from pygtf2.profiles.abg import menc_abg, sigr_abg
from pygtf2.profiles.truncated_nfw import menc_trunc, sigr_trunc
from pygtf2.profiles.exp import menc_exp, sigr_exp
from pygtf2.profiles.king import menc_king, sigr_king

def _as_f64(x):
    """
    Helper function to ensure double point precision for all input values
    """
    a = np.asarray(x, dtype=np.float64)
    return a if a.ndim else float(a)

def menc(r, init, prec, **kwargs):
    """
    Compute enclosed mass at radius r, in units of ms.

    Parameters
    ----------
    r : float or array-like
        Radius in units of scale radius (r / r_s).
    prec : PrecisionParams
        The simulation PrecisionParams object
    init: InitParams
        Initial profile parameters object.

    Returns
    -------
    float or ndarray
        Enclosed mass at r, normalized by Mvir.
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
    else:
        raise ValueError(f"Unsupported profile type: {profile}")

def sigr(r, init, prec, bkg_param, **kwargs):
    """
    Compute radial velocity dispersion squared v^2(r).

    Parameters
    ----------
    r : float or array-like
        Radius in units of scale radius (r / r_s).
    prec : PrecisionParams
        The simulation PrecisionParams object
    init: InitParams
        Initial profile parameters object.
    bkg_param : np.ndarray
        Parameters for background potential.

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
    else:
        raise ValueError(f"Unsupported profile type: {profile}")
