import numpy as np
from pygtf2.profiles.nfw import menc_nfw, sigr_nfw
from pygtf2.profiles.abg import menc_abg, sigr_abg
from pygtf2.profiles.truncated_nfw import menc_trunc, sigr_trunc

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
    profile = init.profile
    if profile == "nfw":
        return menc_nfw(r)
    elif profile == "truncated_nfw":
        return menc_trunc(r, prec, **kwargs)
    elif profile == "abg":
        return menc_abg(r, init, prec)
    else:
        raise ValueError(f"Unsupported profile type: {profile}")

def sigr(r, init, prec, **kwargs):
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

    Returns
    -------
    float or ndarray
        Velocity dispersion squared.
    """
    r = _as_f64(r)
    profile = init.profile
    if profile == "nfw":
        return sigr_nfw(r, prec)
    elif profile == "truncated_nfw":
        return sigr_trunc(r, prec, **kwargs)
    elif profile == "abg":
        return sigr_abg(r, init, prec)
    else:
        raise ValueError(f"Unsupported profile type: {profile}")
