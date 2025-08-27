import numpy as np
from pygtfcode.profiles.nfw import menc_nfw, sigr_nfw
from pygtfcode.profiles.abg import menc_abg, sigr_abg
from pygtfcode.profiles.truncated_nfw import menc_trunc, sigr_trunc

def _as_f64(x):
    """
    Helper function to ensure double point precision for all input values
    """
    a = np.asarray(x, dtype=np.float64)
    return a if a.ndim else float(a)

def menc(r, state, **kwargs):
    """
    Compute enclosed mass at radius r, in units of Mvir.

    Parameters
    ----------
    r : float or array-like
        Radius in units of scale radius (r / r_s).
    state : State
        The simulation state object.

    Returns
    -------
    float or ndarray
        Enclosed mass at r, normalized by Mvir.
    """
    r = _as_f64(r)
    profile = state.config.init.profile
    if profile == "nfw":
        return menc_nfw(r)
    elif profile == "truncated_nfw":
        return menc_trunc(r, state, **kwargs)
    elif profile == "abg":
        return menc_abg(r, state.config)
    else:
        raise ValueError(f"Unsupported profile type: {profile}")

def sigr(r, state):
    """
    Compute radial velocity dispersion squared v^2(r).

    Parameters
    ----------
    r : float or array-like
        Radius in units of scale radius (r / r_s).
    state : State
        The simulation state object.

    Returns
    -------
    float or ndarray
        Velocity dispersion squared.
    """
    r = _as_f64(r)
    profile = state.config.init.profile
    if profile == "nfw":
        return sigr_nfw(r, state.config)
    elif profile == "truncated_nfw":
        return sigr_trunc(r, state)
    elif profile == "abg":
        return sigr_abg(r, state.config)
    else:
        raise ValueError(f"Unsupported profile type: {profile}")


# from pygtfcode.profiles.nfw import fNFW
# # from pygtfcode.profiles.truncated_nfw import toint4
# # from pygtfcode.profiles.abg import toint4b
# from scipy.integrate import quad

# def menc(r, config):
#     """
#     Compute the enclosed mass M(r) in units of Mvir.

#     Parameters
#     ----------
#     r : float or ndarray
#         Radius (in units of r_s).
#     config : Config
#         Global simulation configuration object.

#     Returns
#     -------
#     M_enc : float or ndarray
#         Enclosed mass M(<r) in units of Mvir.
#     """
#     profile = config.init.profile

#     if profile == "nfw":
#         return fNFW(r)

#     elif profile == "truncated_nfw":
#         result, _ = quad(lambda x: toint4(x, config), 0.0, r, epsabs=1e-5, epsrel=1e-5)
#         return result

#     elif profile == "abg":
#         result, _ = quad(lambda x: toint4b(x, config), 0.0, r, epsabs=1e-5, epsrel=1e-5)
#         return result

#     else:
#         raise ValueError(f"Unknown profile type: {profile}")
