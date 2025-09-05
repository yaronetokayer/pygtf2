from pprint import pformat

class CharParams:
    """
    Stores characteristic physical quantities derived from input parameters.

    Attributes
    ----------
    r_s : float
        Scale radius [Mpc].
    fc : float or None
        NFW normalization factor.
    chi : float or None
        ABG normalization factor (None unless using ABG).
    m_s : float
        Characteristic mass scale [Msun].
    t0 : float
        Characteristic time scale [sec].
    v0 : float
        Characteristic velocity scale [km/s].
    rho_s : float
        Characteristic density [Msun / Mpc^3].
    lnL : ndarray
        Coulomb logarithm matrix
    c1 : float
        Parameter c1 from Zhong & Shapiro (2025; arXiv:2505.18251) Eq. 42
    c2 : float
        Parameter c2 from Zhong & Shapiro (2025; arXiv:2505.18251) Eq. 43
    """

    def __init__(self):
        self.r_s = None
        self.fc = None
        self.chi = None
        self.m_s = None
        self.t0 = None
        self.v0 = None
        self.rho_s = None
        self.lnL = None
        self.c1 = None
        self.c2 = None

    def __repr__(self):
        attrs = {k: v for k, v in self.__dict__.items()}
        return f"CharParams(\n{pformat(attrs, indent=4)}\n)"
