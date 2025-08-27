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
    sigma0 : float
        Characteristic cross section [cm^2/g].
    t0 : float
        Characteristic time scale [sec].
    v0 : float
        Characteristic velocity scale [km/s].
    rho_s : float
        Characteristic density [Msun / Mpc^3].
    """

    def __init__(self):
        self.r_s = None
        self.fc = None
        self.chi = None
        self.m_s = None
        self.sigma0 = None
        self.t0 = None
        self.v0 = None
        self.rho_s = None

    def __repr__(self):
        return (
            f"CharParams(r_s={self.r_s}, fc={self.fc}, chi={self.chi}, "
            f"m_s={self.m_s}, sigma0={self.sigma0}, t0={self.t0}, "
            f"v0={self.v0}, rho_s={self.rho_s})"
        )
