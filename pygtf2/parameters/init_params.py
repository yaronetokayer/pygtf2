from typing import Optional

class InitParams:
    """
    Base class for parameters defining the initial density profile.
    Only shape parameters are included, since mass is set globally in Config and determined per species via SpecParams.frac.

    Attributes
    ----------
    cvir : float, optional
        Concentration parameter.
        If set, then r_s is set to None and derived from cvir and mtot upon state initialization.
    r_s : float, optional
        Scale radius in kpc.
        If set, then cvir is set to None and derived from r_s and mtot upon state initialization.
    z : float, optional
        Redshift (must be non-negative).
        Default is 0.0
    profile : str or None
        String identifier for the profile type ('nfw', 'truncated_nfw', 'abg').
    """

    def __init__(self, 
                 cvir: Optional[float] = 20.0,
                 r_s: Optional[float] = None, 
                 z: float = 0.0
        ):
        self._cvir = None
        self._r_s = None
        self._z = None
        self.profile = None  # To be set by subclass

        self.cvir = cvir
        self.r_s = r_s
        self.z = z

    @property
    def cvir(self):
        return self._cvir

    @cvir.setter
    def cvir(self, value):
        if value is not None:
            if value <= 0:
                raise ValueError("cvir must be positive.")
            self._r_s = None
            self._cvir = float(value)
        else:
            self._cvir = None

    @property
    def r_s(self):
        return self._r_s

    @r_s.setter
    def r_s(self, value):
        if value is not None:
            if value <= 0:
                raise ValueError("r_s must be positive.")
            self._cvir = None
            self._r_s = float(value)
        else:
            self._r_s = None

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, value):
        if value < 0:
            raise ValueError("Redshift z must be non-negative.")
        self._z = float(value)

    def __repr__(self):
        return f"{self.__class__.__name__}(cvir={self.cvir}, r_s={self.r_s}, z={self.z})"


class NFWParams(InitParams):
    """Standard NFW profile."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.profile = 'nfw'


class TruncatedNFWParams(InitParams):
    """
    Energy-truncated NFW profile.

    Attributes
    ----------
    Zt : float
        Cutoff value for the potential; maximum binding energy for particles retained in the truncated halo.
    deltaP : float
        step size in potential for initial integration.
    """

    def __init__(self, Zt=0.05938, deltaP=1.0e-5, **kwargs):
        super().__init__(**kwargs)
        self.profile = 'truncated_nfw'

        self._Zt = None
        self._deltaP = None
        self.Zt = Zt
        self.deltaP = deltaP

    @property
    def Zt(self):
        return self._Zt

    @Zt.setter
    def Zt(self, value):
        if value <= 0:
            raise ValueError("Zt must be positive.")
        self._Zt = float(value)

    @property
    def deltaP(self):
        return self._deltaP

    @deltaP.setter
    def deltaP(self, value):
        if value <= 0:
            raise ValueError("deltaP must be positive.")
        self._deltaP = float(value)

    def __repr__(self):
        return (f"TruncatedNFWParams(cvir={self.cvir}, r_s={self.r_s}, z={self.z}, "
                f"Zt={self.Zt}, deltaP={self.deltaP})")


class ABGParams(InitParams):
    """
    Alpha-beta-gamma profile as defined in Zhao (1996) (10.1093/mnras/278.2.488)

    Attributes
    ----------
    alpha : float
    beta : float
    gamma : float
    """

    def __init__(self, alpha=4.0, beta=4.0, gamma=0.1, **kwargs):
        super().__init__(**kwargs)
        self.profile = 'abg'

        self._alpha = None
        self._beta = None
        self._gamma = None
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = float(value)

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        self._beta = float(value)

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        self._gamma = float(value)

    def __repr__(self):
        return (f"ABGParams(cvir={self.cvir}, r_s={self.r_s}, z={self.z}, "
                f"alpha={self.alpha}, beta={self.beta}, gamma={self.gamma})")


def make_init_params(profile, **kwargs):
    """
    Factory function to create the appropriate InitParams subclass.

    Example usage:
    >>> params = make_init_params("abg", alpha=3.5, beta=5.0)

    Parameters
    ----------
    profile : str
        Type of initial profile. Options: 'nfw', 'truncated_nfw', 'abg'.
    **kwargs : dict
        Parameters passed to the corresponding class constructor.

    Returns
    -------
    InitParams
        An instance of NFWParams, TruncatedNFWParams, or ABGParams.

    Raises
    ------
    ValueError
        If the profile name is unrecognized.
    """
    profile = profile.strip().lower()

    if profile == "nfw":
        return NFWParams(**kwargs)
    elif profile == "truncated_nfw":
        return TruncatedNFWParams(**kwargs)
    elif profile == "abg":
        return ABGParams(**kwargs)
    else:
        raise ValueError(f"Unknown profile type: '{profile}'")
