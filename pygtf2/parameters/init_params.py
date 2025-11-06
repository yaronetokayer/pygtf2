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

class ExpParams(InitParams):
    """
    3D exponential density profile
    """
    pass

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

# from typing import Optional

# class InitParams:
#     """
#     Lightweight base for initial-profile parameter containers.

#     This class only provides a container for the profile identifier. Profile
#     shape and cosmological parameters (cvir, r_s, z) have been moved to
#     CosmologicalParams; profile-specific subclasses (NFWParams,
#     TruncatedNFWParams, ABGParams) add their own attributes (e.g. TruncatedNFWParams
#     defines Zt and deltaP, ABGParams defines alpha/beta/gamma).

#     Instances are typically created via make_init_params().

#     Attributes
#     ----------
#     profile : str | None
#         Identifier for the profile type ('nfw', 'truncated_nfw', 'abg')
#         or None until set by a subclass.
#     """

#     def __init__(self):
#         self.profile = None

#     def __repr__(self):
#         return f"{self.__class__.__name__}()"

# class CosmologicalParams(InitParams):
#     """
#     Base for cosmological (halo-like) profiles.
#     Includes redshift and cvir/r_s specification with mutual exclusion.
#     """
#     def __init__(self, cvir: Optional[float] = 20.0,
#                        r_s: Optional[float] = None,
#                        z: float = 0.0):
#         super().__init__()
#         # storage
#         self._z: float = 0.0
#         self._cvir: Optional[float] = None
#         self._r_s: Optional[float] = None

#         # set in order so mutual exclusion works as expected
#         self.cvir = cvir
#         self.r_s = r_s
#         self.z = z

#     # ---- z ----
#     @property
#     def z(self) -> float:
#         return self._z

#     @z.setter
#     def z(self, value: float):
#         if value < 0:
#             raise ValueError("Redshift z must be non-negative.")
#         self._z = float(value)

#     # ---- cvir ----
#     @property
#     def cvir(self) -> Optional[float]:
#         return self._cvir

#     @cvir.setter
#     def cvir(self, value: Optional[float]):
#         if value is not None:
#             if value <= 0:
#                 raise ValueError("cvir must be positive.")
#             # enforce mutual exclusion
#             self._r_s = None
#             self._cvir = float(value)
#         else:
#             self._cvir = None

#     # ---- r_s ----
#     @property
#     def r_s(self) -> Optional[float]:
#         return self._r_s

#     @r_s.setter
#     def r_s(self, value: Optional[float]):
#         if value is not None:
#             if value <= 0:
#                 raise ValueError("r_s must be positive.")
#             # enforce mutual exclusion
#             self._cvir = None
#             self._r_s = float(value)
#         else:
#             self._r_s = None

#     def __repr__(self):
#         return f"{self.__class__.__name__}(cvir={self.cvir}, r_s={self.r_s}, z={self.z})"

# class NFWParams(CosmologicalParams):
#     """Standard NFW profile."""

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.profile = 'nfw'

# class TruncatedNFWParams(CosmologicalParams):
#     """
#     Energy-truncated NFW profile.

#     Zt : cutoff value for the potential (max binding energy kept)
#     deltaP : step size in potential for initial integration
#     """
#     def __init__(self, Zt: float = 0.05938, deltaP: float = 1.0e-5, **kwargs):
#         super().__init__(**kwargs)
#         self.profile = 'truncated_nfw'

#         self._Zt: float = 0.0
#         self._deltaP: float = 0.0

#         self.Zt = Zt
#         self.deltaP = deltaP

#     @property
#     def Zt(self) -> float:
#         return self._Zt

#     @Zt.setter
#     def Zt(self, value: float):
#         if value <= 0:
#             raise ValueError("Zt must be positive.")
#         self._Zt = float(value)

#     @property
#     def deltaP(self) -> float:
#         return self._deltaP

#     @deltaP.setter
#     def deltaP(self, value: float):
#         if value <= 0:
#             raise ValueError("deltaP must be positive.")
#         self._deltaP = float(value)

#     def __repr__(self):
#         return (f"TruncatedNFWParams(cvir={self.cvir}, r_s={self.r_s}, z={self.z}, "
#                 f"Zt={self.Zt}, deltaP={self.deltaP})")

# class ABGParams(InitParams):
#     """
#     Alpha-beta-gamma profile as defined in Zhao (1996) (10.1093/mnras/278.2.488)

#     Attributes
#     ----------
#     alpha : float
#     beta : float
#     gamma : float
#     """

#     def __init__(self, alpha: float = 4.0, beta: float = 4.0, gamma: float = 0.1, **kwargs):
#         super().__init__(**kwargs)
#         self.profile = 'abg'

#         self._alpha: float = 0.0
#         self._beta: float = 0.0
#         self._gamma: float = 0.0

#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma

#     @property
#     def alpha(self) -> float:
#         return self._alpha
#     @alpha.setter
#     def alpha(self, value: float):
#         self._alpha = float(value)

#     @property
#     def beta(self) -> float:
#         return self._beta
#     @beta.setter
#     def beta(self, value: float):
#         self._beta = float(value)

#     @property
#     def gamma(self) -> float:
#         return self._gamma
#     @gamma.setter
#     def gamma(self, value: float):
#         self._gamma = float(value)

#     def __repr__(self):
#         return (f"ABGParams(cvir={self.cvir}, r_s={self.r_s}, z={self.z}, "
#                 f"alpha={self.alpha}, beta={self.beta}, gamma={self.gamma})")


# def make_init_params(profile, **kwargs):
#     """
#     Factory function to create the appropriate InitParams subclass.

#     Example usage:
#     >>> params = make_init_params("abg", alpha=3.5, beta=5.0)

#     Parameters
#     ----------
#     profile : str
#         Type of initial profile. Options: 'nfw', 'truncated_nfw', 'abg'.
#     **kwargs : dict
#         Parameters passed to the corresponding class constructor.

#     Returns
#     -------
#     InitParams
#         An instance of NFWParams, TruncatedNFWParams, or ABGParams.

#     Raises
#     ------
#     ValueError
#         If the profile name is unrecognized.
#     """
#     profile = profile.strip().lower()

#     if profile == "nfw":
#         return NFWParams(**kwargs)
#     elif profile == "truncated_nfw":
#         return TruncatedNFWParams(**kwargs)
#     elif profile == "abg":
#         return ABGParams(**kwargs)
#     else:
#         raise ValueError(f"Unknown profile type: '{profile}'")

