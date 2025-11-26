class PrecisionParams:
    """
    Parameters controlling numerical precision and convergence behavior.

    Attributes
    ----------
    eps_du : float
        Maximum allowed relative change in internal energy (u) per time step.
    eps_dt : float
        Epsilon factor for adjusting time step size.
    epsabs : float
        Absolute tolerance for numerical integration routines.
    epsrel : float
        Relative tolerance for numerical integration routines.
    """

    def __init__(
        self,
        eps_du : float = 1.0e-4,
        eps_dt : float = 1.0e-3,
        epsabs : float = 1e-6,
        epsrel : float = 1e-6
    ):
        self._eps_du = None
        self._eps_dt = None
        self._epsabs = None
        self._epsrel = None

        self.eps_du = eps_du
        self.eps_dt = eps_dt
        self.epsabs = epsabs
        self.epsrel = epsrel

    @property
    def eps_du(self):
        return self._eps_du

    @eps_du.setter
    def eps_du(self, value):
        self._validate_positive(value, "eps_du")
        self._eps_du = float(value)

    @property
    def eps_dt(self):
        return self._eps_dt

    @eps_dt.setter
    def eps_dt(self, value):
        self._validate_positive(value, "eps_dt")
        self._eps_dt = float(value)

    @property
    def epsabs(self):
        return self._epsabs

    @epsabs.setter
    def epsabs(self, value):
        self._validate_positive(value, "epsabs")
        self._epsabs = float(value)

    @property
    def epsrel(self):
        return self._epsrel

    @epsrel.setter
    def epsrel(self, value):
        self._validate_positive(value, "epsrel")
        self._epsrel = float(value)

    def _validate_positive(self, value, name):
        if not (value > 0):
            raise ValueError(f"{name} must be a positive float.")

    def _validate_nonnegative_int(self, value, name):
        if not (isinstance(value, int) and value >= 0):
            raise ValueError(f"{name} must be a non-negative integer.")

    def __repr__(self):
        attrs = [
            attr for attr in dir(self)
            if not attr.startswith("_") and not callable(getattr(self, attr))
        ]
        attr_strs = [f"{attr}={getattr(self, attr)!r}" for attr in attrs]
        return f"{self.__class__.__name__}({', '.join(attr_strs)})"
