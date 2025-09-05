class SimParams:
    """
    Simulation control parameters.

    Attributes
    ----------
    t_halt : float
        Simulation halt time. Must be positive.
    rho_c_halt : float
        Central density at which to halt the simulation. Must be positive.
    lnL_param : float
        Coulomb logarithm calibration parameter. Must be positive.
    alpha : float
        Parameter alpha from Zhong & Shapiro (2025; arXiv:2505.18251)
    b : float
        Parameter b from Zhong & Shapiro (2025; arXiv:2505.18251)
    beta : float
        Parameter beta from Zhong & Shapiro (2025; arXiv:2505.18251)
    """
    def __init__(
            self, 
            t_halt : float = 1e3,
            rho_c_halt : float = 1500,
            lnL_param : float = 0.11,
            alpha : float = 1.217,
            beta : float = 1.0,
            b : float = 0.45
    ):
        self._t_halt = None
        self.rho_c_halt = rho_c_halt
        self._lnL_param = None
        self._alpha = None
        self._beta = None
        self._b = None

        self.t_halt = t_halt
        self.rho_c_halt = rho_c_halt
        self.lnL_param = lnL_param
        self.alpha = alpha
        self.beta = beta
        self.b = b

    @property
    def t_halt(self):
        return self._t_halt

    @t_halt.setter
    def t_halt(self, value):
        if value <= 0:
            raise ValueError("t_halt must be positive")
        self._t_halt = float(value)

    @property
    def rho_c_halt(self):
        return self._rho_c_halt

    @rho_c_halt.setter
    def rho_c_halt(self, value):
        if value <= 0:
            raise ValueError("rho_c_halt must be positive")
        self._rho_c_halt = float(value)

    @property
    def lnL_param(self):
        return self._lnL_param

    @lnL_param.setter
    def lnL_param(self, value):
        if value <= 0:
            raise ValueError("lnL_param must be positive")
        self._lnL_param = float(value)

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if value <= 0:
            raise ValueError("alpha must be positive")
        self._alpha = float(value)

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        if value <= 0:
            raise ValueError("beta must be positive")
        self._beta = float(value)

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        if value <= 0:
            raise ValueError("b must be positive")
        self._b = float(value)

    def __repr__(self):
        attrs = [
            attr for attr in dir(self)
            if not attr.startswith('_') and not callable(getattr(self, attr))
        ]
        attr_strs = []
        for attr in attrs:
            value = getattr(self, attr)
            attr_strs.append(f"{attr}={repr(value)}")
        return f"{self.__class__.__name__}({', '.join(attr_strs)})"
