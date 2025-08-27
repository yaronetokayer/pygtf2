class SimParams:
    """
    Simulation control parameters.

    Attributes
    ----------
    sigma_m : float
        Self-interaction cross-section in cm^2/g. Must be positive.
    t_halt : float
        Simulation halt time. Must be positive.
    rho_c_halt : float
        Central density at which to halt the simulation. Must be positive.
    a : float
        Model parameter 'a'. Must be positive.
    b : float
        Model parameter 'b'. Must be positive.
    c : float
        Model parameter 'c'. Must be positive.
    """
    def __init__(
            self, 
            sigma_m : float = 10.0,
            t_halt : float = 1e3,
            rho_c_halt : float = 1500,
            a : float = 2.256758,
            b: float = 1.38,
            c: float = 0.75
    ):
        self._sigma_m = None
        self._t_halt = None
        self.rho_c_halt = rho_c_halt
        self._a = None
        self._b = None
        self._c = None

        self.sigma_m = sigma_m
        self.t_halt = t_halt
        self.rho_c_halt = rho_c_halt
        self.a = a
        self.b = b
        self.c = c

    @property
    def sigma_m(self):
        return self._sigma_m

    @sigma_m.setter
    def sigma_m(self, value):
        if value <= 0:
            raise ValueError("sigma_m must be positive")
        self._sigma_m = float(value)

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
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        if value <= 0:
            raise ValueError("a must be positive")
        self._a = float(value)

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        if value <= 0:
            raise ValueError("b must be positive")
        self._b = float(value)

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, value):
        if value <= 0:
            raise ValueError("c must be positive")
        self._c = float(value)

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
