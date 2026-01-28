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
    bkg : str
        String representing the background potential. So far implemented {None, 'hernq'}
    """
    VALID_BKG_PROFILES = ('hernq_static', 'hernq_decay')
    DEFAULT_BKG = {'prof': None, 'mass': None, 'length': None, 'other': None}

    def __init__(
            self, 
            t_halt : float = 1000.0,
            rho_c_halt : float = 1.0e8,
            lnL_param : float = 0.11,
            alpha : float = 1.217,
            beta : float = 1.0,
            b : float = 0.45,
            evap : bool = False,
            binaries : bool = False,
            bkg: dict | None = None,
    ):
        self._t_halt = None
        self._rho_c_halt = None
        self._lnL_param = None
        self._alpha = None
        self._beta = None
        self._b = None
        self._evap = None
        self._binaries = None
        self._bkg = dict(self.DEFAULT_BKG)

        self.t_halt = t_halt
        self.rho_c_halt = rho_c_halt
        self.lnL_param = lnL_param
        self.alpha = alpha
        self.beta = beta
        self.b = b
        self.evap = evap
        self.binaries = binaries
        if bkg is not None:
            self.bkg = bkg

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

    @property
    def evap(self):
        return self._evap

    @evap.setter
    def evap(self, value):
        if not isinstance(value, (bool, int)) or (isinstance(value, int) and value not in (0, 1)):
            raise ValueError("evap must be a bool (True/False) or int (0/1)")
        self._evap = bool(value)

    @property
    def binaries(self):
        return self._binaries

    @binaries.setter
    def binaries(self, value):
        if not isinstance(value, (bool, int)) or (isinstance(value, int) and value not in (0, 1)):
            raise ValueError("binaries must be a bool (True/False) or int (0/1)")
        self._binaries = bool(value)

    @property
    def bkg(self):
        return dict(self._bkg)

    @bkg.setter
    def bkg(self, value):
        if value is None:
            self._bkg = dict(self.DEFAULT_BKG)
            return

        if not isinstance(value, dict):
            raise TypeError("bkg must be a dict with keys 'prof','mass','length','other'")

        expected_keys = {'prof', 'mass', 'length', 'other'}
        if set(value.keys()) != expected_keys:
            raise ValueError("bkg must contain exactly the keys: 'prof','mass','length','other'")

        prof = value.get('prof')
        if prof is not None:
            if not isinstance(prof, str) or prof not in self.VALID_BKG_PROFILES:
                raise ValueError(f"bkg['prof'] must be None or one of {self.VALID_BKG_PROFILES}")

        validated = {'prof': prof}
        for key in ('mass', 'length', 'other'):
            val = value.get(key)
            if val is None:
                validated[key] = None
            else:
                try:
                    fval = float(val)
                except Exception:
                    raise TypeError(f"bkg['{key}'] must be a positive float or None")
                if fval <= 0:
                    raise ValueError(f"bkg['{key}'] must be positive")
                validated[key] = fval

        self._bkg = validated

    def __repr__(self):
        exclude = {'VALID_BKG_PROFILES', 'DEFAULT_BKG'}
        attrs = [
            attr for attr in dir(self)
            if not attr.startswith('_')
            and not callable(getattr(self, attr))
            and attr not in exclude
        ]
        attr_strs = []
        for attr in attrs:
            try:
                value = getattr(self, attr)
            except Exception:
                value = '<error>'
            attr_strs.append(f"{attr}={repr(value)}")
        return f"{self.__class__.__name__}({', '.join(attr_strs)})"
