import math

THERMAL_BOUNDARY_STRINGS = ['zero_luminosity', 'fixed_temperature']

class SimParams:
    """
    Simulation control parameters.

    Attributes
    ----------
    t_halt : float
        Absolute halt time in units of char.t0. Must be positive.
    rho0_halt : float
        Total innermost density threshold in units of char.rho_s. Must be
        positive; the integrator applies this limit after the first 1000 steps.
    lnL_param : float
        Coulomb logarithm calibration parameter. Must be positive.
    alpha : float
        Parameter alpha from Zhong & Shapiro (2025; arXiv:2505.18251)
    b : float
        Parameter b from Zhong & Shapiro (2025; arXiv:2505.18251)
    beta : float
        Parameter beta from Zhong & Shapiro (2025; arXiv:2505.18251)
    m_bh : float
        Mass of central black hole in solar masses.
    thermal_boundary : str
        Conductivity boundary condition at inner and outer edges.
        Options are ['zero_luminosity', 'fixed_temperature'].
    tau_boundary : tuple
        Thermal coupling of heat reservoirs with boundary cells.
        Used for fixed temperature thermal boundary
    evap : bool
        Reserved evaporation switch; the current integrator does not apply it.
    binaries : bool
        Include binary heating in simulation
    n_particle: int
        Absolute particle number for binary heat generation
    bkg : dict or None
        Background configuration with exactly the keys prof, mass, length,
        and other. None disables the background. For prof='hernq_static',
        provide mass in Msun and length in kpc; other may be None.
        'hernq_decay' is accepted by the parameter container but is not
        implemented by the enclosed-mass routines. The property returns a
        copy; assign a complete dictionary to change the configuration.
    """
    VALID_BKG_PROFILES = ('hernq_static', 'hernq_decay')
    DEFAULT_BKG = {'prof': None, 'mass': None, 'length': None, 'other': None}

    def __init__(
            self, 
            t_halt              : float = 1000.0,
            rho0_halt           : float = 1.0e8,
            lnL_param           : float = 0.11,
            alpha               : float = 1.217,
            beta                : float = 1.0,
            b                   : float = 0.44,
            m_bh                : float = 0.0,
            thermal_boundary    : str   = 'zero_luminosity',
            tau_boundary        : tuple | None = None,
            evap                : bool  = False,
            binaries            : bool  = False,
            n_particle          : int   = 1e4,
            bkg                 : dict | None = None,
    ):
        
        self._t_halt = None
        self._rho0_halt = None
        self._lnL_param = None
        self._alpha = None
        self._beta = None
        self._b = None
        self._m_bh = None
        self._thermal_boundary = None
        self._tau_boundary = None
        self._evap = None
        self._binaries = None
        self._n_particle = None
        self._bkg = dict(self.DEFAULT_BKG)

        self.t_halt = t_halt
        self.rho0_halt = rho0_halt
        self.lnL_param = lnL_param
        self.alpha = alpha
        self.beta = beta
        self.b = b
        self.m_bh = m_bh
        self.thermal_boundary = thermal_boundary
        self.tau_boundary = tau_boundary
        self.evap = evap
        self.binaries = binaries
        self._n_particle = int(n_particle)
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
    def rho0_halt(self):
        return self._rho0_halt

    @rho0_halt.setter
    def rho0_halt(self, value):
        if value <= 0:
            raise ValueError("rho0_halt must be positive")
        self._rho0_halt = float(value)

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
    def m_bh(self):
        return self._m_bh

    @m_bh.setter
    def m_bh(self, value):
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("m_bh must be finite and non-negative")
        self._m_bh = float(value)

    @property
    def thermal_boundary(self):
        return self._thermal_boundary

    @thermal_boundary.setter
    def thermal_boundary(self, value):
        if value not in THERMAL_BOUNDARY_STRINGS:
            raise ValueError(f"thermal_boundary options are {THERMAL_BOUNDARY_STRINGS}")
        self._thermal_boundary = value

    @property
    def tau_boundary(self):
        return self._tau_boundary

    @tau_boundary.setter
    def tau_boundary(self, value):
        if value is None:
            self._tau_boundary = None
            return

        values = tuple(float(x) for x in value)

        if len(values) != 2:
            raise ValueError("tau_boundary must contain inner and outer times")

        if any(not math.isfinite(x) or x <= 0.0 for x in values):
            raise ValueError("tau_boundary values must be finite and positive")

        self._tau_boundary = values

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
    def n_particle(self):
        return self._n_particle

    @n_particle.setter
    def n_particle(self, value):
        if value <= 0:
            raise ValueError("n_particle must be positive")
        self._n_particle = int(value)

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
