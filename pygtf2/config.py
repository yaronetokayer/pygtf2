from typing import Dict, Any, Optional, Tuple
from pygtf2.parameters.io_params import IOParams
from pygtf2.parameters.grid_params import GridParams
from pygtf2.parameters.init_params import (
    InitParams,
    NFWParams,
    make_init_params,
)
from pygtf2.parameters.spec_params import SpecParams
from pygtf2.parameters.prec_params import PrecisionParams
from pygtf2.parameters.sim_params import SimParams
import pprint

def _init_param(param_class, arg):
    if arg is None:
        return param_class()
    elif isinstance(arg, dict):
        return param_class(**arg)
    elif isinstance(arg, param_class):
        return arg
    else:
        raise TypeError(f"Expected {param_class.__name__}, dict, or None")

class Config:
    """
    Central container for all static simulation parameters.

    Attributes
    ----------
    io : IOParams
    grid : GridParams
    sim : SimParams
    prec : PrecisionParams
    mtot : float
        Total mass of the halo in Msun (global).
    spec : Dict[str, SpecParams]
        Dictionary of species specifications, keyed by species name.
    """

    def __init__(
        self,
        io: Optional[IOParams | Dict[str, Any]] = None,
        grid: Optional[GridParams | Dict[str, Any]] = None,
        sim: Optional[SimParams | Dict[str, Any]] = None,
        prec: Optional[PrecisionParams | Dict[str, Any]] = None,
        *,
        mtot: Optional[float] = 3.0e9,
    ) -> None:
        self.io = _init_param(IOParams, io)
        self.grid = _init_param(GridParams, grid)
        self.sim = _init_param(SimParams, sim)
        self.prec = _init_param(PrecisionParams, prec)

        self.spec: Dict[str, SpecParams] = {}

        self._mtot = None
        self.mtot = mtot

    @property
    def mtot(self):
        return self._mtot
    
    @mtot.setter
    def mtot(self, value):
        if value <= 0:
            raise ValueError("mtot must be positive")
        self._mtot = float(value)

    # --- Species management ---
    def add_species(
        self,
        name: Optional[str] = None,
        *,
        m_part: float = 1.0,
        frac: float = 1.0,
        init: InitParams | str | Tuple[str, Dict[str, Any]] | None = None,
    ) -> SpecParams:
        """
        Create and add a species to the config.

        If no arguments are given, defaults are used and the user can edit later.

        Returns the SpecParams object so it can be modified directly.
        """
        if name is None:
            name = f"species{len(self.spec)}"
        
        if init is None:
            sp_init = NFWParams()
        elif isinstance(init, InitParams):
            sp_init = init
        elif isinstance(init, str):
            sp_init = make_init_params(init)
        elif isinstance(init, tuple) and len(init) == 2:
            profile, kwargs = init
            sp_init = make_init_params(profile, **kwargs)
        else:
            raise TypeError("init must be None, an InitParams, str, or (str, dict) tuple")

        sp = SpecParams(m_part=m_part, frac=frac, init=sp_init)
        if name in self.spec:
            print(f"Species '{name}' already exists in Config; replacing with new definition.")
        self.spec[name] = sp
        return sp

    def remove_species(self, name: str) -> None:
        """
        Remove a species by name.

        Raises KeyError if the species does not exist.
        """
        if name not in self.spec:
            raise KeyError(f"Species '{name}' not found in Config object.")
        del self.spec[name]

    @property
    def s(self) -> int:
        return len(self.spec)
    
    # --- Loading from metadata for a State.from_dir() ---
    @classmethod
    def from_dict(cls, meta: Dict[str, Dict[str, Any]]) -> "Config":
        """"
        Build a Config from the nested dict produced by pygtf2.io.read.import_metadata().

        This is used when a state is constructed with State.from_dir().

        Expected sections:
        "_mtot", "grid", "io", "prec", "sim", "spec"
        Keys may be prefixed with underscores (e.g., "_alpha", "_r_s", ...).
        """

        # Helper to strip leading underscores off keys
        def norm(d: Dict[str, Any]) -> Dict[str, Any]:
            return {k.lstrip("_"): v for k, v in d.items()}
        
        # Fetch top-level section
        def get_section(name: str) -> Dict[str, Any]:
            if name in meta:
                return meta[name]
            alt = "_" + name
            if alt in meta:
                return meta[alt]
            raise KeyError(f"Missing '{name}' section in metadata.")
        
        # Sections
        grid_raw = norm(get_section("grid"))
        io_raw   = norm(get_section("io"))
        prec_raw = norm(get_section("prec"))
        sim_raw  = norm(get_section("sim"))

        io_params   = IOParams(**io_raw)
        grid_params = GridParams(**grid_raw)
        prec_params = PrecisionParams(**prec_raw)
        sim_params  = SimParams(**sim_raw)

        # Instantiate config (no species yet)
        cfg = cls(io=io_params, grid=grid_params, sim=sim_params, prec=prec_params)

        # --- mtot ---
        mtot_val = meta.get("_mtot", meta.get("mtot", None))
        if mtot_val is None:
            raise KeyError("Missing 'mtot'/_mtot in metadata.")
        cfg.mtot = float(mtot_val)

        # --- species ---
        spec_raw_top = meta.get("spec", {})  # already a dict of {name: {...}}
        if not isinstance(spec_raw_top, dict):
            raise TypeError("Expected 'spec' to be a dict mapping species name -> params.")

        for name, sp in spec_raw_top.items():
            spn = norm(sp)  # strip leading underscores on m_part/frac/init, etc.
            # init block
            init_block = norm(spn.get("init", {}))
            profile = init_block.pop("profile", "nfw")
            init_obj = make_init_params(profile, **init_block)

            m_part = float(spn["m_part"])
            frac   = float(spn["frac"])

            # add to config (replace if duplicate name, per earlier behavior)
            cfg.add_species(name=name, m_part=m_part, frac=frac, init=init_obj)

        return cfg

    # --- Representation ===
    def __repr__(self):
        spec_str = pprint.pformat(self.spec, indent=2, compact=True)
        return (
            "Config(\n"
            f"  io={self.io},\n"
            f"  grid={self.grid},\n"
            f"  mtot={self.mtot},\n"
            f"  sim={self.sim},\n"
            f"  prec={self.prec},\n"
            f"  spec={spec_str}\n"
            ")"
        )