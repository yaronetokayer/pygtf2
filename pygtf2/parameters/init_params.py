from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import warnings

# ---- Profile schemas: which fields are applicable and their defaults
PROFILE_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "nfw": {
        "applicable": {"cvir", "r_s", "z"},
        "defaults":   {"cvir": 20.0, "r_s": None, "z": 0.0},
    },
    "truncated_nfw": {
        "applicable": {"cvir", "r_s", "z", "Zt", "deltaP"},
        "defaults":   {"cvir": 20.0, "r_s": None, "z": 0.0, "Zt": 0.05938, "deltaP": 1.0e-5},
    },
    "abg": {
        "applicable": {"cvir", "r_s", "z", "alpha", "beta", "gamma"},
        "defaults":   {"cvir": None, "r_s": 1.0, "z": 0.0, "alpha": 4.0, "beta": 4.0, "gamma": 0.1},
    },
    "exp": {
        "applicable": {"r_s", "z"},
        "defaults":   {"r_s": 1.0, "z": 0.0},
    },
}

def _norm_prof(p: str) -> str:
    if not isinstance(p, str):
        raise TypeError("prof must be a string.")
    p2 = p.strip().lower()
    if p2 not in PROFILE_SCHEMAS:
        raise ValueError(f"Unknown profile '{p}'. Options: {', '.join(PROFILE_SCHEMAS)}")
    return p2

@dataclass
class InitParams:
    """
    Single container for initial profile parameters.
    Fields not applicable to the selected profile are coerced to None with a warning.
    Mutual exclusivity: setting cvir clears r_s, and setting r_s clears cvir.
    """
    # profile tag
    prof: str = None

    # superset of possible fields (Optional where appropriate)
    cvir: Optional[float] = None
    r_s: float = None
    z: Optional[float] = None

    # truncated NFW
    Zt: Optional[float] = None
    deltaP: Optional[float] = None

    # ABG (Zhao) profile
    alpha: Optional[float] = None
    beta: Optional[float] = None
    gamma: Optional[float] = None

    # non-init field to accumulate soft notices if you want to inspect later
    notices: list[str] = field(default_factory=list, repr=False, compare=False)

    def __post_init__(self):
        # normalize prof, apply schema defaults, then coerce/validate
        self.prof = _norm_prof(self.prof)
        self._apply_profile_defaults(only_if_none=True)
        self._coerce_irrelevant_fields()
        self._apply_exclusivity()
        self._validate()

    # ----- Public API

    @classmethod
    def from_profile(cls, profile: str, **kwargs) -> "InitParams":
        """Factory preserving your old make_init_params behavior."""
        prof = _norm_prof(profile)
        obj = cls(prof=prof, **kwargs)
        return obj

    def set_profile(self, profile: str, **overrides) -> "InitParams":
        """
        Switch profile, reset to that profile's defaults, then apply overrides.
        Returns self for fluent usage.
        """
        self.prof = _norm_prof(profile)
        # reset all fields to None, then apply defaults for the new profile
        for k in self._all_field_names():
            if k not in {"prof", "notices"}:
                setattr(self, k, None)
        self._apply_profile_defaults(only_if_none=False)
        # apply overrides
        for k, v in overrides.items():
            setattr(self, k, v)
        # finalize
        self._coerce_irrelevant_fields()
        self._apply_exclusivity()
        self._validate()
        return self

    def update(self, **kwargs) -> "InitParams":
        """
        Update current profile parameters. Irrelevant fields are ignored (soft warning).
        """
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._coerce_irrelevant_fields()
        self._apply_exclusivity()
        self._validate()
        return self

    # ----- Internals

    def _apply_profile_defaults(self, *, only_if_none: bool):
        schema = PROFILE_SCHEMAS[self.prof]
        for k, v in schema["defaults"].items():
            if (not only_if_none) or getattr(self, k) is None:
                setattr(self, k, v)

    def _coerce_irrelevant_fields(self):
        schema = PROFILE_SCHEMAS[self.prof]
        applicable = schema["applicable"]
        for k in self._all_field_names():
            if k in {"prof", "notices"}:
                continue
            if k not in applicable:
                if getattr(self, k) is not None:
                    msg = (
                        f"Field '{k}' is not applicable to profile '{self.prof}'. "
                        f"Overriding to None."
                    )
                    warnings.warn(msg)
                    self.notices.append(msg)
                setattr(self, k, None)

    def _apply_exclusivity(self):
        # cvir vs r_s mutual exclusivity (both optional; if both set, prefer the one set most recently)
        # Heuristic: if both are set and last assignment unknown, prefer keeping the non-None with larger magnitude of change.
        # Simpler rule here: if both non-None, keep whichever is not None *most recently* updated by user.
        # Since we can't track recency easily without custom setters, apply a deterministic rule:
        # If both set, prefer r_s and clear cvir (or flip this if you prefer the opposite).
        if self.cvir is not None and self.r_s is not None:
            msg = "Both cvir and r_s provided; enforcing exclusivity by keeping r_s and clearing cvir."
            warnings.warn(msg)
            self.notices.append(msg)
            self.cvir = None

    def _validate(self):
        def _pos(name: str, val: Optional[float]):
            if val is not None and val <= 0:
                raise ValueError(f"{name} must be positive.")
        _pos("cvir", self.cvir)
        _pos("r_s", self.r_s)
        _pos("Zt", self.Zt)
        _pos("deltaP", self.deltaP)
        if self.z < 0:
            raise ValueError("z must be non-negative.")

    @staticmethod
    def _all_field_names():
        return {"prof", "cvir", "r_s", "z", "Zt", "deltaP", "alpha", "beta", "gamma", "notices"}

    def __repr__(self):
        # Compact repr showing only relevant (applicable) fields for the current profile
        applicable = PROFILE_SCHEMAS[self.prof]["applicable"]
        core = {"prof": self.prof}
        core.update({k: getattr(self, k) for k in applicable})
        return f"InitParams({', '.join(f'{k}={v}' for k, v in core.items())})"


# ----- Backward-compatible factory
def make_init_params(profile: str, **kwargs) -> InitParams:
    return InitParams.from_profile(profile, **kwargs)