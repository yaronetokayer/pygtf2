from typing import Dict, Any, Tuple
from pygtf2.parameters.init_params import (
    InitParams,
    NFWParams,
    make_init_params,
)

class SpecParams:
    """
    Class for a single species' static configuration.

    Attributes
    ----------
    m_part : float
        Particle mass of the species [Msun].
    frac : float
        Mass fraction f_k.
    init : InitParams
        Initial profile object.
    """
    def __init__(
            self, 
            m_part: float = 1.0, 
            frac: float = 1.0, 
            init: InitParams | None = None
        ):
        self._m_part = float(m_part)
        self._frac = float(frac)
        self._init = init if init is not None else NFWParams()

    @property
    def m_part(self) -> float:
        return self._m_part

    @m_part.setter
    def m_part(self, value: float) -> None:
        if value <= 0:
            raise ValueError("m_part must be positive.")
        self._m_part = float(value)

    @property
    def frac(self) -> float:
        return self._frac

    @frac.setter
    def frac(self, value: float) -> None:
        if value <= 0:
            raise ValueError("frac must be positive.")
        self._frac = float(value)

    @property
    def init(self) -> InitParams:
        return self._init

    @init.setter
    def init(self, value: InitParams | str | Tuple[str, Dict[str, Any]]):
        if isinstance(value, InitParams):
            self._init = value
        elif isinstance(value, str):
            self._init = make_init_params(value)
        elif isinstance(value, tuple) and len(value) == 2:
            profile, kwargs = value
            self._init = make_init_params(profile, **kwargs)
        else:
            raise TypeError("init must be an InitParams, str, or (str, dict) tuple")

    def __repr__(self) -> str:
        return f"SpecParams(m_part={self._m_part}, frac={self._frac}, init={self._init})"