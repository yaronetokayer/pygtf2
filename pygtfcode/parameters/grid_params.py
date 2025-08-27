class GridParams:
    """
    Parameters defining the radial grid for the simulation.

    Attributes
    ----------
    rmin : float
        Minimum radius of the grid, in units of the scale radius (r / r_s).
    rmax : float
        Maximum radius of the grid, in units of the scale radius (r / r_s).
    ngrid : int
        Number of radial grid points (must be > 1).
    """

    def __init__(
            self, 
            rmin: float = 1e-2,
            rmax: float = 2e2,
            ngrid: int = 200
            ):
        self._rmin = None
        self._rmax = None
        self._ngrid = None

        self.rmin = rmin
        self.rmax = rmax
        self.ngrid = ngrid

    @property
    def rmin(self):
        return self._rmin

    @rmin.setter
    def rmin(self, value):
        if value <= 0 or (self._rmax is not None and value >= self._rmax):
            raise ValueError("Require 0 < rmin < rmax")
        self._rmin = float(value)

    @property
    def rmax(self):
        return self._rmax

    @rmax.setter
    def rmax(self, value):
        if value <= 0 or (self._rmin is not None and value <= self._rmin):
            raise ValueError("Require 0 < rmin < rmax")
        self._rmax = float(value)

    @property
    def ngrid(self):
        return self._ngrid

    @ngrid.setter
    def ngrid(self, value):
        if not isinstance(value, int) or value <= 1:
            raise ValueError("ngrid must be an integer greater than 1")
        self._ngrid = value

    def __repr__(self):
        return f"GridParams(rmin={self.rmin}, rmax={self.rmax}, ngrid={self.ngrid})"

