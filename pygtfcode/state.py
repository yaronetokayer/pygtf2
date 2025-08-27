import numpy as np
from pygtfcode.parameters.constants import Constants as const
import pprint
from pathlib import Path

def _xH(z, const):
    """
    Returns H(z) in units of km/s/Mpc using cosmological parameters
    defined in config.constants or config.init.

    Parameters
    ----------
    z : float
        Redshift.

    const : Constants
        Configuration object containing cosmological parameters.

    Returns
    -------
    H_z : float
        Hubble parameter at redshift z [km/s/Mpc].
    """
    Omega_m = float(const.Omega_m)
    omega_lambda = 1 - Omega_m
    xH_0 = 100 * float(const.xhubble)  # H_0 in km/s/Mpc

    z = np.asarray(z, dtype=np.float64)

    fac = omega_lambda + (1.0 - omega_lambda - Omega_m) * (1.0 + z)**2 + Omega_m * (1.0 + z)**3

    H = xH_0 * np.sqrt(fac)

    return H if H.ndim else float(H)

def _print_time(start, end, funcname):
    """
    Routine to print elapsed time in a readable way
    """
    elapsed = end - start

    days, rem = divmod(elapsed, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{int(days)}d")
    if hours:
        parts.append(f"{int(hours)}h")
    if minutes:
        parts.append(f"{int(minutes)}m")
    parts.append(f"{seconds:.2f}s")  # Always include seconds

    print(f"Total time for {funcname}:", "".join(parts))

class State:
    """
    Holds characteristic scales, grid, physical variables, time tracking,
    and simulation diagnostics. Constructed from a Config object.
    """

    def __init__(self, config):
        from pygtfcode.io.write import make_dir, write_metadata, write_profile_snapshot

        self.config = config
        self.char = self._set_param()
        if self.config.init.profile == 'truncated_nfw': # Numerical integrations for non-analytic truncated NFW profile
            from pygtfcode.profiles.truncated_nfw import integrate_potential, generate_rho_lookup
            self.rho_interp = generate_rho_lookup(config)
            self.rcut, self.config.grid.rmax, self.pot_interp, self.pot_rad, self.pot = integrate_potential(config, self.rho_interp)

    @classmethod
    def from_config(cls, config):
        """
        Create a State object from a Config object.

        Parameters
        ----------
        config : Config
            Configuration object containing simulation parameters.

        Returns
        -------
        State
            A new State object initialized with the given configuration.
        """
        from pygtfcode.io.write import make_dir, write_metadata, write_profile_snapshot

        state = cls(config)
        state.reset()                                    # Initialize all state variables

        make_dir(state)                                  # Create the model directory if it doesn't exist
        write_metadata(state)                            # Write model metadata to disk
        write_profile_snapshot(state, initialize=True)   # Write initial snapshot to disk

        return state

    @classmethod
    def from_dir(cls, model_dir: str, snapshot: None | int = None):
        """
        Create a State object from an existing model directory.

        Parameters
        ----------
        model_dir : str
            Path to the model directory containing simulation data.
        snapshot : int, optional
            Snapshot index to load. If None, loads the latest snapshot.

        Returns
        -------
        State
            A new State object initialized with data from the specified directory.
        """
        # Check directory exists
        p = Path(model_dir)
        if not p.is_dir():
            raise FileNotFoundError(f"Model directory does not exist: {p}")
        
        # Imports
        from pygtfcode.io.read import import_metadata, load_snapshot_bundle
        from pygtfcode.config import Config

        meta = import_metadata(p)
        snapshot_bundle = load_snapshot_bundle(p, snapshot=snapshot)

        # Construct config and state

        config = Config.from_dict(meta)
        if config.io.chatter:
            print("Set config from metadata.")

        state = cls(config)

        if config.io.chatter:
            print("Setting state variables from snapshot...")     

        prec = config.prec

        state.r = np.insert(10**snapshot_bundle['log_r'].astype(np.float64), 0, 0.0)
        state.rmid = 10**snapshot_bundle['log_rmid'].astype(np.float64)
        state.m = np.insert(snapshot_bundle['m'].astype(np.float64), 0, 0.0)
        state.rho = snapshot_bundle['rho'].astype(np.float64)
        state.v2 = snapshot_bundle['v2'].astype(np.float64)
        state.p = snapshot_bundle['p'].astype(np.float64)
        state.trelax = snapshot_bundle['trelax'].astype(np.float64)
        state.kn = snapshot_bundle['kn'].astype(np.float64)
        state.t = float(snapshot_bundle['time'])
        state.step_count = int(snapshot_bundle['step_count'])
        state.snapshot_index = int(snapshot_bundle['snapshot_index'])

        state.dt = float(prec.eps_dt)
        state.du_max = float(prec.eps_du)
        state.dr_max = float(prec.eps_dr)

        state.maxvel = float(np.sqrt(np.max(state.v2)))
        state.minkn = float(np.min(state.kn))
        state.mintrelax = float(np.min(state.trelax))

        # For diagnostics
        state.n_iter_cr = 0
        state.n_iter_dr = 0
        state.dt_cum = 0.0
        state.dr_max_cum = 0.0
        state.du_max_cum = 0.0
        state.dt_over_trelax_cum = 0.0

        if config.io.chatter:
            print("State loaded.")

        return state

    def _set_param(self):
        """
        Compute and set characteristic physical quantities based on InitParams.
        """
        from pygtfcode.parameters.char_params import CharParams
        from pygtfcode.profiles.nfw import fNFW

        if self.config.io.chatter:
            print("Computing characteristic parameters for simulation...")
        init = self.config.init # Access the InitParams object from config
        sim = self.config.sim # Access the SimParams object from config

        char = CharParams() # Instantiate CharParams object

        # Ensure double point precision
        Mvir  = float(init.Mvir)
        cvir  = float(init.cvir)
        z     = float(init.z)

        rvir = 0.169 * (Mvir / 1.0e12)**(1.0/3.0)
        rvir *= (float(const.Delta_vir) / 178.0)**(-1.0/3.0)
        rvir *= (_xH(z, const) / (100.0 * float(const.xhubble)))**(-2.0/3.0)
        rvir /= float(const.xhubble)

        Mvir_h = Mvir / float(const.xhubble)
        char.fc = float(fNFW(cvir))
        char.r_s = rvir / cvir

        if init.profile != 'abg':
            char.m_s = Mvir_h / char.fc
            
        else:
            from pygtfcode.profiles.abg import chi
            char.chi = float(chi(self.config))
            char.m_s = Mvir_h / char.chi
            char.r_s *= ( char.fc / char.chi )**(1.0/3.0)

        char.rho_s = char.m_s / ( 4.0 * np.pi * char.r_s**3 )
        char.v0 = float(np.sqrt(const.gee * char.m_s / char.r_s))
        sigma0 = 4.0 * np.pi * char.r_s**2 / char.m_s # In Mpc^2 / Msun^2
        char.sigma0 = sigma0 * float(const.Mpc_to_cm)**2 / float(const.Msun_to_gram) # In cm^2 / g

        v0_cgs = char.v0 * 1.0e5
        rho_s_cgs = char.rho_s * float(const.Msun_to_gram) / float(const.Mpc_to_cm)**3
        char.t0 = 1.0 / (float(sim.a) * float(sim.sigma_m) * v0_cgs * rho_s_cgs)
        char.sigma_m_char = float(sim.sigma_m) / char.sigma0 # sigma_m in dimensionless form

        return char  # Store the CharParams object in config
    
    def _setup_grid(self):
        """
        Constructs the radial grid in log-space between rmin and rmax.

        Parameters
        ----------
        config : Config
            The simulation configuration object.

        Returns
        -------
        r : ndarray of shape (ngrid + 1,)
            Radial Lagrangian grid points, with r[0] = 0 and the rest spaced
            logarithmically between rmin and rmax.
        """
        if self.config.io.chatter:
            print("Setting up radial grid...")

        rmin  = float(self.config.grid.rmin)
        rmax  = float(self.config.grid.rmax)
        ngrid = int(self.config.grid.ngrid)

        xlgrmin = float(np.log10(rmin))
        xlgrmax = float(np.log10(rmax))

        r = np.empty(ngrid + 1, dtype=np.float64)
        r[0] = 0.0
        r[1:] = 10.0 ** np.linspace(xlgrmin, xlgrmax, ngrid, dtype=np.float64)

        return r
    
    def _initialize_grid(self):
        """
        Computes initial physical quantities on the radial grid using the
        initial profile defined in config.

        Sets the following attributes:
            - m: Enclosed mass at r[i+1]
            - rho: Density in each shell (size ngrid)
            - p: Pressure in each shell
            - u: Internal energy in each shell
            - v2: Velocity dispersion squared in each shell
            - kn: Knudsen number in each shell
            - maxvel: maximum velocity dispersion
            - minkn: minimum Knudsen number
        """
        from pygtfcode.profiles.profile_routines import menc, sigr

        if self.config.io.chatter:
            print("Initializing profiles...")

        r = self.r.astype(np.float64, copy=False)
        r_mid = 0.5 * (r[1:] + r[:-1])          # Midpoint of each shell
        dr3 = r[1:]**3 - r[:-1]**3              # Volume difference per shell

        m = np.zeros_like(r, dtype=np.float64)
        m[1:] = menc(r[1:], self)             # m[i] at shell edges

        v2 = np.asarray(sigr(r_mid, self), dtype=np.float64)
        rho = 3.0 * ( m[1:] - m[:-1] ) / dr3
        p = rho * v2
        u = 1.5 * v2
        kn = 1.0 / (self.char.sigma_m_char * np.sqrt(p))
        trelax = 1.0 / (np.sqrt(v2) * rho)

        # Apply central smoothing if using regular NFW profile (imode = 1)
        # This helps reduce artificial gradients in innermost cell
        if self.config.init.profile == "nfw":
            r1 = r[1]
            rho_c_ideal = 1.0 / (r1 * (1.0 + r1)**2)
            rho[0] = 2.0 * rho_c_ideal - rho[1]

            dr_ratio = (r[2] - r[0]) / (r[3] - r[1])
            p[0] = p[1] - dr_ratio * (p[2] - p[1])

            v2[0] = p[0] / rho[0]
            u[0] = 1.5 * v2[0]

        self.m = m
        self.rmid = r_mid
        self.rho = rho
        self.p = p
        self.u = u
        self.v2 = v2
        self.kn = kn
        self.trelax = trelax

    def _ensure_virial_equilibfrium(self):
        """
        Fine-tunes initial profile to ensure hydrostatic equilibrium.
        Iteratively runs revirialize() until max |dr/r| < eps_dr.
        """
        from pygtfcode.evolve.hydrostatic import revirialize
        chatter = self.config.io.chatter

        if chatter:
            print("Ensuring initial hydrostatic equilibrium...")

        r_new = self.r.astype(np.float64, copy=True)
        rho_new = self.rho.astype(np.float64, copy=True)
        p_new = self.p.astype(np.float64, copy=True)
        m = self.m.astype(np.float64, copy=False)

        eps_dr = float(self.config.prec.eps_dr)

        i = 0
        while True:
            i += 1
            r_new, rho_new, p_new, dr_max_new = revirialize(r_new, rho_new, p_new, m)
            if dr_max_new < eps_dr:
                break
            if i >= 100:
                raise RuntimeError("Failed to achieve hydrostatic equilibrium in 100 iterations")
            
        v2_new = p_new / rho_new
        self.r = r_new
        self.rho = rho_new
        self.p = p_new
        self.v2 = v2_new

        self.rmid = 0.5 * (r_new[1:] + r_new[:-1])
        self.u = 1.5 * v2_new
        self.kn = 1.0 / (self.char.sigma_m_char * np.sqrt(p_new))
        self.trelax = 1.0 / (np.sqrt(v2_new) * rho_new)

        if chatter:
            print(f"Hydrostatic equilibrium achieved in {i} iterations. Max |dr/r|/eps_dr = {dr_max_new/eps_dr:.2e}")

    def reset(self):
        """
        Resets initial state
        """
        config = self.config
        prec = config.prec

        self.r = self._setup_grid()
        self._initialize_grid()
        self._ensure_virial_equilibfrium()

        self.t = 0.0                        # Current time in simulation units
        self.step_count = 0                 # Global integration step counter (never reset)
        self.snapshot_index = 0             # Counts profile output snapshots
        self.dt = 1e-6                      # Initial time step (will be updated adaptively)
        self.du_max = prec.eps_du           # Initialize the max du to upper limit
        self.dr_max = prec.eps_dr           # Initialize the max dr to upper limit

        self.maxvel = float(np.sqrt(np.max(self.v2)))
        self.minkn = float(np.min(self.kn))
        self.mintrelax = float(np.min(self.trelax))

        # For diagnostics
        self.n_iter_cr = 0
        self.n_iter_dr = 0
        self.dt_cum = 0.0
        self.dr_max_cum = 0.0
        self.du_max_cum = 0.0
        self.dt_over_trelax_cum = 0.0

        if config.io.chatter:
            print("State initialized.")

    def run(self, steps=None, stoptime=None, rho_c=None):
        """
        Run the simulation until a halting criterion is met.
        User can set halting criteria to run for a specified duration.
        These are overridden by the halting criteria in self.config.

        Arguments 
        ---------
        steps : int, optional
            Number of steps to advance the simulation
        stoptime : float, optional
            Amount of simulation time by which to advance the simulation
        rho_c: float, optional
            Max central denisty value to advance until
        """
        from pygtfcode.evolve.integrator import run_until_stop
        from pygtfcode.io.write import write_log_entry, write_profile_snapshot, write_time_evolution
        from time import time as _now

        start = _now()
        start_step = self.step_count

        # Prepare kwargs for run_until_stop if any halting criteria are provided
        kwargs = {}
        if steps is not None:
            kwargs['steps'] = steps
        if stoptime is not None:
            kwargs['stoptime'] = stoptime
        if rho_c is not None:
            kwargs['rho_c'] = rho_c

        # Write initial state to disk 
        write_profile_snapshot(self)
        write_time_evolution(self)
        write_log_entry(self, start_step)

        # Integrate forward in time until a halting criterion is met
        run_until_stop(self, start_step, **kwargs)

        # Write final state to disk
        write_profile_snapshot(self)
        write_time_evolution(self)
        write_log_entry(self, start_step)

        end = _now()
        _print_time(start, end, funcname="run()")
        
    def get_phys(self):
        """
        Method to print characteristic quantities in physical units
        """
        from pygtfcode.profiles.profile_routines import menc
        char = self.char
        init = self.config.init

        Mtot = menc(self.config.grid.rmax, self, chatter=False) * char.m_s
        rvir = 0.169 * (init.Mvir / 1.0e12)**(1/3)
        rvir *= (const.Delta_vir / 178.0)**(-1.0/3.0)
        rvir *= (_xH(init.z, const) / (100 * const.xhubble))**(-2/3)
        rvir /= const.xhubble
        vvir = np.sqrt(const.gee * init.Mvir / const.xhubble / rvir)

        params_dict = {
            'log[Mvir/Msun]'            : np.log10(init.Mvir / const.xhubble),
            'log[Mtot/Msun]'            : np.log10(Mtot),
            'Vvir [km/s]'               : vvir,
            'v_0 [km/s]'                : char.v0,
            'log[rho_s/(Msun/kpc^3)]'   : np.log10(char.rho_s * 1.0e-9),
            'r_s [kpc]'                 : char.r_s * 1.0e3,
            't_0 [Gyr]'                 : char.t0 * const.sec_to_Gyr
        }

        return params_dict

    def plot_time_evolution(self, **kwargs):
        """
        Plot any time-evolution quantity vs. time for for the simulation represented by
        the State object

        Arguments
        ---------
        quantity : str, optional
            Key from the time_evolution.txt file to plot on the y-axis.
            Default is 'rho_c'.
            Options are 't_phys', 'rho_c', 'rho_c_phys', 'v_max', 'v_max_phys', 'kn_min', 'mintrel', 'mintrel_phys'.
        ylabel : str, optional
            Custom y-axis label. Defaults to quantity.
        logy : bool, optional
            Use logarithmic scale on y-axis. Default is True.
        filepath : str, optional
            If specified, saves the figure to this path.
        show : bool, optional
            If True, show the plot even if saving.  Default is False.
        grid : bool, optional
            If True, shows grid on axis
        """
        from pygtfcode.plot.time_evolution import plot_time_evolution

        plot_time_evolution(self, **kwargs)

    def plot_snapshots(self, **kwargs):
        """
        Method to plot up to three profiles at specified points in time for the simulation represented by
        the State object

        Arguments
        ---------
        snapshots : int or list of int, optional
            Snapshot indices to plot, default is the current state
        profiles : str or list of str, optional
            Profiles to plot.  Options are 'rho', 'm', 'v2', 'p', 'trelax', 'kn'
        filepath : str, optional
            If provided, save the plot to this file.
        show : bool, optional
            If True, show the plot even if saving.  Default is False.
        grid : bool, optional
            If True, shows grid on axes
        """
        from pygtfcode.plot.snapshot import plot_snapshots

        snapshots = kwargs.pop('snapshots', -1)
        plot_snapshots(self, snapshots=snapshots, **kwargs)
        
    def make_movie(self, **kwargs):
        """
        Method to animate up to three profiles for the simulation represented by
        the State object

        Arguments
        ---------
        filepath : str, optional
            Save the plot to this file.  Defaults to '/base_dir/ModelXXX/movie_{profiles}.mp4'
        profiles : str or list of str, optional
            Profiles to plot.  Options are 'rho', 'm', 'v2', 'p', 'trelax', 'kn'
        grid : bool, optional
            If True, shows grid on axes
        fps : int, optional
            Frames per second for the output movie. Default is 20

        Returns
        -------
        None
            Saves the movie as an MP4 file in the model directory.
        """
        from pygtfcode.plot.snapshot import make_movie

        make_movie(self, **kwargs)

    def __repr__(self):
        # Copy the __dict__ and omit the 'config' key
        filtered = {k: v for k, v in self.__dict__.items() if k != "config"}
        return f"{self.__class__.__name__}(\n{pprint.pformat(filtered, indent=2)}\n)"

