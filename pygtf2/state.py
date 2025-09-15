import numpy as np
from pygtf2.parameters.constants import Constants as const
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
        from pygtf2.io.write import make_dir, write_metadata, write_profile_snapshot

        self.config = config
        if config.s < 1:
            raise ValueError("No species defined; add at least one before instantiating a State.")
        self._set_species_hierarchy()
        self.char = self._set_param()

        # Check for truncated NFW profile - numerically integrate potential
        for name in self.labels:
            first = True
            if self.config.spec[name].init.profile == 'truncated_nfw':
                print(f"Computing truncated NFW potential for species {name}:")
                from pygtf2.profiles.truncated_nfw import integrate_potential, generate_rho_lookup
                prec = config.prec
                chatter = config.io.chatter
                init = config.spec[name].init
                if first:
                    self.rho_interp, self.rcut, self.pot_interp, self.pot_rad, self.pot = ({} for _ in range(5))
                self.rho_interp[name] = generate_rho_lookup(init, prec, chatter)
                self.rcut[name], config.grid.rmax, self.pot_interp[name], self.pot_rad[name], self.pot[name] = integrate_potential(
                    init, config.grid, chatter, self.rho_interp[name]
                    )
                first = False

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
        from pygtf2.io.write import make_dir, write_metadata, write_profile_snapshot

        state = cls(config)
        state.reset()                                    # Initialize all state variables

        make_dir(state)                                  # Create the model directory if it doesn't exist
        write_metadata(state)                            # Write model metadata to disk
        write_profile_snapshot(state, initialize=True)   # Write initial snapshot to disk

        return state

    @classmethod
    def from_dir(cls, model_dir: str, snapshot: None | int = None):
        """
        Create a State object from an existing model directory (multi-species aware).

        Parameters
        ----------
        model_dir : str
            Path to the model directory containing simulation data.
        snapshot : int, optional
            Snapshot index to load. If None, loads the latest snapshot.

        Returns
        -------
        State
        """
        # --- basic checks
        p = Path(model_dir)
        if not p.is_dir():
            raise FileNotFoundError(f"Model directory does not exist: {p}")
        
        # --- imports
        from pygtf2.io.read import import_metadata, load_snapshot_bundle
        from pygtf2.config import Config

        # --- metadata + snapshot
        meta = import_metadata(p)
        snap = load_snapshot_bundle(p, snapshot=snapshot)

        # --- construct config and state
        config = Config.from_dict(meta)
        if config.io.chatter:
            print("Set config from metadata.")

        state = cls(config)
        if config.io.chatter:
            print("Setting state variables from snapshot...") 

        # ===== Grid (aligned across species at snapshot time) =====
        # log_r are the outer edges (i=1..N) in log10; prepend r[0]=0
        log_r = snap['log_r'].astype(np.float64)        # (N,)
        r_edges = np.empty(log_r.size + 1, dtype=np.float64)
        r_edges[0]  = 0.0
        r_edges[1:] = 10.0**log_r                       # (N+1,)

        # log_rmid are shell centers (common for all species at snapshot time)
        rmid_1d = 10.0**snap['log_rmid'].astype(np.float64)   # (N,)

        # species ordering: use config.spec keys (in insertion order)
        labels = list(config.spec.keys())
        s = len(labels)
        N = rmid_1d.size
        Np1 = N + 1

        # broadcast per-species grids
        state.r    = np.broadcast_to(r_edges, (s, Np1)).copy()   # (s, N+1)
        state.rmid = np.broadcast_to(rmid_1d, (s, N)).copy()     # (s, N)

        # ===== Totals (from file) =====
        # Totals are single arrays at snapshot time
        # m_tot is given at outer edges (i=1..N); prepend 0.0 for r[0]
        m_tot_edges = np.empty(Np1, dtype=np.float64)
        m_tot_edges[0]  = 0.0
        m_tot_edges[1:] = snap['m_tot'].astype(np.float64)       # (N,)
        state.m_tot   = m_tot_edges                               # (N+1,)
        state.rho_tot = snap['rho_tot'].astype(np.float64)        # (N,)
        state.v2_tot  = snap['v2_tot'].astype(np.float64)         # (N,)
        state.p_tot   = snap['p_tot'].astype(np.float64)          # (N,)
        state.u_tot   = 1.5 * state.v2_tot.copy()

        # ===== Per-species fields =====
        # Allocate per-species arrays
        state.m      = np.zeros((s, Np1), dtype=np.float64)
        state.rho    = np.zeros((s, N),   dtype=np.float64)
        state.v2     = np.zeros((s, N),   dtype=np.float64)
        state.p      = np.zeros((s, N),   dtype=np.float64)
        state.trelax = np.zeros((s, N),   dtype=np.float64)
        state.u      = np.zeros((s, N),   dtype=np.float64)

        species_block = snap['species']   # dict: name -> dict of arrays

        # For each species name in the config, pull data from snapshot
        for k, name in enumerate(labels):
            if name not in species_block:
                raise KeyError(f"Species '{name}' in config not found in snapshot file.")
            sd = species_block[name]
            # m is given at outer edges (length N); prepend 0.0 for r[0]
            m_edges = np.empty(Np1, dtype=np.float64)
            m_edges[0]  = 0.0
            m_edges[1:] = sd['m'].astype(np.float64)
            state.m[k]      = m_edges
            state.rho[k]    = sd['rho'].astype(np.float64)
            state.v2[k]     = sd['v2'].astype(np.float64)
            state.p[k]      = sd['p'].astype(np.float64)
            state.trelax[k] = sd['trelax'].astype(np.float64)
            state.u[k]      = 1.5 * state.v2[k].copy()

        # ===== Time + bookkeeping =====
        state.t             = float(snap['time'])
        state.step_count    = int(snap['step_count'])
        state.snapshot_index= int(snap['snapshot_index'])

        # thresholds / proposed dt
        prec         = config.prec
        state.dt     = float(prec.eps_dt)
        state.du_max = float(prec.eps_du)
        state.dr_max = float(prec.eps_dr)

        # quick diagnostics (global)
        state.maxvel     = float(np.sqrt(np.max(state.v2_tot)))
        state.mintrelax  = float(np.min(state.trelax))

        # running diagnostics
        state.n_iter_cr = 0
        state.n_iter_dr = 0
        state.dt_cum = 0.0
        state.dr_max_cum = 0.0
        state.du_max_cum = 0.0
        state.dt_over_trelax_cum = 0.0

        if config.io.chatter:
            print("State loaded.")

        return state

    def _set_species_hierarchy(self):
        """
        Validate species and set the species hierarchy
        Populates:
            self.labels : (s,) array[str]
            self.m_part : (s,) float64
            self.frac   : (s,) float64  (sums to 1 within tol; renormalized if needed)
        """
        config = self.config
        if config.io.chatter:
            print("Setting species hierarchy...")
        spec = config.spec
        s = config.s
    
        if s < 1:
            raise ValueError("No species defined in Config.spec.")

        labels_list = []
        m_part = np.empty(s, dtype=np.float64)
        frac   = np.empty(s, dtype=np.float64)

        # Import species parameters
        for ind, label in enumerate(spec):
            labels_list.append(label)
            m_part[ind] = spec[label].m_part
            frac[ind] = spec[label].frac
        labels = np.array(labels_list, dtype=object)

        # Validate fractions
        if np.any(frac <= 0.0):
            negs = labels[frac <= 0.0]
            raise ValueError(f"All species mass fractions must be > 0. Offenders: {list(negs)}")

        sfrac = float(frac.sum())
        if not np.isfinite(sfrac) or sfrac <= 0.0:
            raise ValueError("Species mass fractions sum is non-finite or <= 0.")
        if abs(sfrac - 1.0) > 0.0:
            # If close, renormalize; if far, error out.
            if abs(sfrac - 1.0) <= 1e-5:
                frac = frac / sfrac
            else:
                raise ValueError(f"Mass fractions must sum to 1.0 (got {sfrac:.6g}).")

        # Sort by descending particle mass
        order = np.argsort(-m_part)
        m_part = m_part[order]
        frac   = frac[order]
        labels = labels[order]

        # Store as attributes
        self.labels = labels           # array[str]
        self.m_part = m_part           # array[float64]
        self.frac   = frac             # array[float64]
        self.mrat   = m_part / m_part[0]

    def _set_param(self):
        """
        Compute and set characteristic physical quantities.
        lnL, mrat, 
        """
        from pygtf2.parameters.char_params import CharParams
        from pygtf2.profiles.nfw import fNFW

        config = self.config
        if config.io.chatter:
            print("Computing characteristic parameters for simulation...")
        sim = config.sim

        char = CharParams() # Instantiate CharParams object

        #--- Set r_s and m_s ---
        mtot  = float(config.mtot)

        # Choose initial profile of most massive particle mass for setting scales
        kref = int(np.argmax(self.m_part))
        label_ref = self.labels[kref]
        init_ref = config.spec[label_ref].init
        rs      = init_ref.r_s
        profile = init_ref.profile

        # --- Virial radius (global, from mtot) ---
        z = float(getattr(init_ref, "z", 0.0))
        rvir = 0.169 * (mtot / 1.0e12)**(1.0/3.0)
        rvir *= (float(const.Delta_vir) / 178.0)**(-1.0/3.0)
        rvir *= (_xH(z, const) / (100.0 * float(const.xhubble)))**(-2.0/3.0)
        rvir /= float(const.xhubble)

        if profile in ['abg']:
            from pygtf2.profiles.abg import chi
            char.chi = float(chi(self.config.prec, init_ref))

            if rs is not None: # User specified scale radius
                char.r_s = float(rs)
                cvir = rvir / rs
            else: # User specified concentration parameter
                cvir = init_ref.cvir
                if cvir is None:
                    raise RuntimeError("Either cvir or rs must be specified in the initial profile")
                char.fc = float(fNFW(cvir))
                char.r_s = (rvir / cvir) * ( char.fc / char.chi )**(1.0/3.0)

            char.m_s = mtot
                
        elif profile in ['nfw', 'truncated_nfw']:
            if rs is None: # cvir is specified
                cvir = init_ref.cvir
                if cvir is None:
                    raise RuntimeError("Either cvir or rs must be specified in the initial profile")
                char.r_s = rvir / cvir

            else: # rs is specified
                char.r_s = float(rs)
                cvir = rvir / rs

            char.fc = float(fNFW(cvir))
            char.m_s = mtot / float(const.xhubble) / char.fc

        #--- Set rho_s and v0 ---
        char.rho_s = char.m_s / ( 4.0 * np.pi * char.r_s**3 )
        char.v0 = float(np.sqrt(const.gee * char.m_s / char.r_s))

        #--- Set Coulomb logarithm and t0 --- 
        s = config.s
        m_part = self.m_part
        lnL = np.empty((s,s), dtype=np.float64)
        for i in range(s):
            for j in range(s):
                lnL[i,j] = np.log(sim.lnL_param * 2.0 * char.m_s) / (m_part[i] + m_part[j])

        lnL_term = lnL[0,0] if s == 1 else lnL[0,s - 1]

        t0 = char.v0**3.0 / (12.0 * np.pi * const.gee**2.0 * m_part[kref] * char.rho_s * lnL_term)
        char.t0 = t0 * const.kpc_to_km * const.sec_to_Gyr

        char.lnL = lnL / lnL_term

        #--- Set luminosity calculation parameter ---
        char.c1 = 1.0 / np.sqrt(3.0 * np.pi)
        char.c2 = (np.sqrt(2.0) / 9.0) * sim.alpha * sim.beta * sim.b

        return char  # Store the CharParams object in config
    
    def _setup_grid(self):
        """
        Constructs the Lagrangian radial grid in log-space between rmin and rmax.

        Parameters
        ----------
        config : Config
            The simulation configuration object.

        Returns
        -------
        r : ndarray, shape (s, ngrid+1)
            r[k, :] are the edge radii for species k, with r[k,0] = 0.0 and
            r[k,1:] logarithmically spaced between rmin and rmax (common grid).
        """
        config = self.config
        if config.io.chatter:
            print("Setting up radial grids for all species...")

        rmin  = float(config.grid.rmin)
        rmax  = float(config.grid.rmax)
        ngrid = int(config.grid.ngrid)

        xlgrmin = float(np.log10(rmin))
        xlgrmax = float(np.log10(rmax))

        # Common log-spaced edges (excluding the central point which we set to 0)
        edges = np.empty(ngrid + 1, dtype=np.float64)
        edges[0] = 0.0
        edges[1:] = 10.0 ** np.linspace(xlgrmin, xlgrmax, ngrid, dtype=np.float64)

        # Tile/broadcast for each species: r[k, :] = edges
        r = np.broadcast_to(edges, (config.s, ngrid + 1)).copy()

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
            - trelax: Relaxation time in each shell
        """
        from pygtf2.profiles.profile_routines import menc, sigr
        config = self.config
        prec = config.prec
        spec = config.spec
        labels = self.labels
        frac = self.frac
        chatter = config.io.chatter
        if chatter:
            print("Initializing profiles...")

        r = self.r.astype(np.float64, copy=False)
        r_mid = 0.5 * (r[:, 1:] + r[:, :-1])          # Midpoint of each shell
        dr3 = r[:, 1:]**3 - r[:, :-1]**3              # Volume difference per shell

        m = np.zeros_like(r, dtype=np.float64)
        v2 = np.zeros_like(r_mid, dtype=np.float64)

        # Compute m and v2 for all radial bins
        for i, name in enumerate(labels):
            init = spec[name].init

            # kwargs needed for non-analytic truncated NFW profile
            pot_rad = pot_interp = rho_interp = rcut = None
            if init.profile == 'truncated_nfw':
                pot_rad = self.pot_rad[name]
                pot_interp = self.pot_interp[name]
                rho_interp = self.rho_interp[name]
                rcut = self.rcut[name]
            
            m_base = menc(self.r[i, 1:], init, prec, 
                          chatter=chatter, pot_rad=pot_rad, pot_interp=pot_interp, rho_interp=rho_interp)
            m[i, 1:] = frac[i] * m_base                 # Scale by mass fraction
            v2[i, :] = sigr(r_mid[i, :], init, prec,
                            chatter=chatter, grid=config.grid, rcut=rcut, 
                            pot_rad=pot_rad, pot_interp=pot_interp, rho_interp=rho_interp)

        # Rho, u, and p from equation of state
        rho = 3.0 * ( m[:, 1:] - m[:, :-1] ) / dr3
        u = 1.5 * v2
        p = rho * v2

        # Central smoothing for NFW profile
        for i, name in enumerate(labels):
            if spec[name].init.profile == 'nfw':
                r1 = r[i, 1]
                rho_c_ideal = 1.0 / (r1 * (1.0 + r1)**2)
                rho[i, 0] = 2.0 * rho_c_ideal - rho[i, 1]
                dr_ratio = (r[i, 2] - r[i, 0]) / (r[i, 3] - r[i, 1])
                p[i, 0] = p[i, 1] - dr_ratio * (p[i, 2] - p[i, 1])
                v2[i, 0] = p[i, 0] / rho[i, 0]
                u[i, 0] = 1.5 * v2[i, 0]

        trelax = 1.0 / (np.sqrt(v2) * rho)

        # Compute totals
        m_tot   = m.sum(axis=0)
        rho_tot = rho.sum(axis=0)
        p_tot   = p.sum(axis=0)
        v2_tot = p_tot / rho_tot
        u_tot  = 1.5 * v2_tot

        self.m          = m
        self.rmid       = r_mid
        self.rho        = rho
        self.p          = p
        self.u          = u
        self.v2         = v2
        self.trelax     = trelax

        self.m_tot      = m_tot
        self.rho_tot    = rho_tot
        self.p_tot      = p_tot
        self.v2_tot     = v2_tot
        self.u_tot      = u_tot

    def _ensure_virial_equilibrium(self):
        """
        Fine-tunes initial profile to ensure hydrostatic equilibrium.
        Iteratively runs revirialize() until max |dr/r| < eps_dr.
        """
        from pygtf2.evolve.hydrostatic import revirialize
        from pygtf2.evolve.realign import realign
        chatter = self.config.io.chatter

        if chatter:
            print("Ensuring initial hydrostatic equilibrium...")

        r_new = self.r.astype(np.float64, copy=True)
        rho_new = self.rho.astype(np.float64, copy=True)
        p_new = self.p.astype(np.float64, copy=True)
        m_tot_new = self.m_tot.astype(np.float64, copy=False)

        eps_dr = float(self.config.prec.eps_dr)

        i = 0
        while True:
            i += 1
            r_new, rho_new, p_new, dr_max_new = revirialize(r_new, rho_new, p_new, m_tot_new)
            v2_new = p_new / rho_new
            r_new, rho_new, v2_new, p_new, m_new, m_tot_new = realign(r_new, rho_new, v2_new)
            if dr_max_new < eps_dr:
                break
            if i >= 100:
                raise RuntimeError("Failed to achieve hydrostatic equilibrium in 100 iterations")
            
        self.r = r_new
        self.rho = rho_new
        self.p = p_new
        self.v2 = v2_new
        self.m = m_new
        self.rmid = 0.5 * (r_new[:, 1:] + r_new[:, :-1])
        self.u = 1.5 * v2_new
        self.trelax = 1.0 / (np.sqrt(v2_new) * rho_new)
        
        self.m_tot      = m_tot_new
        self.rho_tot    = rho_new.sum(axis=0)
        self.p_tot      = p_new.sum(axis=0)
        self.v2_tot     = self.p_tot / self.rho_tot
        self.u_tot      = 1.5 * self.v2_tot

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
        self._ensure_virial_equilibrium()

        self.t = 0.0                        # Current time in simulation units
        self.step_count = 0                 # Global integration step counter (never reset)
        self.snapshot_index = 0             # Counts profile output snapshots
        self.dt = 1e-7                      # Initial time step (will be updated adaptively)
        self.du_max = prec.eps_du           # Initialize the max du to upper limit
        self.dr_max = prec.eps_dr           # Initialize the max dr to upper limit

        self.maxvel = float(np.sqrt(np.max(self.v2)))
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
        from pygtf2.evolve.integrator import run_until_stop
        from pygtf2.io.write import write_log_entry, write_profile_snapshot, write_time_evolution
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

        # # Write final state to disk
        write_profile_snapshot(self)
        write_time_evolution(self)
        write_log_entry(self, start_step)

        end = _now()
        _print_time(start, end, funcname="run()")

    def plot_time_evolution(self, **kwargs):
        """
        Plot any time-evolution quantity vs. time for for the simulation represented by
        the State object

        Arguments
        ---------
        quantity : str, optional
            Key from the time_evolution.txt file to plot on the y-axis.
            Default is 'rho_c'.
            Options are 'rho_c', 'v_max', 'mintrel', 'r_enc'.
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
        from pygtf2.plot.time_evolution import plot_time_evolution

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
        from pygtf2.plot.snapshot import plot_snapshots

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
        from pygtf2.plot.snapshot import make_movie

        make_movie(self, **kwargs)

    def __repr__(self):
        # Copy the __dict__ and omit the 'config' key
        filtered = {k: v for k, v in self.__dict__.items() if k != "config"}
        return f"{self.__class__.__name__}(\n{pprint.pformat(filtered, indent=2)}\n)"

