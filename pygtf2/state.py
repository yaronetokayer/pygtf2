import numpy as np
from pygtf2.parameters.constants import Constants as const
import pprint
from pathlib import Path

def _xH(z, const):
    """
    Returns H(z) in units of km/s/Mpc using cosmological parameters
    supplied by the const argument.

    Parameters
    ----------
    z : float or array-like
        Redshift.

    const : Constants
        Configuration object containing cosmological parameters.

    Returns
    -------
    H_z : float or ndarray or array-like
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
    Mutable multi-species simulation state and characteristic scales.

    Use State.from_config(config) to initialize the grids and output files.
    The constructor alone sets species ordering, scales, and profile lookup
    functions; it does not initialize all evolving arrays. The Config is
    retained by reference. Species are ordered by descending particle mass.
    Radial edges and enclosed masses have shape (s, ngrid+1); densities and
    one-dimensional velocity dispersions squared have shape (s, ngrid).
    Evolving arrays and time use dimensionless simulation units; char stores
    the physical conversions. Loading a saved State is not yet supported.
    """

    def __init__(self, config):
        self.config = config
        if config.s < 1:
            raise ValueError("No species defined; add at least one before instantiating a State.")
        self._set_species_hierarchy()
        self.char = self._set_param()

        # Check for truncated NFW profile - numerically integrate potential
        first = True
        for name in self.labels:
            if self.config.spec[name].init.prof == 'truncated_nfw':
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
            if self.config.spec[name].init.prof == 'king':
                print(f"Computing King potential for species {name}:")
                from pygtf2.profiles.king import generate_nu_lookup, integrate_W_king
                prec = config.prec
                chatter = config.io.chatter
                init = config.spec[name].init
                if first:
                    self.nu_interp, self.rcut, self.w_interp, self.pot_rad, self.pot = ({} for _ in range(5))
                self.nu_interp[name] = generate_nu_lookup(init, prec, chatter)
                self.rcut[name], config.grid.rmax, self.w_interp[name], self.pot_rad[name], self.pot[name] = integrate_W_king(
                    init, config.grid, chatter, self.nu_interp[name]
                    )
                first = False

    @classmethod
    def from_config(cls, config):
        """
        Initialize a State in hydrostatic equilibrium and write initial outputs.

        Creates the model directory, writes metadata and characteristic scales,
        and writes snapshot 0 and its conversion-table entry. Existing files at
        these paths may be replaced, regardless of config.io.overwrite.

        Parameters
        ----------
        config : Config
            Configuration object containing simulation parameters.

        Returns
        -------
        State
            A new State object initialized with the given configuration.
        """
        from pygtf2.io.write import make_dir, write_metadata, write_profile_snapshot, write_char_params

        state = cls(config)
        state.reset()                                    # Initialize all state variables

        make_dir(state)                                  # Create the model directory if it doesn't exist
        write_char_params(state)                         # Write model characteristic parameters to disk
        write_metadata(state)                            # Write model metadata to disk
        write_profile_snapshot(state, initialize=True)   # Write initial snapshot to disk

        return state

    @classmethod
    def from_dir(cls, model_dir: str, snapshot: None | int = None):
        """
        Reserved interface for restoring a State from saved output.

        Currently always raises RuntimeError because restart support is still
        in development. Use pygtf2.io.read.load_snapshot_bundle to inspect saved
        arrays and Config.from_dict(import_metadata(...)) to recover parameters;
        neither resumes integration.
        """
        raise RuntimeError("this module is still in development")
        # --- basic checks
        p = Path(model_dir)
        if not p.is_dir():
            raise FileNotFoundError(f"Model directory does not exist: {p}")
        
        # --- imports
        from pygtf2.io.read import import_metadata, load_snapshot_bundle
        from pygtf2.config import Config
        from pygtf2.util.calc import calc_rho_v2_r_c, calc_r50_spread

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

        # species ordering: use config.spec keys (in insertion order)
        labels = list(config.spec.keys())
        s = len(labels)
        N = snap['log_rmid'].size
        Np1 = N + 1

        # ===== Per-species fields =====
        # Allocate per-species arrays
        state.r      = np.zeros((s, Np1), dtype=np.float64)
        state.m      = np.zeros((s, Np1), dtype=np.float64)
        state.rho    = np.zeros((s, N),   dtype=np.float64)
        state.v2     = np.zeros((s, N),   dtype=np.float64)
        state.trelax = np.zeros((s, N),   dtype=np.float64)
        state.u      = np.zeros((s, N),   dtype=np.float64)

        species_block = snap['species']   # dict: name -> dict of arrays

        # For each species name in the config, pull data from snapshot
        for k, name in enumerate(labels):
            if name not in species_block:
                raise KeyError(f"Species '{name}' in config not found in snapshot file.")
            sd = species_block[name]
            # m and r given at outer edges (length N); prepend 0.0
            r_edges = np.empty(Np1, dtype=np.float64)
            r_edges[0]  = 0.0
            r_edges[1:] = 10**sd['lgr'].astype(np.float64)
            m_edges = np.empty(Np1, dtype=np.float64)
            m_edges[0]  = 0.0
            m_edges[1:] = sd['m'].astype(np.float64)
            state.r[k]      = r_edges
            state.m[k]      = m_edges
            state.rho[k]    = sd['rho'].astype(np.float64)
            state.v2[k]     = sd['v2'].astype(np.float64)
            state.trelax[k] = sd['trelax'].astype(np.float64)
        state.rmid = 0.5 * (state.r[:,1:] + state.r[:, :-1])

        # ===== Time + bookkeeping =====
        state.t             = float(snap['time'])
        state.step_count    = int(snap['step_count'])
        state.snapshot_index= int(snap['snapshot_index'])

        # thresholds / proposed dt
        prec         = config.prec
        state.dt     = float(prec.eps_dt)

        # quick diagnostics (global)
        state.mintrelax                     = float(np.min(state.trelax))
        state.rho_c, state.v2_c, state.r_c  = calc_rho_v2_r_c(state.rmid, state.rho, state.v2)
        state.r50_spread                    = calc_r50_spread(state.r, state.m)

        # running diagnostics
        state.dt_cum = 0.0
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
        Return characteristic scales using the heaviest species' profile.

        Masses are in Msun, lengths in kpc, velocities in km/s, and time in Gyr.
        The returned CharParams also contains the normalized Coulomb logarithm
        matrix and the dimensionless transport coefficients c1 and c2.
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
        profile = init_ref.prof

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
            char.m_s = mtot / char.fc / float(const.xhubble)
        
        elif profile in ['exp', 'king', 'etnfw']:
            char.r_s = float(rs)
            char.m_s = float(mtot)

        #--- Set rho_s and v0 ---
        char.rho_s = char.m_s / ( 4.0 * np.pi * char.r_s**3 )
        char.v0 = float(np.sqrt(const.gee * char.m_s / char.r_s))

        #--- Set Coulomb logarithm and t0 --- 
        s = config.s
        m_part = self.m_part

        lnL = np.empty((s,s), dtype=np.float64)
        for i in range(s):
            for j in range(s):
                lnL[i,j] = np.log(sim.lnL_param * 2.0 * char.m_s / (m_part[i] + m_part[j]) )

        lnL_term = lnL[0,s - 1]

        t0 = char.v0**3.0 / (12.0 * np.pi * const.gee**2.0 * m_part[kref] * char.rho_s * lnL_term)
        char.t0 = t0 * const.kpc_to_km * const.sec_to_Gyr

        char.lnL = lnL / lnL_term

        #--- Set luminosity calculation parameter ---
        char.c1 = 1.0 / np.sqrt(3.0 * np.pi)
        char.c2 = (np.sqrt(2.0) / 9.0) * sim.alpha * sim.beta * sim.b

        return char  # Store the CharParams object in config

    def _set_bkg_param(self):
        """
        Encode the configured background as a float64 array of length four.

        Entries are [profile_code, mass/char.m_s, length/char.r_s, other].
        Codes are -1 for no background, 0 for static Hernquist, and 1 for the
        reserved, unimplemented decaying Hernquist profile. Unused entries are
        zero. A configured background also updates char.t0 and char.lnL using
        its enclosed mass at the outer grid radius.
        """
        from pygtf2.util.calc import add_bkg_pot_scalar

        config = self.config
        char = self.char
        chatter = config.io.chatter
        sim = config.sim
        bkg = sim.bkg
        if chatter:
            print("Setting parameters for background potential...")

        VALID_BKG_PROFILES = sim.VALID_BKG_PROFILES

        bkg_param = np.zeros(4, dtype=np.float64)

        prof = bkg['prof']

        # Set the profile code
        if prof is None:
            bkg_param[0] = -1
            if chatter:
                print("\tNo background potential.")
            return bkg_param

        elif isinstance(prof, str):
            try:
                idx = VALID_BKG_PROFILES.index(prof)
            except ValueError:
                raise ValueError(f"Unknown background profile '{prof}'. Valid options: {VALID_BKG_PROFILES}")
            bkg_param[0] = float(idx)

        # Set additional params
        bkg_param[1] = bkg['mass'] / char.m_s
        bkg_param[2] = bkg['length'] / char.r_s

        # Profile-specific parameters
        if prof == 'hernq_decay': # Blank
            bkg_param[3] = bkg['other']

        #--- Readjust lnL and t0
        s = config.s
        m_part = self.m_part
        kref = int(np.argmax(self.m_part))

        rmax = config.grid.rmax
        # m_bkg_lnL = bkg['mass']
        m_bkg_lnL = char.m_s * add_bkg_pot_scalar(rmax, bkg_param)

        lnL = np.empty((s,s), dtype=np.float64)
        for i in range(s):
            for j in range(s):
                lnL[i,j] = np.log(sim.lnL_param * 2.0 * (char.m_s + m_bkg_lnL) / (m_part[i] + m_part[j]) )

        lnL_term = lnL[0,s - 1]

        t0 = char.v0**3.0 / (12.0 * np.pi * const.gee**2.0 * m_part[kref] * char.rho_s * lnL_term)
        
        self.char.t0 = t0 * const.kpc_to_km * const.sec_to_Gyr
        self.char.lnL = lnL / lnL_term

        if chatter:
            print(f"\tSet background potential parameters for profile {prof}.")
        
        return bkg_param

    def _setup_grid(self):
        """
        Constructs the Lagrangian radial grid in log-space between rmin and rmax.

        Uses self.config.grid and self.config.s.

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
        Populate dimensionless species profiles on self.r.

        Sets m at all edges (shape (s, ngrid+1)), including zero at the origin,
        and rmid, rho, and v2 at shell centers (shape (s, ngrid)). Each mass
        profile is scaled by its species mass fraction. The central pressure
        is adjusted before the subsequent hydrostatic-equilibrium iteration.
        """
        from pygtf2.profiles.profile_init_routines import menc, sigr
        config = self.config
        prec = config.prec
        spec = config.spec
        labels = self.labels
        frac = self.frac
        bkg_param = self.bkg_param
        chatter = config.io.chatter
        if chatter:
            print("Initializing profiles...")

        r = self.r.astype(np.float64, copy=False)
        r_mid = 0.5 * (r[:, 1:] + r[:, :-1])          # Midpoint of each shell
        dr3 = r[:, 1:]**3 - r[:, :-1]**3              # Volume difference per shell

        m   = np.zeros_like(r, dtype=np.float64)
        v2  = np.zeros_like(r_mid, dtype=np.float64)
        rho = np.zeros_like(r_mid, dtype=np.float64)

        # Compute m and v2 for all radial bins
        for i, name in enumerate(labels):
            init = spec[name].init

            # kwargs needed for non-analytic truncated NFW profile
            kwargs = {}
            if init.prof == 'truncated_nfw':
                kwargs['pot_rad'] = self.pot_rad[name]
                kwargs['pot_interp'] = self.pot_interp[name]
                kwargs['rho_interp'] = self.rho_interp[name]
                kwargs['rcut'] = self.rcut[name]
            elif init.prof == 'king':
                kwargs['rt'] = self.rcut[name]
                kwargs['w_interp'] = self.w_interp[name]
                kwargs['nu_interp'] = self.nu_interp[name]

            m_base = menc(self.r[i, 1:], init, prec, 
                          chatter=chatter, **kwargs)
            m[i, 1:] = frac[i] * m_base                 # Scale by mass fraction
            v2[i, :] = sigr(r_mid[i, :], init, prec, bkg_param,
                            chatter=chatter, grid=config.grid, **kwargs)

        # Rho, u, and p from equation of state
        rho = 3.0 * ( m[:, 1:] - m[:, :-1] ) / dr3
        p = rho * v2

        # Central smoothing for NFW profile
        for i, name in enumerate(labels):
            if spec[name].init.prof == 'nfw':
                r1 = r[i, 1]
                rho0_ideal = 1.0 / (r1 * (1.0 + r1)**2)
                rho[i, 0] = 2.0 * rho0_ideal - rho[i, 1]
                dr_ratio = (r[i, 2] - r[i, 0]) / (r[i, 3] - r[i, 1])
                p[i, 0] = p[i, 1] - dr_ratio * (p[i, 2] - p[i, 1])
                v2[i, 0] = p[i, 0] / rho[i, 0]

        # Recompute pressure of central bin such that HE is guaranteed
        srho0 = rho[:, 1] + rho[:, 0]
        r_c = r[:, 1]
        dr_c = r[:, 2] - r[:, 0]
        m_enc_c = np.sum(m[:, 1])
        p[:, 0] = p[:, 1] + srho0 * dr_c * m_enc_c / (4.0 * r_c**2)
        v2[:, 0] = p[:, 0] / rho[:, 0]

        self.m          = m
        self.rmid       = r_mid
        self.rho        = rho
        self.v2         = v2

    def _ensure_hydrostatic_equilibrium(self):
        """
        Fine-tunes initial profile to ensure hydrostatic equilibrium.
        First update pressure with a backward sweep, then
        iteratively applies Jacobi revirialization until max |dr/r| < 1e-10.
        Raises RuntimeError on shell crossing or failure within 100 iterations.
        """
        from pygtf2.evolve.hydrostatic import revirialize_interp_jacobi_diagnostics, compute_he_pressures_with_resid, STATUS_SHELL_CROSSING
        chatter = self.config.io.chatter
        bkg_param = self.bkg_param

        if chatter:
            print("Ensuring initial hydrostatic equilibrium...")

        r_new = self.r.astype(np.float64, copy=True)
        rho_new = self.rho.astype(np.float64, copy=True)
        p_new = rho_new * self.v2.astype(np.float64, copy=True)
        m = self.m.astype(np.float64, copy=False)

        # --- Update pressure with backward sweep ---
        res_old, res_new = compute_he_pressures_with_resid(r_new, rho_new, p_new, m, bkg_param)
        if chatter:
            print(f"\tInitial pressure correction applied. HE residual improved {float(res_old):.3e} -> {float(res_new):.3e}.")

        # --- Iterative revir ---
        # Preallocate arrays
        s, Np1 = r_new.shape
        n_int = Np1 - 2
        a  = np.empty(n_int, dtype=np.float64)
        b  = np.empty(n_int, dtype=np.float64)
        c  = np.empty(n_int, dtype=np.float64)
        y  = np.empty(n_int, dtype=np.float64)
        xk  = np.empty(n_int, dtype=np.float64)
        vol_old = np.empty(Np1 - 1, dtype=np.float64)
        K_all   = np.empty((s, Np1), dtype=np.float64)
        m_tot_all = np.empty((s, Np1), dtype=np.float64)

        eps_dr = 1e-10
        i = 0
        while True:
            i += 1
            # status, dr_max_new, he_res = revirialize_interp_gs_diagnostics(r_new, rho_new, p_new, m, bkg_param)
            status, dr_max_new, he_res = revirialize_interp_jacobi_diagnostics(
                r_new, rho_new, p_new, m, bkg_param,
                a, b, c, y, xk, vol_old, K_all, m_tot_all,
                )
            
            if status == STATUS_SHELL_CROSSING:
                raise RuntimeError(f"Initial revir iter {i}: Shell crossing!")

            if dr_max_new < eps_dr:
                break

            if i >= 100:
                raise RuntimeError("Failed to achieve hydrostatic equilibrium in 100 iterations")

        # DEBUGGING
        # from pygtf2.dev.debug import plot_r_markers
        # reference = float(r_new[0,1])
        # for i in range(10000):
        #     status, dr_max_new, he_res = revirialize_interp_jacobi_diagnostics(r_new, rho_new, p_new, m, bkg_param)
        #     print(compute_he_resid_norm(r_new, rho_new, p_new, m, bkg_param))
            # compute_he_pressures(r_new, rho_new, p_new, m, bkg_param)
            # if abs(r_new[0,1])/reference - 1.0 > 1e-2:
            #     print(r_new[0,1], abs(r_new[0,1])/reference - 1.0)
            #     print(r_new[:,1:3])
            #     print(i)
            #     break
        ### END

        v2_new = p_new / rho_new

        self.r = r_new
        self.rho = rho_new
        self.v2 = v2_new
        self.rmid[:,:] = 0.5 * (r_new[:, 1:] + r_new[:, :-1])

        if chatter:
            print(f"\tHydrostatic equilibrium achieved in {i} iterations. Max |dr/r| = {dr_max_new:.2e}.  HE res {he_res}")

    def reset(self):
        """
        Reinitialize grids and profiles and reset time, counters, and diagnostics.

        Uses the current config with the existing species hierarchy, characteristic
        scales, and profile lookup tables. This is not a full rebuild after arbitrary
        configuration changes; use State.from_config for that. No output files are
        changed until a subsequent write or run, which can replace existing data.
        """
        from pygtf2.util.calc import calc_r50_spread, mass_fraction_radii
        from pygtf2.util.interpolate import sum_intensive_loglog_single

        config = self.config

        self.r = self._setup_grid()
        self.bkg_param = self._set_bkg_param()
        self._initialize_grid()
        self._ensure_hydrostatic_equilibrium()

        self.t = 0.0                        # Current time in simulation units
        self.step_count = 0                 # Global integration step counter (never reset)
        self.snapshot_index = 0             # Counts profile output snapshots
        self.dt = 1e-7                      # Initial time step (will be updated adaptively)
        self.du_max = 0.0                   # Max du of most recent step (used for adaptive time stepping)

        # Species r50 drift metric
        s = config.s
        self.r50evo = np.zeros((s, 2))
        frac = np.array([0.5])
        for k in range(s):
            self.r50evo[k,0] = mass_fraction_radii(self.r[k], self.m[k], frac)[0]

        self.rho0       = sum_intensive_loglog_single(np.min(self.rmid[:,0]), self.rmid, self.rho)
        self.r50_spread = calc_r50_spread(self.r, self.m, self.r50evo)

        # For diagnostics
        self.n_iter_du = 0

        self.dt_cum     = 0.0
        self.du_max_cum = 0.0

        if config.io.chatter:
            print("State initialized.")

    def run(self, steps=None, stoptime=None, rho0=None):
        """
        Advance the state until any requested or configured stop condition is met.

        Parameters
        ----------
        steps : int, optional
            Number of additional integration steps.
        stoptime : float, optional
            Additional duration in units of char.t0, measured from this call's
            starting time (not an absolute end time).
        rho0 : float, optional
            Total innermost density threshold in units of char.rho_s.

        Notes
        -----
        Configured sim.t_halt remains an absolute time limit; sim.rho0_halt is
        also checked after the first 1000 total steps. Conditions are checked
        at step boundaries, so time/density thresholds can be overshot.
        Writes initial/final snapshots, time-history records, and log entries,
        and updates the State in place. Returns None.
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
        if rho0 is not None:
            kwargs['rho0'] = rho0

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
        Plot this model's saved time-history data.

        Forwards keyword arguments to pygtf2.plot_time_evolution. The default
        quantity is 'rho_c' (core density), with logarithmic y scale. Supported
        quantities are 'rho_c', 'rho0', 'v2_c', 'r_c', 'eta_c', 'm_c',
        'v20_tot', and 'r_enc'. Uses time_evolution.txt, not unsaved state arrays.
        See the top-level function for labels, saving, and display options.
        Returns None.
        """
        from pygtf2.plot.time_evolution import plot_time_evolution

        plot_time_evolution(self, **kwargs)

    def plot_snapshots(self, **kwargs):
        """
        Plot saved radial profiles for this model.

        Forwards keyword arguments to pygtf2.plot_snapshots, but defaults
        snapshots to -1 (latest saved snapshot). This does not plot unsaved
        state arrays. Profiles are 'rho', 'm', 'v2', and 'eta', with 'rho' as
        the default. See the top-level function for axis and display options.
        Returns None.
        """
        from pygtf2.plot.snapshot import plot_snapshots

        snapshots = kwargs.pop('snapshots', -1)
        plot_snapshots(self, snapshots=snapshots, **kwargs)
        
    def make_movie(self, **kwargs):
        """Animate this simulation with fixed axes and optional insets.

        Accepts the keyword arguments of pygtf2.make_movie, including
        parallel=False for serial rendering and insets=False for no insets.
        Profiles are 'rho', 'm', 'v2', and 'eta'; the default is ['rho', 'v2'].
        Output defaults to movie.mp4 in the model directory. Requires ffmpeg.
        See help(pygtf2.make_movie) for all options.
        """
        from pygtf2.plot.snapshot import make_movie

        make_movie(self, **kwargs)

    def __repr__(self):
        # Copy the __dict__ and omit the 'config' key
        filtered = {k: v for k, v in self.__dict__.items() if k != "config"}
        return f"{self.__class__.__name__}(\n{pprint.pformat(filtered, indent=2)}\n)"