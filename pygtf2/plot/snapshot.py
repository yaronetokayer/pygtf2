import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import subprocess
from tqdm import tqdm
import shutil
from pygtf2.io.read import extract_snapshot_data, extract_snapshot_indices, extract_time_evolution_data

def plot_profile(ax, profile, data_list, axislims=None,
                 legend=True, no_spec=False, grid=False, for_movie=False):
    """
    Plot specified profile on the passed axis object

    Arguments
    ---------
    ax : Axis
        Axis object on which to plot
    profile : str
        Profile to plot.  Options are 'rho', 'm', 'v2', 'p', 'trelax', 'eta'
    data_list : dict
        Dictionary returned by extract_snapshot_data()
    axislims : list of tuples or None
        [(xmin, xmax), (ymin, ymax)]
    legend : bool, optional
        If True, include a legend in the plot
    no_spec : bool, optional
        If True, do not include species legend
    grid : bool, optional
        If True, shows grid on axes
    for_movie : bool, should not be set by user
        If True, then plot_snapshots() is being called by make_movie()
        This controls the colormap of the plots
    """
    # Set colormap
    if for_movie:
        from matplotlib.colors import ListedColormap
        if len(data_list) == 1:
            cmap = ListedColormap(['black'])
        else:
            cmap = ListedColormap(['gray', 'black'])
    else:
        cmap = plt.get_cmap('tab20')

    # Pick linestyles for species; stable mapping by species name
    species_names = sorted(data_list[0]['species'].keys()) if data_list and 'species' in data_list[0] else []
    # base_styles = ['dashed', 'dashdot', 'dotted', (0, (1, 1)), (0, (3, 1, 1, 1))]
    base_styles = [
        (0, (5, 2)),              # long dash
        (0, (1, 1)),              # fine dotted
        (0, (3, 1, 1, 1)),        # dash-dot pattern
        (0, (7, 3, 1, 3)),        # long dash, small dot, medium gap
        (0, (5, 1)),              # medium dash, tight gaps
        (0, (3, 5, 1, 5, 1, 5)),  # mixed dash/dot combo
    ]
    style_map = {name: base_styles[i % len(base_styles)] for i, name in enumerate(species_names)}

    # Totals use the global x keys
    xkey_tot = 'log_r' if profile == 'm' else 'log_rmid'

    if axislims is None:
        # Prepare global axis limits (log scale → ignore nonpositive)
        posmin = np.inf
        posmax = 0.0
        xmin = np.inf
        xmax = -np.inf

        # Pass 1: compute limits across all plotted series
        for data in data_list:
            x_tot = data[xkey_tot]
            xmin = min(xmin, 10.0**np.min(x_tot))
            xmax = max(xmax, 10.0**np.max(x_tot))
            
            if profile in {'rho', 'm', 'p'}:
                y_tot = data[profile + '_tot'] if profile != 'm' else data['m_tot']
                y_candidates = [y_tot]
            elif profile in {'eta'}:
                y_tot = data[profile]
                y_candidates = [y_tot]
            else:  # 'trelax' and 'v2' have no total
                y_candidates = []

            if profile not in {'eta'}: # 'eta' has no per-species
                for sp in species_names:
                    y_sp = data['species'][sp][profile]
                    y_candidates.append(y_sp)

            for y in y_candidates:
                y_pos = y[y > 0]
                if y_pos.size:
                    posmin = min(posmin, float(np.min(y_pos)))
                    posmax = max(posmax, float(np.max(y_pos)))
    else:
        xmin    = axislims[0][0]
        xmax    = axislims[0][1]
        posmin  = axislims[1][0]
        posmax  = axislims[1][1]

    # Safeguard limits
    if not np.isfinite(posmin) or posmin <= 0:
        posmin = 1e-99
    if posmax <= 0:
        posmax = 1.0
    if profile == 'eta' and posmax < 0.6:  # To put horizontal line at 0.5
        posmax = 0.6

    # Pass 2: plot
    for ind, data in enumerate(data_list):
        color = cmap(ind % 10)
        time_lbl = f"t={data['time']:.2e}"

        # totals (solid), if applicable
        x_tot = data[xkey_tot]
        X_tot = 10.0**x_tot
        if profile in {'rho', 'm', 'p'}:
            y_tot = data[profile + '_tot'] if profile != 'm' else data['m_tot']
        elif profile in {'eta'}:
            y_tot = data[profile]
        if profile in {'rho', 'm', 'p', 'eta'}:
            ax.plot(X_tot, y_tot, lw=2.2, color=color, ls='solid', label=time_lbl)
        
        # species (own linestyle), no extra legend spam
        if profile not in {'eta'}:
            sp_xkey = 'lgr' if profile == 'm' else 'lgrm'
            for ind, sp in enumerate(species_names):
                label = '_nolegend_'
                if profile in {'trelax', 'v2'} and ind == 1:
                    label = time_lbl
                x_sp = data['species'][sp][sp_xkey]   # per-species log grid
                X_sp = 10.0**x_sp
                y_sp = data['species'][sp][profile]
                ax.plot(X_sp, y_sp, lw=1.8, color=color, ls=style_map[sp], label=label)

    # Cosmetics
    ax.set_xscale('log')
    if profile not in {'eta'}:
        ax.set_yscale('log')
    if profile == 'eta':
        ax.axhline(0.5, color='black', ls='--', lw=2)
        ax.text(x=xmax, y=0.501, s='equipartition', ha='right', va='bottom')
    ax.set_xlim([xmin * 0.8, xmax * 1.2])
    if profile not in {'eta'}:
        ax.set_ylim([posmin * 0.5, posmax * 10.0])
    else:   # Linear scale
        ax.set_ylim([posmin * 0.9, posmax * 1.1])
    ax.set_xlabel(r'Radius [$r_\mathrm{s,0}$]', fontsize=14)
    ax.set_ylabel(profile, fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    if legend:
        # ax.legend() # OLD
        # 1) Time legend (colors): use the labels already attached to plotted lines
        time_legend = ax.legend(loc='lower left', frameon=True)

        # 2) Species legend (linestyles in black), including 'total' as solid
        if not no_spec:
            species_handles = [Line2D([0], [0], lw=2.2, color='black', ls='solid', label='total')]
            species_handles += [
                Line2D([0], [0], lw=1.8, color='black', ls=style_map[sp], label=sp)
                for sp in species_names
            ]

            species_legend = ax.legend(handles=species_handles, loc='lower center', frameon=True, ncol=1)

        # Keep both legends
        ax.add_artist(time_legend)
        if not no_spec:
            ax.add_artist(species_legend)
    if grid:
        ax.grid(True, which="both", ls="--", alpha=0.4)

def plot_snapshots(model, snapshots=[0], profiles='rho', base_dir=None, filepath=None, show=False, grid=False, for_movie=False):
    """
    Plot up to three profiles at specified points in time for one simulation

    Arguments
    ---------
    model : State object, Config object, or model_no
        Each model can be a State, Config, or integer model number.
    snapshots : int or list of int
        Snapshot indices to plot
    profiles : str or list of str, optional
        Profiles to plot.  Options are 'rho', 'm', 'v2', 'p', 'trelax', 'eta'
    base_dir : str, optional
        Required if any model is passed as an integer.  The directory in which all ModelXXX subdirectories reside.
    filepath : str, optional
        If provided, save the plot to this file.
    show : bool, optional
        If True, show the plot even if saving.  Default is False.
    grid : bool, optional
        If True, shows grid on axes
    for_movie : bool, should not be set by user
        If True, then being called by make_movie()
        This controls the colormap of the plots
    """

    if type(snapshots) != list:
        snapshots = [snapshots]

    def _resolve_dir(model, ind):
        if hasattr(model, 'config'): # Passed state object
            return os.path.join(model.config.io.base_dir, model.config.io.model_dir, f"profile_{ind}.dat")
        elif hasattr(model, 'io'): # Passed config object
            return os.path.join(model.io.base_dir, model.io.model_dir, f"profile_{ind}.dat")
        elif isinstance(model, int): # Passed model number
            if base_dir is None:
                raise ValueError("'base_dir' (base directory) must be specified if using model numbers.")
            model_dir = f"Model{model:03d}"
            return os.path.join(base_dir, model_dir, f"profile_{ind}.dat")
        else:
            raise TypeError(f"Unrecognized model type: {type(model)}. Must be a State object, Config object, or integer.")

    # Change any '-1' entries to the last snapshot index
    for ind, val in enumerate(snapshots):
        if val == -1:
            snapshot_indices_data = extract_snapshot_indices(os.path.dirname(_resolve_dir(model, 0)))
            snapshots[ind] = snapshot_indices_data['snapshot_index'][-1]

    profiles_list = profiles if isinstance(profiles, list) else [profiles]
    n = len(profiles_list) # Number of panels

    data_list = [extract_snapshot_data(_resolve_dir(model,ind)) for ind in snapshots]

    fig, axs = plt.subplots(1, n, figsize=(6*n, 5))

    if n == 1:
        no_spec = profiles_list[0] == 'eta'
        plot_profile(axs, profiles_list[0], data_list, legend=True, no_spec=no_spec, grid=grid, for_movie=for_movie)
    else:
        for ind, ax in enumerate(axs):
            # legend = False if ind < len(axs) - 1 else True
            legend = False if ind > 0 else True
            plot_profile(ax, profiles_list[ind], data_list, legend=legend, grid=grid, for_movie=for_movie)

    if filepath:
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)
    else:
        plt.show()

def make_movie(model, filepath=None, base_dir=None, profiles='rho', grid=False, fps=20):
    """
    Animate up to three profiles for one simulation

    Arguments
    ---------
    model : State object, Config object, or model_no
        Each model can be a State, Config, or integer model number.
    filepath : str, optional
        Save the plot to this file.  Defaults to '/base_dir/ModelXXX/movie_{profiles}.mp4'
    base_dir : str, optional
        Required if any model is passed as an integer.  The directory in which all ModelXXX subdirectories reside.
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

    n = 1 if type(profiles) != list else len(profiles) # number of panels

    # Get the model directory
    if hasattr(model, 'config'):        # Passed state object
        model_dir = os.path.join(model.config.io.base_dir, model.config.io.model_dir)
    elif hasattr(model, 'io'):          # Passed config object
        model_dir = os.path.join(model.io.base_dir, model.io.model_dir)
    elif isinstance(model, int):        # Passed model number
        if base_dir is None:
            raise ValueError("'base_dir' (base directory) must be specified if using model numbers.")
        model_dir = f"Model{model:03d}"
        model_dir = os.path.join(base_dir, model_dir)
    else:
        raise TypeError(f"Unrecognized model type: {type(model)}. Must be a State object, Config object, or integer.")
    
    # Load snapshot indices
    snapshot_indices_data = extract_snapshot_indices(model_dir)
    indices = snapshot_indices_data['snapshot_index']

    # Create a temporary directory for storing images
    temp_dir = os.path.join(model_dir, "temp_images")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)         # Delete the directory and all its contents
    os.makedirs(temp_dir)

    image_paths = []                    # List to store paths of generated images

    print(f"Generating {len(indices)} frames...")
    for ind in tqdm(indices, desc="Frames", unit="frame"):
        snapshot_path = os.path.join(model_dir, f"profile_{ind}.dat")
        if not os.path.isfile(snapshot_path):
            continue                    # Skip if the snapshot file does not exist

        # Define the output image path for the current frame
        image_path = os.path.join(temp_dir, f"frame_{ind:04d}.png")

        # Plot the profile, including the initial profile for comparison
        if ind == 0:
            plot_snapshots(model, profiles=profiles, base_dir=base_dir, filepath=image_path, grid=grid, for_movie=True)
        else:
            plot_snapshots(model, snapshots=[0,ind], profiles=profiles, base_dir=base_dir, filepath=image_path, grid=grid, for_movie=True)

        image_paths.append(image_path)  # Add the image path to the list

    print("Compiling into a movie using ffmpeg...")
    # Define the output movie path
    if isinstance(profiles, (list, tuple)):
        profiles_str = "_".join(map(str, profiles))
    else:
        profiles_str = str(profiles)

    output_movie_path = (
        filepath if filepath is not None 
        else os.path.join(model_dir, f"movie_{profiles_str}.mp4")
    )

    # Construct the ffmpeg command to create the movie
    movie_command = [
        "ffmpeg",
        "-y",                                           # Overwrite output file if it exists
        "-framerate", str(fps),                         # Set frames per second
        "-i", os.path.join(temp_dir, "frame_%04d.png"), # Input image sequence
        "-c:v", "libx264",                              # Use H.264 codec
        "-pix_fmt", "yuv420p",                          # Set pixel format for compatibility
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",     # Ensure even dimensions
        output_movie_path
    ]

    # Run the ffmpeg command
    subprocess.run(movie_command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True)

    print("Deleting frames...")
    # Clean up temporary images
    shutil.rmtree(temp_dir, ignore_errors=True)

    # Print the location of the saved movie
    print(f"Movie saved to {output_movie_path}")

def make_movie_deluxe(model, filepath=None, base_dir=None, grid=False, etaplot=False, fps=20):
    """
    Animate rho and v2 profiles for a simulation with inset of time evolution.
    By default, includes profiles 'rho' and 'v2'.  etaplot=True adds a third panel for eta.
    Scale stays constant throughout.

    Arguments
    ---------
    model : State object, Config object, or model_no
        Each model can be a State, Config, or integer model number.
    filepath : str, optional
        Save the plot to this file.  Defaults to '/base_dir/ModelXXX/movie_{profiles}.mp4'
    base_dir : str, optional
        Required if any model is passed as an integer.  The directory in which all ModelXXX subdirectories reside.
    grid : bool, optional
        If True, shows grid on axes
    etaplot : bool, optional
        If True, add panel for eta
    fps : int, optional
        Frames per second for the output movie. Default is 20

    Returns
    -------
    None
        Saves the movie as an MP4 file in the model directory.
    """
    profiles=['rho','v2']
    if etaplot:
        profiles.append('eta')

    # Number of panels
    n = 1 if type(profiles) != list else len(profiles) 

    # Get the model directory
    if hasattr(model, 'config'):        # Passed state object
        model_dir = os.path.join(model.config.io.base_dir, model.config.io.model_dir)
    elif hasattr(model, 'io'):          # Passed config object
        model_dir = os.path.join(model.io.base_dir, model.io.model_dir)
    elif isinstance(model, int):        # Passed model number
        if base_dir is None:
            raise ValueError("'base_dir' (base directory) must be specified if using model numbers.")
        model_dir = f"Model{model:03d}"
        model_dir = os.path.join(base_dir, model_dir)
    else:
        raise TypeError(f"Unrecognized model type: {type(model)}. Must be a State object, Config object, or integer.")
    
    # Load rhoc time evolution data
    print(f"Getting time evolution data...")
    time_evolution_path = os.path.join(model_dir, f"time_evolution.txt")
    time_data = extract_time_evolution_data(time_evolution_path)
    tevo_t = time_data['time']; tevo_rho = time_data['rho_c_tot']
    if etaplot:
        tevo_eta = time_data['eta_c']

    # Load snapshot indices
    snapshot_indices_data   = extract_snapshot_indices(model_dir)
    indices                 = snapshot_indices_data['snapshot_index']
    index_t                 = snapshot_indices_data['t_t0']

    # Get axis limits
    print(f"Getting axis limits...")
    
    xmin    = np.inf
    xmax    = 0.0
    rhomin  = np.inf
    rhomax  = 0.0
    v2min   = np.inf
    v2max   = 0.0
    if etaplot:
        etamin   = np.inf
        etamax   = 0.0  

    for ind in indices:
        snapshot_path = os.path.join(model_dir, f"profile_{ind}.dat")
        if not os.path.isfile(snapshot_path):
            continue                            # Skip if the snapshot file does not exist
        data_list = extract_snapshot_data(snapshot_path)
        r   = 10**data_list['log_rmid']
        rho = data_list['rho_tot']
        if np.min(r) < xmin:
            xmin = np.min(r)
        if np.max(r) > xmax:
            xmax = np.max(r)
        if np.min(rho) < rhomin:
            rhomin = np.min(rho)
        if np.max(rho) > rhomax:
            rhomax = np.max(rho)
        if etaplot:
            eta  = data_list['eta']
            if np.min(eta) < etamin:
                etamin = np.min(eta)
            if np.max(eta) > etamax:
                etamax = np.max(eta)
        for spec_data in data_list['species'].values():
            rho = spec_data['rho']
            if np.min(rho) < rhomin:
                rhomin = np.min(rho)
            if np.max(rho) > rhomax:
                rhomax = np.max(rho)
            v2  = spec_data['v2']
            if np.min(v2) < v2min:
                v2min = np.min(v2)
            if np.max(v2) > v2max:
                v2max = np.max(v2)


    # Create a temporary directory for storing images
    temp_dir = os.path.join(model_dir, "temp_images")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)             # Delete the directory and all its contents
    os.makedirs(temp_dir)

    image_paths = []                        # List to store paths of generated images

    print(f"Generating {len(indices)} frames...")
    for ind in tqdm(indices, desc="Frames", unit="frame"):
        snapshot_path = os.path.join(model_dir, f"profile_{ind}.dat")
        if not os.path.isfile(snapshot_path):
            continue                        # Skip if the snapshot file does not exist

        # Define the output image path for the current frame
        image_path = os.path.join(temp_dir, f"frame_{ind:04d}.png")

        # Extract data for current frame and initial frame
        initial_snapshot_path   = os.path.join(model_dir, f"profile_0.dat")
        data_list               = [
            extract_snapshot_data(initial_snapshot_path), 
            extract_snapshot_data(snapshot_path)
            ]

        # Plot profile and initial profile
        fig, axs = plt.subplots(1, n, figsize=(6*n, 5))

        for i, ax in enumerate(axs):
            legend = False if i > 0 else True
            axislims = [(xmin, xmax)]
            if profiles[i] == 'rho':
                axislims.append((rhomin, rhomax))
            elif profiles[i] == 'v2':
                axislims.append((v2min, v2max))
            elif profiles[i] == 'eta':
                axislims.append((etamin, etamax))
            plot_profile(ax, profiles[i], data_list, axislims=axislims, legend=legend, grid=grid, for_movie=True)
        
        # Plot inset of rho_c over time
        axinrho = axs[1].inset_axes([0.55, 0.65, 0.45, 0.35])
        axinrho.axvline(index_t[ind], color='grey')
        axinrho.plot(tevo_t, tevo_rho, color='black')
        axinrho.scatter(index_t[ind], np.interp(index_t[ind], tevo_t, tevo_rho),
                     color='red', s=50)
        axinrho.set_ylabel(r'$\rho_\mathrm{c}$', fontsize=12)
        axinrho.set_xlabel('$t$', fontsize=12)
        axinrho.set_yscale('log')
        axinrho.tick_params(
            axis='both',
            which='both',
            labelbottom=False,
            labelleft=False,
            labeltop=False,
            labelright=False,
            top=True,
            bottom=True,
            left=True,
            right=True,
            direction='in'
        )

        # If eta, plot inset of eta_c over time
        if etaplot:
            axineta = axs[2].inset_axes([0.55, 0.65, 0.45, 0.35])
            axineta.axvline(index_t[ind], color='grey')
            axineta.plot(tevo_t, tevo_eta, color='black')
            axineta.scatter(index_t[ind], np.interp(index_t[ind], tevo_t, tevo_eta),
                            color='red', s=50)
            axineta.set_ylabel(r'$\eta_\mathrm{c}$', fontsize=12)
            axineta.set_xlabel('$t$', fontsize=12)
            axineta.tick_params(
                axis='both',
                which='both',
                labelbottom=False,
                labelleft=False,
                labeltop=False,
                labelright=False,
                top=True,
                bottom=True,
                left=True,
                right=True,
                direction='in'
            )
        
        fig.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        image_paths.append(image_path)  # Add the image path to the list

    print("Compiling into a movie using ffmpeg...")

    if filepath is not None:
        output_movie_path = filepath
    elif etaplot:
        output_movie_path = os.path.join(model_dir, f"movie_deluxe_eta.mp4")
    else:
        output_movie_path = os.path.join(model_dir, f"movie_deluxe.mp4")

    # Construct the ffmpeg command to create the movie
    movie_command = [
        "ffmpeg",
        "-y",                                           # Overwrite output file if it exists
        "-framerate", str(fps),                         # Set frames per second
        "-i", os.path.join(temp_dir, "frame_%04d.png"), # Input image sequence
        "-c:v", "libx264",                              # Use H.264 codec
        "-pix_fmt", "yuv420p",                          # Set pixel format for compatibility
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",     # Ensure even dimensions
        output_movie_path
    ]

    # Run the ffmpeg command
    subprocess.run(movie_command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True)

    print("Deleting frames...")
    # Clean up temporary images
    shutil.rmtree(temp_dir, ignore_errors=True)

    # Print the location of the saved movie
    print(f"Movie saved to {output_movie_path}")

# def make_movie_deluxe(model, filepath=None, base_dir=None, grid=False, fps=20):
#     """
#     Animate rho and v2 profiles for a simulation with inset of time evolution.
#     For now, only designed to work for profiles=['rho','v2'].
#     Scale stays constant throughout.

#     Arguments
#     ---------
#     model : State object, Config object, or model_no
#         Each model can be a State, Config, or integer model number.
#     filepath : str, optional
#         Save the plot to this file.  Defaults to '/base_dir/ModelXXX/movie_{profiles}.mp4'
#     base_dir : str, optional
#         Required if any model is passed as an integer.  The directory in which all ModelXXX subdirectories reside.
#     grid : bool, optional
#         If True, shows grid on axes
#     fps : int, optional
#         Frames per second for the output movie. Default is 20

#     Returns
#     -------
#     None
#         Saves the movie as an MP4 file in the model directory.
#     """
#     profiles=['rho','v2'] # Perhaps will change in future versions

#     # Number of panels - this will always be two for current version.
#     n = 1 if type(profiles) != list else len(profiles) 

#     # Get the model directory
#     if hasattr(model, 'config'):        # Passed state object
#         model_dir = os.path.join(model.config.io.base_dir, model.config.io.model_dir)
#     elif hasattr(model, 'io'):          # Passed config object
#         model_dir = os.path.join(model.io.base_dir, model.io.model_dir)
#     elif isinstance(model, int):        # Passed model number
#         if base_dir is None:
#             raise ValueError("'base_dir' (base directory) must be specified if using model numbers.")
#         model_dir = f"Model{model:03d}"
#         model_dir = os.path.join(base_dir, model_dir)
#     else:
#         raise TypeError(f"Unrecognized model type: {type(model)}. Must be a State object, Config object, or integer.")
    
#     # Load rhoc time evolution data
#     print(f"Getting time evolution data...")
#     time_evolution_path = os.path.join(model_dir, f"time_evolution.txt")
#     time_data = extract_time_evolution_data(time_evolution_path)
#     tevo_t = time_data['time']; tevo_rho = time_data['rho_c_tot']

#     # Load snapshot indices
#     snapshot_indices_data   = extract_snapshot_indices(model_dir)
#     indices                 = snapshot_indices_data['snapshot_index']
#     index_t                 = snapshot_indices_data['t_t0']

#     # Get axis limits
#     print(f"Getting axis limits...")
    
#     xmin    = np.inf
#     xmax    = 0.0
#     rhomin  = np.inf
#     rhomax  = 0.0
#     v2min   = np.inf
#     v2max   = 0.0

#     for ind in indices:
#         snapshot_path = os.path.join(model_dir, f"profile_{ind}.dat")
#         if not os.path.isfile(snapshot_path):
#             continue                            # Skip if the snapshot file does not exist
#         data_list = extract_snapshot_data(snapshot_path)
#         r   = 10**data_list['log_rmid']
#         rho = data_list['rho_tot']
#         if np.min(r) < xmin:
#             xmin = np.min(r)
#         if np.max(r) > xmax:
#             xmax = np.max(r)
#         if np.min(rho) < rhomin:
#             rhomin = np.min(rho)
#         if np.max(rho) > rhomax:
#             rhomax = np.max(rho)
#         for spec_data in data_list['species'].values():
#             rho = spec_data['rho']
#             if np.min(rho) < rhomin:
#                 rhomin = np.min(rho)
#             if np.max(rho) > rhomax:
#                 rhomax = np.max(rho)
#             v2  = spec_data['v2']
#             if np.min(v2) < v2min:
#                 v2min = np.min(v2)
#             if np.max(v2) > v2max:
#                 v2max = np.max(v2)

#     # Create a temporary directory for storing images
#     temp_dir = os.path.join(model_dir, "temp_images")
#     if os.path.exists(temp_dir):
#         shutil.rmtree(temp_dir)             # Delete the directory and all its contents
#     os.makedirs(temp_dir)

#     image_paths = []                        # List to store paths of generated images

#     print(f"Generating {len(indices)} frames...")
#     for ind in tqdm(indices, desc="Frames", unit="frame"):
#         snapshot_path = os.path.join(model_dir, f"profile_{ind}.dat")
#         if not os.path.isfile(snapshot_path):
#             continue                        # Skip if the snapshot file does not exist

#         # Define the output image path for the current frame
#         image_path = os.path.join(temp_dir, f"frame_{ind:04d}.png")

#         # Extract data for current frame and initial frame
#         initial_snapshot_path   = os.path.join(model_dir, f"profile_0.dat")
#         data_list               = [
#             extract_snapshot_data(initial_snapshot_path), 
#             extract_snapshot_data(snapshot_path)
#             ]

#         # Plot profile and initial profile
#         fig, axs = plt.subplots(1, n, figsize=(6*n, 5))

#         for i, ax in enumerate(axs):
#             legend = False if i > 0 else True
#             axislims = [(xmin, xmax)]
#             if profiles[i] == 'rho':
#                 axislims.append((rhomin, rhomax))
#             elif profiles[i] == 'v2':
#                 axislims.append((v2min, v2max))
#             plot_profile(ax, profiles[i], data_list, axislims=axislims, legend=legend, grid=grid, for_movie=True)
        
#         # Plot inset of rho_c over time
#         axin = axs[1].inset_axes([0.55, 0.65, 0.45, 0.35])
#         axin.axvline(index_t[ind], color='grey')
#         axin.plot(tevo_t, tevo_rho, color='black')
#         axin.scatter(index_t[ind], np.interp(index_t[ind], tevo_t, tevo_rho),
#                      color='red', s=50)
#         axin.set_ylabel(r'$\rho_\mathrm{c}$', fontsize=12)
#         axin.set_xlabel('$t$', fontsize=12)
#         axin.set_yscale('log')
#         axin.tick_params(
#             axis='both',
#             which='both',
#             labelbottom=False,
#             labelleft=False,
#             labeltop=False,
#             labelright=False,
#             top=True,
#             bottom=True,
#             left=True,
#             right=True,
#             direction='in'
#         )
        
#         fig.savefig(image_path, dpi=300, bbox_inches='tight')
#         plt.close(fig)
#         image_paths.append(image_path)  # Add the image path to the list

#     print("Compiling into a movie using ffmpeg...")

#     output_movie_path = (
#         filepath if filepath is not None 
#         else os.path.join(model_dir, f"movie_deluxe.mp4")
#     )

#     # Construct the ffmpeg command to create the movie
#     movie_command = [
#         "ffmpeg",
#         "-y",                                           # Overwrite output file if it exists
#         "-framerate", str(fps),                         # Set frames per second
#         "-i", os.path.join(temp_dir, "frame_%04d.png"), # Input image sequence
#         "-c:v", "libx264",                              # Use H.264 codec
#         "-pix_fmt", "yuv420p",                          # Set pixel format for compatibility
#         "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",     # Ensure even dimensions
#         output_movie_path
#     ]

#     # Run the ffmpeg command
#     subprocess.run(movie_command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True)

#     print("Deleting frames...")
#     # Clean up temporary images
#     shutil.rmtree(temp_dir, ignore_errors=True)

#     # Print the location of the saved movie
#     print(f"Movie saved to {output_movie_path}")