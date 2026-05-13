import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.transforms as transforms
import subprocess
from tqdm import tqdm
import shutil
from pygtf2.io.read import extract_snapshot_data, extract_snapshot_indices, extract_time_evolution_data

def get_profile_axis_limits(profile, data_list, xaxis='r'):
    """
    Compute global axis limits for multi-species profile plots.
    Includes total profiles where plotted, and species profiles where plotted.
    """

    xlim_lower = np.inf
    xlim_upper = -np.inf
    ylim_lower = np.inf
    ylim_upper = -np.inf

    species_names = (
        sorted(data_list[0]['species'].keys())
        if data_list and 'species' in data_list[0]
        else []
    )

    for data in data_list:

        # ---------- Total profiles ----------
        if profile in {'rho', 'm', 'p', 'eta'}:
            if xaxis == 'r':
                xkey_tot = 'log_r' if profile == 'm' else 'log_rmid'
                x_tot = 10.0**data[xkey_tot]
            elif xaxis == 'm':
                x_tot = data['m_tot']

            if profile in {'rho', 'p'}:
                y_tot = data[profile + '_tot']
            elif profile == 'm':
                y_tot = data['m_tot']
            elif profile == 'eta':
                y_tot = data['eta']

            xlim_lower = min(xlim_lower, np.min(x_tot) * 0.8)
            xlim_upper = max(xlim_upper, np.max(x_tot) * 1.2)

            positive_y = y_tot[y_tot > 0]
            if positive_y.size:
                ylim_lower = min(ylim_lower, np.min(positive_y) * 0.5)
            ylim_upper = max(ylim_upper, np.max(y_tot) * 10.0)

        # ---------- Species profiles ----------
        if profile != 'eta':
            for sp in species_names:
                sp_data = data['species'][sp]

                if xaxis == 'r':
                    sp_xkey = 'lgr' if profile == 'm' else 'lgrm'
                    x_sp = 10.0**sp_data[sp_xkey]
                elif xaxis == 'm':
                    x_sp = data['m_tot']

                y_sp = sp_data[profile]

                xlim_lower = min(xlim_lower, np.min(x_sp) * 0.8)
                xlim_upper = max(xlim_upper, np.max(x_sp) * 1.2)

                positive_y = y_sp[y_sp > 0]
                if positive_y.size:
                    ylim_lower = min(ylim_lower, np.min(positive_y) * 0.5)
                ylim_upper = max(ylim_upper, np.max(y_sp) * 10.0)

    # Safeguards
    if not np.isfinite(ylim_lower) or ylim_lower <= 0:
        ylim_lower = 1e-99
    if not np.isfinite(ylim_upper) or ylim_upper <= 0:
        ylim_upper = 1.0

    if profile == 'eta':
        if ylim_upper < 0.6:
            ylim_upper = 0.6
        ylim_lower *= 0.9
        ylim_upper *= 1.1

    return (xlim_lower, xlim_upper), (ylim_lower, ylim_upper)

def plot_profile(ax, profile, data_list, xaxis='r', axislims=None,
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
    xaxis : str, optional
        X-axis to plot.  Default is 'r'.  Other option is 'm'.
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
    if xaxis == 'r':
        xkey_tot = 'log_r' if profile == 'm' else 'log_rmid'
    elif xaxis == 'm':
        xkey_tot = 'm_tot'
    # xkey_tot = 'log_r' if profile == 'm' else 'log_rmid' # OLD

    if axislims is None:
        xlim, ylim = get_profile_axis_limits(profile, data_list, xaxis=xaxis)
    else:
        xlim, ylim = axislims

    # Plot
    for ind, data in enumerate(data_list):
        color = cmap(ind % 10)
        time_lbl = f"t={data['time']:.2e}"

        # totals (solid), if applicable
        if xaxis == 'r':
            X_tot = 10.0**data[xkey_tot]
        elif xaxis == 'm':
            X_tot = data[xkey_tot]
        if profile in {'rho', 'm', 'p'}:
            y_tot = data[profile + '_tot'] if profile != 'm' else data['m_tot']
        elif profile in {'eta'}:
            y_tot = data[profile]
        if profile in {'rho', 'm', 'p', 'eta'}:
            ax.plot(X_tot, y_tot, lw=2.2, color=color, ls='solid', label=time_lbl)
        
        # species (own linestyle), no extra legend spam
        if profile not in {'eta'}:
            for isp, sp in enumerate(species_names):
                label = '_nolegend_'
                if profile in {'trelax', 'v2'} and isp == 1:
                    label = time_lbl

                if xaxis == 'r':
                    sp_xkey = 'lgr' if profile == 'm' else 'lgrm'
                    X_sp = 10.0**data['species'][sp][sp_xkey]
                elif xaxis == 'm':
                    X_sp = data['m_tot']

                y_sp = data['species'][sp][profile]
                ax.plot(X_sp, y_sp, lw=1.8, color=color, ls=style_map[sp], label=label)

    # Cosmetics
    ax.set_xscale('log')
    if profile not in {'eta'}:
        ax.set_yscale('log')
    if profile == 'eta':
        ax.axhline(0.5, color='black', ls='--', lw=2)
        ax.text(x=xlim[1], y=0.501, s='equipartition', ha='right', va='bottom')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if xaxis == 'r':
        ax.set_xlabel(r'Radius [$r_\mathrm{s,0}$]', fontsize=14)
    elif xaxis == 'm':
        ax.set_xlabel(r'$M_\mathrm{enc}$ [$M_\mathrm{s}$]', fontsize=14)
    ax.set_ylabel(profile, fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    if legend:
        # ax.legend() # OLD
        # 1) Time legend (colors): use the labels already attached to plotted lines
        time_legend = ax.legend(loc='lower center', frameon=True)

        # 2) Species legend (linestyles in black), including 'total' as solid
        if not no_spec and profile != 'eta':
            species_handles = [Line2D([0], [0], lw=2.2, color='black', ls='solid', label='total')]
            species_handles += [
                Line2D([0], [0], lw=1.8, color='black', ls=style_map[sp], label=sp)
                for sp in species_names
            ]

            species_legend = ax.legend(handles=species_handles, loc='lower left', frameon=True, ncol=1)

        # Keep both legends
        ax.add_artist(time_legend)
        if not no_spec and profile != 'eta':
            ax.add_artist(species_legend)

    if grid:
        ax.grid(True, which="both", ls="--", alpha=0.4)

def plot_snapshots(model, snapshots=[0], profiles='rho', xaxis=None, base_dir=None, filepath=None, show=False, grid=False, for_movie=False):
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
    xaxis : list of str, optional
        X-axis for profiles to plot.  Default is 'r'.  Other option is 'm'.
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

    if xaxis is None:
        xaxis = ['r'] * len(profiles)
    elif isinstance(xaxis, str):
        xaxis = [xaxis]

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
            snapshots[ind] = snapshot_indices_data['index'][-1]

    profiles_list = profiles if isinstance(profiles, list) else [profiles]
    n = len(profiles_list) # Number of panels

    data_list = [extract_snapshot_data(_resolve_dir(model,ind)) for ind in snapshots]

    fig, axs = plt.subplots(1, n, figsize=(6*n, 5))

    if n == 1:
        no_spec = profiles_list[0] == 'eta'
        plot_profile(axs, profiles_list[0], data_list, xaxis=xaxis[0], legend=True, no_spec=no_spec, grid=grid, for_movie=for_movie)
    else:
        for ind, ax in enumerate(axs):
            # legend = False if ind < len(axs) - 1 else True
            legend = False if ind > 0 else True
            plot_profile(ax, profiles_list[ind], data_list, xaxis=xaxis[ind], legend=legend, grid=grid, for_movie=for_movie)

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

def make_movie_deluxe(model, profiles=None, insets=None, xaxis=None, add_radii=None, filepath=None, base_dir=None, grid=False, fps=20):
    """
    Animate profiles wit constant scale and with inset for time evolution.
    Scale stays constant throughout.

    Arguments
    ---------
    model : State object, Config object, or model_no
        Each model can be a State, Config, or integer model number.
    profiles : list of str, optional
        Profiles to plot.  Options are 'rho', 'm', 'v2', 'trelax', 'eta'.
    insets : list of str or None, optional
        Inset plots to include.  Options are any quantity in time_evolution.txt
    xaxis : list of str, optional
        X-axis for profiles to plot.  Default is 'r'.  Other option is 'm'.
    add_radii : list, optional
        List of radii to add to profiles from time_evolution.txt
        Options: 'r_c', 'r_50[heavy]', etc
    filepath : str, optional
        Save the plot to this file.  Defaults to '/base_dir/ModelXXX/movie_deluxe.mp4'
    base_dir : str, optional
        Required if any model is passed as an integer.  The directory in which all ModelXXX subdirectories reside.
    grid : bool, optional
        If True, shows grid on axes
    fps : int, optional
        Frames per second for the output movie. Default is 20

    Returns
    -------
    None
        Saves the movie as an MP4 file in the model directory.
    """
    # Collect profiles and insets
    if profiles is None:
        profiles = ['rho', 'v2']
    elif isinstance(profiles, str):
        profiles = [profiles]
    if insets is None:
        insets = ['rho_c_tot'] + [None] * (len(profiles) - 1)
    elif isinstance(insets, str) or insets is None:
        insets = [insets]
    if xaxis is None:
        xaxis = ['r'] * len(profiles)
    elif isinstance(xaxis, str):
        xaxis = [xaxis]

    # Validate profiles
    valid_profiles = ['rho', 'm', 'v2', 'trelax', 'eta']
    if any(profile not in valid_profiles for profile in profiles):
        raise ValueError(f"Invalid profile specified. Valid options are: {valid_profiles}")
    
    # Validate radii
    valid_radii = ['r_c', 'r01', 'r05', 'r10', 'r20', 'r50', 'r90']
    if add_radii is not None:
        if isinstance(add_radii, str):
            add_radii = [add_radii]
        if any(radius not in valid_radii for radius in add_radii):
            raise ValueError(f"Invalid radius specified. Valid options are: {valid_radii}")
        
    # Validate xaxis
    valid_xaxis = ['r', 'm']
    if any(x not in valid_xaxis for x in xaxis):
        raise ValueError(f"Invalid x-axis specified. Valid options are: {valid_xaxis}")

    # Number of panels
    n = len(profiles) 

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
    tevo_t = time_data['time']

    # Validate insets
    valid_insets = list(time_data.keys())
    if any(inset not in valid_insets for inset in insets if inset is not None):
        raise ValueError(f"Invalid inset specified. Valid options are: {valid_insets}")
    if len(insets) != len(profiles):
        raise ValueError("'insets' must have the same length as 'profiles'.")

    # Load snapshot indices
    snapshot_indices_data   = extract_snapshot_indices(model_dir)
    indices                 = snapshot_indices_data['index']
    index_t                 = snapshot_indices_data['time']

    # Get axis limits
    print(f"Getting axis limits...")
    snapshot_data_list = []

    for ind in indices:
        snapshot_path = os.path.join(model_dir, f"profile_{ind}.dat")

        if not os.path.isfile(snapshot_path):
            continue

        snapshot_data_list.append(extract_snapshot_data(snapshot_path))

    axislims = {}

    for i, profile in enumerate(profiles):
        xlim, ylim = get_profile_axis_limits(profile, snapshot_data_list, xaxis=xaxis[i])
        axislims[profile] = (xlim, ylim)

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
        axs = np.atleast_1d(axs)

        for i, ax in enumerate(axs):
            profile = profiles[i]
            inset   = insets[i]
            xax     = xaxis[i]

            legend = True if i == 0 else False
            plot_profile(ax, profile, data_list, xaxis=xax, axislims=axislims[profile], legend=legend, grid=grid, for_movie=True)

            if add_radii is not None:
                frac_up = 0.15
                for radius in add_radii:
                    if radius in ['r01', 'r05', 'r10', 'r20', 'r50', 'r90']: # One per species
                        for spec in time_data['species']:
                            r = np.interp(index_t[ind], tevo_t, time_data['species'][spec][radius])
                            if xax == 'r':
                                # If r is outside the x-axis limits, skip plotting
                                if r < axislims[profile][0][0] or r > axislims[profile][0][1]:
                                    continue
                                ax.axvline(r, color='red', ls='--', zorder=-10)
                                trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
                                ax.text(r, frac_up, f"{radius}[{spec}]", rotation=90, color='red', fontsize=10, ha='right', va='bottom', zorder=-10, transform=trans)
                            elif xax == 'm':
                                m = np.interp(r, 10**data_list[1]['log_r'], data_list[1]['m_tot'])
                                # If r is outside the x-axis limits, skip plotting
                                if m < axislims[profile][0][0] or m > axislims[profile][0][1]:
                                    continue
                                ax.axvline(m, color='red', ls='--', zorder=-10)
                                trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
                                ax.text(m, frac_up, f"{radius}[{spec}]", rotation=90, color='red', fontsize=10, ha='right', va='bottom', zorder=-10, transform=trans)
                    else:
                        r = np.interp(index_t[ind], tevo_t, time_data[radius])
                        if xax == 'r':
                            # If r is outside the x-axis limits, skip plotting
                            if r < axislims[profile][0][0] or r > axislims[profile][0][1]:
                                continue
                            ax.axvline(r, color='red', ls='--', zorder=-10)
                            trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
                            ax.text(r, frac_up, radius, rotation=90, color='red', fontsize=10, ha='right', va='bottom', zorder=-10, transform=trans)
                        elif xax == 'm':
                            m = np.interp(r, 10**data_list[1]['log_r'], data_list[1]['m'])
                            # If r is outside the x-axis limits, skip plotting
                            if m < axislims[profile][0][0] or m > axislims[profile][0][1]:
                                continue
                            ax.axvline(m, color='red', ls='--', zorder=-10)
                            trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
                            ax.text(m, frac_up, radius, rotation=90, color='red', fontsize=10, ha='right', va='bottom', zorder=-10, transform=trans)

            if inset is not None:
                tevo_y = time_data[inset]
                if profile != 'trelax':
                    axin = ax.inset_axes([0.55, 0.65, 0.45, 0.35])
                else:
                    axin = ax.inset_axes([0.0, 0.65, 0.45, 0.35])
                axin.axvline(index_t[ind], color='grey')
                axin.plot(tevo_t, tevo_y, color='black')
                axin.scatter(index_t[ind], np.interp(index_t[ind], tevo_t, tevo_y),
                            color='red', s=50)
                axin.set_ylabel(inset, fontsize=12)
                axin.set_xlabel('$t$', fontsize=12)
                axin.set_yscale('log')
                axin.tick_params(
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
        
        # # Plot inset of rho_c over time
        # axinrho = axs[1].inset_axes([0.55, 0.65, 0.45, 0.35])
        # axinrho.axvline(index_t[ind], color='grey')
        # axinrho.plot(tevo_t, tevo_rho, color='black')
        # axinrho.scatter(index_t[ind], np.interp(index_t[ind], tevo_t, tevo_rho),
        #              color='red', s=50)
        # axinrho.set_ylabel(r'$\rho_\mathrm{c}$', fontsize=12)
        # axinrho.set_xlabel('$t$', fontsize=12)
        # axinrho.set_yscale('log')
        # axinrho.tick_params(
        #     axis='both',
        #     which='both',
        #     labelbottom=False,
        #     labelleft=False,
        #     labeltop=False,
        #     labelright=False,
        #     top=True,
        #     bottom=True,
        #     left=True,
        #     right=True,
        #     direction='in'
        # )
        
        fig.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        image_paths.append(image_path)  # Add the image path to the list

    print("Compiling into a movie using ffmpeg...")

    if filepath is not None:
        output_movie_path = filepath
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

# def make_movie_deluxe(model, filepath=None, base_dir=None, grid=False, etaplot=False, fps=20):
#     """
#     Animate rho and v2 profiles for a simulation with inset of time evolution.
#     By default, includes profiles 'rho' and 'v2'.  etaplot=True adds a third panel for eta.
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
#     etaplot : bool, optional
#         If True, add panel for eta
#     fps : int, optional
#         Frames per second for the output movie. Default is 20

#     Returns
#     -------
#     None
#         Saves the movie as an MP4 file in the model directory.
#     """
#     profiles=['rho','v2']
#     if etaplot:
#         profiles.append('eta')

#     # Number of panels
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
#     if etaplot:
#         tevo_eta = time_data['eta_c']

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
#     if etaplot:
#         etamin   = np.inf
#         etamax   = 0.0  

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
#         if etaplot:
#             eta  = data_list['eta']
#             if np.min(eta) < etamin:
#                 etamin = np.min(eta)
#             if np.max(eta) > etamax:
#                 etamax = np.max(eta)
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
#             elif profiles[i] == 'eta':
#                 axislims.append((etamin, etamax))
#             plot_profile(ax, profiles[i], data_list, axislims=axislims, legend=legend, grid=grid, for_movie=True)
        
#         # Plot inset of rho_c over time
#         axinrho = axs[1].inset_axes([0.55, 0.65, 0.45, 0.35])
#         axinrho.axvline(index_t[ind], color='grey')
#         axinrho.plot(tevo_t, tevo_rho, color='black')
#         axinrho.scatter(index_t[ind], np.interp(index_t[ind], tevo_t, tevo_rho),
#                      color='red', s=50)
#         axinrho.set_ylabel(r'$\rho_\mathrm{c}$', fontsize=12)
#         axinrho.set_xlabel('$t$', fontsize=12)
#         axinrho.set_yscale('log')
#         axinrho.tick_params(
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

#         # If eta, plot inset of eta_c over time
#         if etaplot:
#             axineta = axs[2].inset_axes([0.55, 0.65, 0.45, 0.35])
#             axineta.axvline(index_t[ind], color='grey')
#             axineta.plot(tevo_t, tevo_eta, color='black')
#             axineta.scatter(index_t[ind], np.interp(index_t[ind], tevo_t, tevo_eta),
#                             color='red', s=50)
#             axineta.set_ylabel(r'$\eta_\mathrm{c}$', fontsize=12)
#             axineta.set_xlabel('$t$', fontsize=12)
#             axineta.tick_params(
#                 axis='both',
#                 which='both',
#                 labelbottom=False,
#                 labelleft=False,
#                 labeltop=False,
#                 labelright=False,
#                 top=True,
#                 bottom=True,
#                 left=True,
#                 right=True,
#                 direction='in'
#             )
        
#         fig.savefig(image_path, dpi=300, bbox_inches='tight')
#         plt.close(fig)
#         image_paths.append(image_path)  # Add the image path to the list

#     print("Compiling into a movie using ffmpeg...")

#     if filepath is not None:
#         output_movie_path = filepath
#     elif etaplot:
#         output_movie_path = os.path.join(model_dir, f"movie_deluxe_eta.mp4")
#     else:
#         output_movie_path = os.path.join(model_dir, f"movie_deluxe.mp4")

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