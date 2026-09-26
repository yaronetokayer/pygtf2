import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.transforms as transforms
import subprocess
from tqdm import tqdm
import shutil
from pygtf2.io.read import extract_snapshot_data, extract_snapshot_indices, extract_time_evolution_data
from concurrent.futures import ProcessPoolExecutor, as_completed

VALID_PROFILES = ('rho', 'm', 'v2', 'eta')
TOTAL_PROFILE_KEYS = {
    'rho': 'rho_tot',
    'm': 'm_tot',
    'v2': 'v2_tot',
}


def _validate_profile(profile):
    if profile not in VALID_PROFILES:
        raise ValueError(
            f"Invalid profile '{profile}'. Valid options are: {list(VALID_PROFILES)}"
        )


def _mass_coordinate_from_radius(data, radius):
    """Return total enclosed mass M_tot(<r) at arbitrary radius/radii."""
    r_edges = 10.0 ** np.asarray(data['log_r'])
    m_edges = np.asarray(data['m_tot'])

    # profile files omit the r=0 edge because log10(0) is undefined.
    # Reconstruct it so interpolation inside the first resolved edge is sensible.
    r_interp = np.concatenate(([0.0], r_edges))
    m_interp = np.concatenate(([0.0], m_edges))

    return np.interp(radius, r_interp, m_interp)


def _profile_x_values(data, profile, xaxis='r', species=None):
    """Return x values appropriate to one total or species profile."""
    _validate_profile(profile)

    if species is None:
        log_r_key = 'log_r' if profile == 'm' else 'log_rmid'
        radius = 10.0 ** np.asarray(data[log_r_key])
    else:
        sp = data['species'][species]
        log_r_key = 'lgr' if profile == 'm' else 'lgrm'
        radius = 10.0 ** np.asarray(sp[log_r_key])

    if xaxis == 'r':
        return radius
    if xaxis == 'm':
        # m_tot itself already lives on the total radial edges.
        if species is None and profile == 'm':
            return np.asarray(data['m_tot'])
        return _mass_coordinate_from_radius(data, radius)

    raise ValueError("xaxis must be either 'r' or 'm'.")


def _profile_y_values(data, profile, species=None):
    """Return y values for one total/global or species profile."""
    _validate_profile(profile)

    if species is None:
        if profile == 'eta':
            return np.asarray(data['eta'])

        key = TOTAL_PROFILE_KEYS[profile]
        if key not in data:
            raise KeyError(
                f"Snapshot is missing required total profile column '{key}'. "
                "Regenerate snapshots with the updated writer."
            )
        return np.asarray(data[key])

    if profile == 'eta':
        raise ValueError("eta is a global multi-species profile and has no per-species curve.")

    return np.asarray(data['species'][species][profile])


def _valid_time_insets(time_data):
    """Top-level 1D time-series fields suitable for deluxe movie insets."""
    if 'time' not in time_data:
        return []

    n = len(np.asarray(time_data['time']))
    valid = []
    for key, value in time_data.items():
        if key in {'species', 'model_id'}:
            continue
        arr = np.asarray(value)
        if arr.ndim == 1 and len(arr) == n and np.issubdtype(arr.dtype, np.number):
            valid.append(key)
    return valid


def get_profile_axis_limits(profile, data_list, xaxis='r'):
    """
    Compute global axis limits for multi-species profile plots.

    rho, m, and v2 include both the total-system and per-species curves.
    eta is a single global profile and is plotted on a linear y-axis.
    """
    _validate_profile(profile)
    if xaxis not in {'r', 'm'}:
        raise ValueError("xaxis must be either 'r' or 'm'.")
    if not data_list:
        raise ValueError("data_list must contain at least one snapshot.")

    x_values = []
    y_values = []

    species_names = sorted(data_list[0].get('species', {}).keys())

    for data in data_list:
        # Global/total curve.
        x_values.append(_profile_x_values(data, profile, xaxis=xaxis))
        y_values.append(_profile_y_values(data, profile))

        # Species curves where applicable.
        if profile != 'eta':
            for sp in species_names:
                x_values.append(
                    _profile_x_values(data, profile, xaxis=xaxis, species=sp)
                )
                y_values.append(_profile_y_values(data, profile, species=sp))

    finite_positive_x = np.concatenate([
        np.asarray(x)[np.isfinite(x) & (np.asarray(x) > 0)]
        for x in x_values
    ])
    if finite_positive_x.size == 0:
        raise ValueError("No positive finite x-values available for logarithmic x-axis.")

    xlim = (
        np.min(finite_positive_x) * 0.8,
        np.max(finite_positive_x) * 1.2,
    )

    if profile == 'eta':
        finite_y = np.concatenate([
            np.asarray(y)[np.isfinite(y)]
            for y in y_values
        ])

        if finite_y.size == 0:
            ylim = (0.0, 0.6)
        else:
            ymin = float(np.min(finite_y))
            ymax = max(float(np.max(finite_y)), 0.6)  # keep equipartition line visible
            span = ymax - ymin
            if span <= 0:
                span = max(abs(ymax), 1.0)
            pad = 0.05 * span
            ylim = (ymin - pad, ymax + pad)
    else:
        finite_positive_y = np.concatenate([
            np.asarray(y)[np.isfinite(y) & (np.asarray(y) > 0)]
            for y in y_values
        ])

        if finite_positive_y.size == 0:
            ylim = (1e-99, 1.0)
        else:
            ylim = (
                np.min(finite_positive_y) * 0.5,
                np.max(finite_positive_y) * 10.0,
            )

    return xlim, ylim


def plot_profile(ax, profile, data_list, xaxis='r', axislims=None,
                 legend=True, no_spec=False, grid=False, for_movie=False):
    """
    Plot a specified snapshot profile on the passed axis.

    Parameters
    ----------
    ax : matplotlib Axis
        Axis object on which to plot.
    profile : {'rho', 'm', 'v2', 'eta'}
        Profile to plot. rho, m, and v2 show a solid total-system curve plus
        per-species curves. eta is a single global curve.
    data_list : list of dict
        Dictionaries returned by extract_snapshot_data().
    xaxis : {'r', 'm'}, optional
        Plot against radius or total enclosed mass. Default is 'r'.
    axislims : tuple or None
        ((xmin, xmax), (ymin, ymax)).
    legend : bool, optional
        If True, include a legend.
    no_spec : bool, optional
        If True, suppress the species/linestyle legend.
    grid : bool, optional
        If True, show a grid.
    for_movie : bool, optional
        Internal flag controlling movie colors.
    """
    _validate_profile(profile)
    if xaxis not in {'r', 'm'}:
        raise ValueError("xaxis must be either 'r' or 'm'.")

    # Set colormap
    if for_movie:
        from matplotlib.colors import ListedColormap
        if len(data_list) == 1:
            cmap = ListedColormap(['black'])
        else:
            cmap = ListedColormap(['gray', 'black'])
    else:
        cmap = plt.get_cmap('tab20')

    # Stable linestyle mapping by species name.
    species_names = (
        sorted(data_list[0]['species'].keys())
        if data_list and 'species' in data_list[0]
        else []
    )
    base_styles = [
        (0, (5, 2)),
        (0, (1, 1)),
        (0, (3, 1, 1, 1)),
        (0, (7, 3, 1, 3)),
        (0, (5, 1)),
        (0, (3, 5, 1, 5, 1, 5)),
    ]
    style_map = {
        name: base_styles[i % len(base_styles)]
        for i, name in enumerate(species_names)
    }

    if axislims is None:
        xlim, ylim = get_profile_axis_limits(profile, data_list, xaxis=xaxis)
    else:
        xlim, ylim = axislims

    for ind, data in enumerate(data_list):
        color = cmap(ind % 10)
        time_lbl = f"t={data['time']:.2e}"

        # Global/total curve.  For rho, m, and v2 this is the solid total-system
        # profile; eta is intrinsically a single global profile.
        X_tot = _profile_x_values(data, profile, xaxis=xaxis)
        y_tot = _profile_y_values(data, profile)
        ax.plot(X_tot, y_tot, lw=2.2, color=color, ls='solid', label=time_lbl)

        # Per-species curves.
        if profile != 'eta':
            for sp in species_names:
                X_sp = _profile_x_values(
                    data, profile, xaxis=xaxis, species=sp
                )
                y_sp = _profile_y_values(data, profile, species=sp)
                ax.plot(
                    X_sp,
                    y_sp,
                    lw=1.8,
                    color=color,
                    ls=style_map[sp],
                    label='_nolegend_',
                )

    # Cosmetics
    ax.set_xscale('log')
    if profile != 'eta':
        ax.set_yscale('log')
    else:
        ax.axhline(0.5, color='black', ls='--', lw=2)
        ax.text(
            x=xlim[1], y=0.501, s='equipartition',
            ha='right', va='bottom'
        )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if xaxis == 'r':
        ax.set_xlabel(r'Radius [$r_\mathrm{s,0}$]', fontsize=14)
    else:
        ax.set_xlabel(r'$M_\mathrm{enc}$ [$M_\mathrm{s}$]', fontsize=14)

    ax.set_ylabel(profile, fontsize=14)
    ax.tick_params(axis='both', labelsize=12)

    if legend:
        time_legend = ax.legend(loc='lower center', frameon=True)

        if not no_spec and profile != 'eta':
            species_handles = [
                Line2D([0], [0], lw=2.2, color='black', ls='solid', label='total')
            ]
            species_handles += [
                Line2D(
                    [0], [0], lw=1.8, color='black',
                    ls=style_map[sp], label=sp
                )
                for sp in species_names
            ]
            species_legend = ax.legend(
                handles=species_handles,
                loc='lower left',
                frameon=True,
                ncol=1,
            )
            ax.add_artist(species_legend)

        ax.add_artist(time_legend)

    if grid:
        ax.grid(True, which='both', ls='--', alpha=0.4)


def plot_snapshots(model, snapshots=[0], profiles='rho', xaxis=None,
                   base_dir=None, filepath=None, show=False, grid=False,
                   for_movie=False):
    """
    Plot up to three profiles at specified points in time for one simulation.

    Parameters
    ----------
    model : State object, Config object, or model_no
        Each model can be a State, Config, or integer model number.
    snapshots : int or list of int
        Snapshot indices to plot.
    profiles : str or list of str, optional
        Profiles to plot. Options are 'rho', 'm', 'v2', 'eta'.
    xaxis : str or list of str, optional
        X-axis for each profile. Options are 'r' and 'm'. Default is 'r'.
    base_dir : str, optional
        Required if model is passed as an integer.
    filepath : str, optional
        If provided, save the plot to this file.
    show : bool, optional
        If True, show the plot even if saving.
    grid : bool, optional
        If True, show grid on axes.
    for_movie : bool, optional
        Internal flag controlling movie colors.
    """
    if not isinstance(snapshots, list):
        snapshots = [snapshots]

    profiles_list = profiles if isinstance(profiles, list) else [profiles]
    for profile in profiles_list:
        _validate_profile(profile)

    if xaxis is None:
        xaxis = ['r'] * len(profiles_list)
    elif isinstance(xaxis, str):
        xaxis = [xaxis]

    if len(xaxis) != len(profiles_list):
        raise ValueError("'xaxis' must have the same length as 'profiles'.")
    if any(x not in {'r', 'm'} for x in xaxis):
        raise ValueError("xaxis entries must be either 'r' or 'm'.")

    def _resolve_dir(model, ind):
        if hasattr(model, 'config'):
            return os.path.join(
                model.config.io.base_dir,
                model.config.io.model_dir,
                f"profile_{ind}.dat",
            )
        if hasattr(model, 'io'):
            return os.path.join(
                model.io.base_dir,
                model.io.model_dir,
                f"profile_{ind}.dat",
            )
        if isinstance(model, int):
            if base_dir is None:
                raise ValueError(
                    "'base_dir' (base directory) must be specified if using model numbers."
                )
            model_dir = f"Model{model:05d}"
            return os.path.join(base_dir, model_dir, f"profile_{ind}.dat")
        raise TypeError(
            f"Unrecognized model type: {type(model)}. Must be a State object, "
            "Config object, or integer."
        )

    # Change any '-1' entries to the last snapshot index.
    for ind, val in enumerate(snapshots):
        if val == -1:
            snapshot_indices_data = extract_snapshot_indices(
                os.path.dirname(_resolve_dir(model, 0))
            )
            snapshots[ind] = snapshot_indices_data['index'][-1]

    data_list = [
        extract_snapshot_data(_resolve_dir(model, ind))
        for ind in snapshots
    ]

    n = len(profiles_list)
    fig, axs = plt.subplots(1, n, figsize=(6 * n, 5))

    if n == 1:
        no_spec = profiles_list[0] == 'eta'
        plot_profile(
            axs,
            profiles_list[0],
            data_list,
            xaxis=xaxis[0],
            legend=True,
            no_spec=no_spec,
            grid=grid,
            for_movie=for_movie,
        )
    else:
        for ind, ax in enumerate(axs):
            legend = ind == 0
            plot_profile(
                ax,
                profiles_list[ind],
                data_list,
                xaxis=xaxis[ind],
                legend=legend,
                grid=grid,
                for_movie=for_movie,
            )

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
        Save the plot to this file.  Defaults to '/base_dir/ModelXXXXX/movie_{profiles}.mp4'
    base_dir : str, optional
        Required if any model is passed as an integer.  The directory in which all ModelXXXXX subdirectories reside.
    profiles : str or list of str, optional
        Profiles to plot. Options are 'rho', 'm', 'v2', 'eta'
    grid : bool, optional
        If True, shows grid on axes
    fps : int, optional
        Frames per second for the output movie. Default is 20

    Returns
    -------
    None
        Saves the movie as an MP4 file in the model directory.
    """

    # Get the model directory
    if hasattr(model, 'config'):        # Passed state object
        model_dir = os.path.join(model.config.io.base_dir, model.config.io.model_dir)
    elif hasattr(model, 'io'):          # Passed config object
        model_dir = os.path.join(model.io.base_dir, model.io.model_dir)
    elif isinstance(model, int):        # Passed model number
        if base_dir is None:
            raise ValueError("'base_dir' (base directory) must be specified if using model numbers.")
        model_dir = f"Model{model:05d}"
        model_dir = os.path.join(base_dir, model_dir)
    else:
        raise TypeError(f"Unrecognized model type: {type(model)}. Must be a State object, Config object, or integer.")
    
    # Load snapshot indices
    snapshot_indices_data = extract_snapshot_indices(model_dir)
    indices = snapshot_indices_data['index']

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

def _deluxe_frame(args):
    """
    Worker function for rendering one movie frame.

    Must be top-level so ProcessPoolExecutor can pickle it.
    """
    (
        ind, model_dir, temp_dir, n, profiles, insets, xaxis, add_radii,
        axislims, grid, tevo_t, time_data,
    ) = args

    import os
    import numpy as np

    # Safer for multiprocessing / headless rendering.
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib import transforms

    snapshot_path = os.path.join(model_dir, f"profile_{ind}.dat")
    if not os.path.isfile(snapshot_path):
        return None

    image_path = os.path.join(temp_dir, f"frame_{ind:04d}.png")

    # Extract data for current frame and initial frame
    initial_snapshot_path = os.path.join(model_dir, "profile_0.dat")
    data_list = [
        extract_snapshot_data(initial_snapshot_path),
        extract_snapshot_data(snapshot_path),
    ]
    frame_t = data_list[1]["time"]

    # Plot profile and initial profile
    fig, axs = plt.subplots(1, n, figsize=(6 * n, 5))
    axs = np.atleast_1d(axs)

    for i, ax in enumerate(axs):
        profile = profiles[i]
        inset = insets[i]
        xax = xaxis[i]

        legend = True if i == 0 else False

        plot_profile(
            ax,
            profile,
            data_list,
            xaxis=xax,
            axislims=axislims[i],
            legend=legend,
            grid=grid,
            for_movie=True,
        )

        if add_radii is not None:
            frac_up = 0.15

            for radius in add_radii:

                # One radius per species
                if radius in ["r01", "r05", "r10", "r20", "r50", "r90"]:

                    for spec in time_data["species"]:
                        r = np.interp(
                            frame_t,
                            tevo_t,
                            time_data["species"][spec][radius],
                        )

                        if xax == "r":

                            # If r is outside the x-axis limits, skip plotting
                            if (
                                r < axislims[i][0][0]
                                or r > axislims[i][0][1]
                            ):
                                continue

                            ax.axvline(
                                r,
                                color="red",
                                ls="--",
                                zorder=-10,
                            )

                            trans = transforms.blended_transform_factory(
                                ax.transData,
                                ax.transAxes,
                            )

                            ax.text(
                                r,
                                frac_up,
                                f"{radius}[{spec}]",
                                rotation=90,
                                color="red",
                                fontsize=10,
                                ha="right",
                                va="bottom",
                                zorder=-10,
                                transform=trans,
                            )

                        elif xax == "m":

                            m = _mass_coordinate_from_radius(data_list[1], r)

                            # If m is outside the x-axis limits, skip plotting
                            if (
                                m < axislims[i][0][0]
                                or m > axislims[i][0][1]
                            ):
                                continue

                            ax.axvline(
                                m,
                                color="red",
                                ls="--",
                                zorder=-10,
                            )

                            trans = transforms.blended_transform_factory(
                                ax.transData,
                                ax.transAxes,
                            )

                            ax.text(
                                m,
                                frac_up,
                                f"{radius}[{spec}]",
                                rotation=90,
                                color="red",
                                fontsize=10,
                                ha="right",
                                va="bottom",
                                zorder=-10,
                                transform=trans,
                            )

                else:

                    r = np.interp(
                        frame_t,
                        tevo_t,
                        time_data[radius],
                    )

                    if xax == "r":

                        # If r is outside the x-axis limits, skip plotting
                        if (
                            r < axislims[i][0][0]
                            or r > axislims[i][0][1]
                        ):
                            continue

                        ax.axvline(
                            r,
                            color="red",
                            ls="--",
                            zorder=-10,
                        )

                        trans = transforms.blended_transform_factory(
                            ax.transData,
                            ax.transAxes,
                        )

                        ax.text(
                            r,
                            frac_up,
                            radius,
                            rotation=90,
                            color="red",
                            fontsize=10,
                            ha="right",
                            va="bottom",
                            zorder=-10,
                            transform=trans,
                        )

                    elif xax == "m":

                        m = _mass_coordinate_from_radius(data_list[1], r)

                        # If m is outside the x-axis limits, skip plotting
                        if (
                            m < axislims[i][0][0]
                            or m > axislims[i][0][1]
                        ):
                            continue

                        ax.axvline(
                            m,
                            color="red",
                            ls="--",
                            zorder=-10,
                        )

                        trans = transforms.blended_transform_factory(
                            ax.transData,
                            ax.transAxes,
                        )

                        ax.text(
                            m,
                            frac_up,
                            radius,
                            rotation=90,
                            color="red",
                            fontsize=10,
                            ha="right",
                            va="bottom",
                            zorder=-10,
                            transform=trans,
                        )

        if inset is not None:
            tevo_y = time_data[inset]

            axin = ax.inset_axes([0.55, 0.65, 0.45, 0.35])

            axin.axvline(frame_t, color="grey")
            axin.plot(tevo_t, tevo_y, color="black")

            axin.scatter(
                frame_t,
                np.interp(frame_t, tevo_t, tevo_y),
                color="red",
                s=50,
            )

            axin.set_ylabel(inset, fontsize=12)
            axin.set_xlabel("$t$", fontsize=12)
            finite_tevo_y = np.asarray(tevo_y)[np.isfinite(tevo_y)]
            if finite_tevo_y.size and np.all(finite_tevo_y > 0):
                axin.set_yscale("log")

            axin.tick_params(
                axis="both",
                which="both",
                labelbottom=False,
                labelleft=False,
                labeltop=False,
                labelright=False,
                top=True,
                bottom=True,
                left=True,
                right=True,
                direction="in",
            )

    fig.savefig(
        image_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return image_path

def make_movie_deluxe_parallel(
    model,
    profiles=None,
    insets=None,
    xaxis=None,
    add_radii=None,
    filepath=None,
    base_dir=None,
    grid=False,
    fps=20,
):
    """
    Animate profiles with constant scale and with inset for time evolution.
    Scale stays constant throughout.

    Arguments
    ---------
    model : State object, Config object, or model_no
        Each model can be a State, Config, or integer model number.
    profiles : list of str, optional
        Profiles to plot. Options are 'rho', 'm', 'v2', 'eta'.
    insets : list of str or None, optional
        Inset plots to include. Options are global 1D quantities in time_evolution.txt.
    xaxis : list of str, optional
        X-axis for profiles to plot. Default is 'r'. Other option is 'm'.
    add_radii : list, optional
        List of radii to add to profiles from time_evolution.txt.
        Options: 'r_c', 'r01', 'r05', 'r10', 'r20', 'r50', 'r90'.
    filepath : str, optional
        Save the plot to this file.
    base_dir : str, optional
        Required if model is passed as an integer.
    grid : bool, optional
        If True, shows grid on axes.
    fps : int, optional
        Frames per second for the output movie. Default is 20.

    Returns
    -------
    None
        Saves the movie as an MP4 file in the model directory.
    """

    # Collect profiles and insets
    if profiles is None:
        profiles = ["rho", "v2"]
    elif isinstance(profiles, str):
        profiles = [profiles]

    if insets is None:
        insets = ["rho_c_tot"] + [None] * (len(profiles) - 1)
    elif isinstance(insets, str) or insets is None:
        insets = [insets]

    if xaxis is None:
        xaxis = ["r"] * len(profiles)
    elif isinstance(xaxis, str):
        xaxis = [xaxis]

    # Validate profiles
    valid_profiles = list(VALID_PROFILES)

    if any(profile not in valid_profiles for profile in profiles):
        raise ValueError(
            f"Invalid profile specified. Valid options are: {valid_profiles}"
        )

    # Validate radii
    valid_radii = ["r_c", "r01", "r05", "r10", "r20", "r50", "r90"]

    if add_radii is not None:

        if isinstance(add_radii, str):
            add_radii = [add_radii]

        if any(radius not in valid_radii for radius in add_radii):
            raise ValueError(
                f"Invalid radius specified. Valid options are: {valid_radii}"
            )

    # Validate xaxis
    valid_xaxis = ["r", "m"]

    if any(x not in valid_xaxis for x in xaxis):
        raise ValueError(
            f"Invalid x-axis specified. Valid options are: {valid_xaxis}"
        )
    if len(xaxis) != len(profiles):
        raise ValueError("'xaxis' must have the same length as 'profiles'.")

    # Number of panels
    n = len(profiles)

    # Get the model directory
    if hasattr(model, "config"):              # Passed state object

        model_dir = os.path.join(
            model.config.io.base_dir,
            model.config.io.model_dir,
        )

    elif hasattr(model, "io"):                # Passed config object

        model_dir = os.path.join(
            model.io.base_dir,
            model.io.model_dir,
        )

    elif isinstance(model, int):              # Passed model number

        if base_dir is None:
            raise ValueError(
                "'base_dir' (base directory) must be specified "
                "if using model numbers."
            )

        model_dir = f"Model{model:05d}"
        model_dir = os.path.join(base_dir, model_dir)

    else:

        raise TypeError(
            f"Unrecognized model type: {type(model)}. "
            "Must be a State object, Config object, or integer."
        )

    # Load time evolution data
    print("Getting time evolution data...")

    time_evolution_path = os.path.join(
        model_dir,
        "time_evolution.txt",
    )

    time_data = extract_time_evolution_data(
        time_evolution_path
    )

    tevo_t = time_data["time"]

    # Validate insets
    valid_insets = _valid_time_insets(time_data)

    if any(
        inset not in valid_insets
        for inset in insets
        if inset is not None
    ):
        raise ValueError(
            f"Invalid inset specified. Valid options are: {valid_insets}"
        )

    if len(insets) != len(profiles):
        raise ValueError(
            "'insets' must have the same length as 'profiles'."
        )

    # Load snapshot indices
    snapshot_indices_data = extract_snapshot_indices(model_dir)

    indices = snapshot_indices_data["index"]

    # Get axis limits
    print("Getting axis limits...")

    snapshot_data_list = []

    for ind in indices:

        snapshot_path = os.path.join(
            model_dir,
            f"profile_{ind}.dat",
        )

        if not os.path.isfile(snapshot_path):
            continue

        snapshot_data_list.append(
            extract_snapshot_data(snapshot_path)
        )

    axislims = []

    for i, profile in enumerate(profiles):

        xlim, ylim = get_profile_axis_limits(
            profile,
            snapshot_data_list,
            xaxis=xaxis[i],
        )

        axislims.append((xlim, ylim))

    # Create a temporary directory for storing images
    temp_dir = os.path.join(
        model_dir,
        "temp_images",
    )

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    os.makedirs(temp_dir)

    image_paths = []

    # Determine number of parallel processes
    max_workers = max(
        1,
        min(os.cpu_count() - 2, 7),
    )

    print(
        f"Generating {len(indices)} frames using "
        f"{max_workers} parallel processes..."
    )

    frame_args = [
        (
            ind,
            model_dir,
            temp_dir,
            n,
            profiles,
            insets,
            xaxis,
            add_radii,
            axislims,
            grid,
            tevo_t,
            time_data,
        )
        for ind in indices
    ]

    with ProcessPoolExecutor(
        max_workers=max_workers
    ) as executor:

        futures = [
            executor.submit(
                _deluxe_frame,
                args,
            )
            for args in frame_args
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Frames",
            unit="frame",
        ):

            image_path = future.result()

            if image_path is not None:
                image_paths.append(image_path)

    # Keep list deterministic
    # (although we never end up using it)
    image_paths.sort()

    print("Compiling into a movie using ffmpeg...")

    if filepath is not None:
        output_movie_path = filepath
    else:
        output_movie_path = os.path.join(
            model_dir,
            "movie_deluxe.mp4",
        )

    # Construct the ffmpeg command
    movie_command = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        os.path.join(
            temp_dir,
            "frame_%04d.png",
        ),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        output_movie_path,
    ]

    subprocess.run(
        movie_command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        check=True,
    )

    print("Deleting frames...")

    shutil.rmtree(
        temp_dir,
        ignore_errors=True,
    )

    print(
        f"Movie saved to {output_movie_path}"
    )

def make_movie_deluxe_serial(model, profiles=None, insets=None, xaxis=None, add_radii=None, filepath=None, base_dir=None, grid=False, fps=20,):
    """
    Animate profiles wit constant scale and with inset for time evolution.
    Scale stays constant throughout.

    Arguments
    ---------
    model : State object, Config object, or model_no
        Each model can be a State, Config, or integer model number.
    profiles : list of str, optional
        Profiles to plot. Options are 'rho', 'm', 'v2', 'eta'.
    insets : list of str or None, optional
        Inset plots to include. Options are global 1D quantities in time_evolution.txt.
    xaxis : list of str, optional
        X-axis for profiles to plot.  Default is 'r'.  Other option is 'm'.
    add_radii : list, optional
        List of radii to add to profiles from time_evolution.txt
        Options: 'r_c', 'r01', 'r05', 'r10', 'r20', 'r50', 'r90'.
    filepath : str, optional
        Save the plot to this file.  Defaults to '/base_dir/ModelXXXXX/movie_deluxe.mp4'
    base_dir : str, optional
        Required if any model is passed as an integer.  The directory in which all ModelXXXXX subdirectories reside.
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
    valid_profiles = list(VALID_PROFILES)
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
    if len(xaxis) != len(profiles):
        raise ValueError("'xaxis' must have the same length as 'profiles'.")

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
        model_dir = f"Model{model:05d}"
        model_dir = os.path.join(base_dir, model_dir)
    else:
        raise TypeError(f"Unrecognized model type: {type(model)}. Must be a State object, Config object, or integer.")
    
    # Load time evolution data
    print(f"Getting time evolution data...")
    time_evolution_path = os.path.join(model_dir, f"time_evolution.txt")
    time_data = extract_time_evolution_data(time_evolution_path)
    tevo_t = time_data['time']

    # Validate insets
    valid_insets = _valid_time_insets(time_data)
    if any(inset not in valid_insets for inset in insets if inset is not None):
        raise ValueError(f"Invalid inset specified. Valid options are: {valid_insets}")
    if len(insets) != len(profiles):
        raise ValueError("'insets' must have the same length as 'profiles'.")

    # Load snapshot indices
    snapshot_indices_data   = extract_snapshot_indices(model_dir)
    indices                 = snapshot_indices_data['index']

    # Get axis limits
    print(f"Getting axis limits...")
    snapshot_data_list = []

    for ind in indices:
        snapshot_path = os.path.join(model_dir, f"profile_{ind}.dat")

        if not os.path.isfile(snapshot_path):
            continue

        snapshot_data_list.append(extract_snapshot_data(snapshot_path))

    axislims = []

    for i, profile in enumerate(profiles):
        xlim, ylim = get_profile_axis_limits(profile, snapshot_data_list, xaxis=xaxis[i])
        axislims.append((xlim, ylim))

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
        frame_t = data_list[1]['time']

        # Plot profile and initial profile
        fig, axs = plt.subplots(1, n, figsize=(6*n, 5))
        axs = np.atleast_1d(axs)

        for i, ax in enumerate(axs):
            profile = profiles[i]
            inset   = insets[i]
            xax     = xaxis[i]

            legend = True if i == 0 else False
            plot_profile(ax, profile, data_list, xaxis=xax, axislims=axislims[i], legend=legend, grid=grid, for_movie=True)

            if add_radii is not None:
                frac_up = 0.15
                for radius in add_radii:
                    if radius in ['r01', 'r05', 'r10', 'r20', 'r50', 'r90']: # One per species
                        for spec in time_data['species']:
                            r = np.interp(frame_t, tevo_t, time_data['species'][spec][radius])
                            if xax == 'r':
                                # If r is outside the x-axis limits, skip plotting
                                if r < axislims[i][0][0] or r > axislims[i][0][1]:
                                    continue
                                ax.axvline(r, color='red', ls='--', zorder=-10)
                                trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
                                ax.text(r, frac_up, f"{radius}[{spec}]", rotation=90, color='red', fontsize=10, ha='right', va='bottom', zorder=-10, transform=trans)
                            elif xax == 'm':
                                m = _mass_coordinate_from_radius(data_list[1], r)
                                # If r is outside the x-axis limits, skip plotting
                                if m < axislims[i][0][0] or m > axislims[i][0][1]:
                                    continue
                                ax.axvline(m, color='red', ls='--', zorder=-10)
                                trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
                                ax.text(m, frac_up, f"{radius}[{spec}]", rotation=90, color='red', fontsize=10, ha='right', va='bottom', zorder=-10, transform=trans)
                    else:
                        r = np.interp(frame_t, tevo_t, time_data[radius])
                        if xax == 'r':
                            # If r is outside the x-axis limits, skip plotting
                            if r < axislims[i][0][0] or r > axislims[i][0][1]:
                                continue
                            ax.axvline(r, color='red', ls='--', zorder=-10)
                            trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
                            ax.text(r, frac_up, radius, rotation=90, color='red', fontsize=10, ha='right', va='bottom', zorder=-10, transform=trans)
                        elif xax == 'm':
                            m = _mass_coordinate_from_radius(data_list[1], r)
                            # If r is outside the x-axis limits, skip plotting
                            if m < axislims[i][0][0] or m > axislims[i][0][1]:
                                continue
                            ax.axvline(m, color='red', ls='--', zorder=-10)
                            trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
                            ax.text(m, frac_up, radius, rotation=90, color='red', fontsize=10, ha='right', va='bottom', zorder=-10, transform=trans)

            if inset is not None:
                tevo_y = time_data[inset]
                axin = ax.inset_axes([0.55, 0.65, 0.45, 0.35])
                axin.axvline(frame_t, color='grey')
                axin.plot(tevo_t, tevo_y, color='black')
                axin.scatter(frame_t, np.interp(frame_t, tevo_t, tevo_y),
                            color='red', s=50)
                axin.set_ylabel(inset, fontsize=12)
                axin.set_xlabel('$t$', fontsize=12)
                finite_tevo_y = np.asarray(tevo_y)[np.isfinite(tevo_y)]
                if finite_tevo_y.size and np.all(finite_tevo_y > 0):
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

def make_movie_deluxe(model, parallel=True, **kwargs):
    """
    Top-level function for calling make_movie_deluxe,
    either serial or parallel.
    """
    if parallel:
        make_movie_deluxe_parallel(model, **kwargs)
    else:
        make_movie_deluxe_serial(model, **kwargs)

# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
# import matplotlib.transforms as transforms
# import subprocess
# from tqdm import tqdm
# import shutil
# from pygtf2.io.read import extract_snapshot_data, extract_snapshot_indices, extract_time_evolution_data
# from concurrent.futures import ProcessPoolExecutor, as_completed

# def get_profile_axis_limits(profile, data_list, xaxis='r'):
#     """
#     Compute global axis limits for multi-species profile plots.
#     Includes total profiles where plotted, and species profiles where plotted.
#     """

#     xlim_lower = np.inf
#     xlim_upper = -np.inf
#     ylim_lower = np.inf
#     ylim_upper = -np.inf

#     species_names = (
#         sorted(data_list[0]['species'].keys())
#         if data_list and 'species' in data_list[0]
#         else []
#     )

#     for data in data_list:

#         # ---------- Total profiles ----------
#         if profile in {'rho', 'm', 'p', 'eta'}:
#             if xaxis == 'r':
#                 xkey_tot = 'log_r' if profile == 'm' else 'log_rmid'
#                 x_tot = 10.0**data[xkey_tot]
#             elif xaxis == 'm':
#                 x_tot = data['m_tot']

#             if profile in {'rho', 'p'}:
#                 y_tot = data[profile + '_tot']
#             elif profile == 'm':
#                 y_tot = data['m_tot']
#             elif profile == 'eta':
#                 y_tot = data['eta']

#             xlim_lower = min(xlim_lower, np.min(x_tot) * 0.8)
#             xlim_upper = max(xlim_upper, np.max(x_tot) * 1.2)

#             positive_y = y_tot[y_tot > 0]
#             if positive_y.size:
#                 ylim_lower = min(ylim_lower, np.min(positive_y) * 0.5)
#             ylim_upper = max(ylim_upper, np.max(y_tot) * 10.0)

#         # ---------- Species profiles ----------
#         if profile != 'eta':
#             for sp in species_names:
#                 sp_data = data['species'][sp]

#                 if xaxis == 'r':
#                     sp_xkey = 'lgr' if profile == 'm' else 'lgrm'
#                     x_sp = 10.0**sp_data[sp_xkey]
#                 elif xaxis == 'm':
#                     x_sp = data['m_tot']

#                 y_sp = sp_data[profile]

#                 xlim_lower = min(xlim_lower, np.min(x_sp) * 0.8)
#                 xlim_upper = max(xlim_upper, np.max(x_sp) * 1.2)

#                 positive_y = y_sp[y_sp > 0]
#                 if positive_y.size:
#                     ylim_lower = min(ylim_lower, np.min(positive_y) * 0.5)
#                 ylim_upper = max(ylim_upper, np.max(y_sp) * 10.0)

#     # Safeguards
#     if not np.isfinite(ylim_lower) or ylim_lower <= 0:
#         ylim_lower = 1e-99
#     if not np.isfinite(ylim_upper) or ylim_upper <= 0:
#         ylim_upper = 1.0

#     if profile == 'eta':
#         if ylim_upper < 0.6:
#             ylim_upper = 0.6
#         ylim_lower *= 0.9
#         ylim_upper *= 1.1

#     return (xlim_lower, xlim_upper), (ylim_lower, ylim_upper)

# def plot_profile(ax, profile, data_list, xaxis='r', axislims=None,
#                  legend=True, no_spec=False, grid=False, for_movie=False):
#     """
#     Plot specified profile on the passed axis object

#     Arguments
#     ---------
#     ax : Axis
#         Axis object on which to plot
#     profile : str
#         Profile to plot.  Options are 'rho', 'm', 'v2', 'p', 'trelax', 'eta'
#     data_list : dict
#         Dictionary returned by extract_snapshot_data()
#     xaxis : str, optional
#         X-axis to plot.  Default is 'r'.  Other option is 'm'.
#     axislims : list of tuples or None
#         [(xmin, xmax), (ymin, ymax)]
#     legend : bool, optional
#         If True, include a legend in the plot
#     no_spec : bool, optional
#         If True, do not include species legend
#     grid : bool, optional
#         If True, shows grid on axes
#     for_movie : bool, should not be set by user
#         If True, then plot_snapshots() is being called by make_movie()
#         This controls the colormap of the plots
#     """
#     # Set colormap
#     if for_movie:
#         from matplotlib.colors import ListedColormap
#         if len(data_list) == 1:
#             cmap = ListedColormap(['black'])
#         else:
#             cmap = ListedColormap(['gray', 'black'])
#     else:
#         cmap = plt.get_cmap('tab20')

#     # Pick linestyles for species; stable mapping by species name
#     species_names = sorted(data_list[0]['species'].keys()) if data_list and 'species' in data_list[0] else []
#     base_styles = [
#         (0, (5, 2)),              # long dash
#         (0, (1, 1)),              # fine dotted
#         (0, (3, 1, 1, 1)),        # dash-dot pattern
#         (0, (7, 3, 1, 3)),        # long dash, small dot, medium gap
#         (0, (5, 1)),              # medium dash, tight gaps
#         (0, (3, 5, 1, 5, 1, 5)),  # mixed dash/dot combo
#     ]
#     style_map = {name: base_styles[i % len(base_styles)] for i, name in enumerate(species_names)}

#     # Totals use the global x keys
#     if xaxis == 'r':
#         xkey_tot = 'log_r' if profile == 'm' else 'log_rmid'
#     elif xaxis == 'm':
#         xkey_tot = 'm_tot'
#     # xkey_tot = 'log_r' if profile == 'm' else 'log_rmid' # OLD

#     if axislims is None:
#         xlim, ylim = get_profile_axis_limits(profile, data_list, xaxis=xaxis)
#     else:
#         xlim, ylim = axislims

#     # Plot
#     for ind, data in enumerate(data_list):
#         color = cmap(ind % 10)
#         time_lbl = f"t={data['time']:.2e}"

#         # totals (solid), if applicable
#         if xaxis == 'r':
#             X_tot = 10.0**data[xkey_tot]
#         elif xaxis == 'm':
#             X_tot = data[xkey_tot]
#         if profile in {'rho', 'm', 'p'}:
#             y_tot = data[profile + '_tot'] if profile != 'm' else data['m_tot']
#         elif profile in {'eta'}:
#             y_tot = data[profile]
#         if profile in {'rho', 'm', 'p', 'eta'}:
#             ax.plot(X_tot, y_tot, lw=2.2, color=color, ls='solid', label=time_lbl)
        
#         # species (own linestyle), no extra legend spam
#         if profile not in {'eta'}:
#             for isp, sp in enumerate(species_names):
#                 label = '_nolegend_'
#                 if profile in {'trelax', 'v2'} and isp == 1:
#                     label = time_lbl

#                 if xaxis == 'r':
#                     sp_xkey = 'lgr' if profile == 'm' else 'lgrm'
#                     X_sp = 10.0**data['species'][sp][sp_xkey]
#                 elif xaxis == 'm':
#                     X_sp = data['m_tot']

#                 y_sp = data['species'][sp][profile]
#                 ax.plot(X_sp, y_sp, lw=1.8, color=color, ls=style_map[sp], label=label)

#     # Cosmetics
#     ax.set_xscale('log')
#     if profile not in {'eta'}:
#         ax.set_yscale('log')
#     if profile == 'eta':
#         ax.axhline(0.5, color='black', ls='--', lw=2)
#         ax.text(x=xlim[1], y=0.501, s='equipartition', ha='right', va='bottom')
#     ax.set_xlim(xlim)
#     ax.set_ylim(ylim)
#     if xaxis == 'r':
#         ax.set_xlabel(r'Radius [$r_\mathrm{s,0}$]', fontsize=14)
#     elif xaxis == 'm':
#         ax.set_xlabel(r'$M_\mathrm{enc}$ [$M_\mathrm{s}$]', fontsize=14)
#     ax.set_ylabel(profile, fontsize=14)
#     ax.tick_params(axis='both', labelsize=12)
#     if legend:
#         # ax.legend() # OLD
#         # 1) Time legend (colors): use the labels already attached to plotted lines
#         time_legend = ax.legend(loc='lower center', frameon=True)

#         # 2) Species legend (linestyles in black), including 'total' as solid
#         if not no_spec and profile != 'eta':
#             species_handles = [Line2D([0], [0], lw=2.2, color='black', ls='solid', label='total')]
#             species_handles += [
#                 Line2D([0], [0], lw=1.8, color='black', ls=style_map[sp], label=sp)
#                 for sp in species_names
#             ]

#             species_legend = ax.legend(handles=species_handles, loc='lower left', frameon=True, ncol=1)

#         # Keep both legends
#         ax.add_artist(time_legend)
#         if not no_spec and profile != 'eta':
#             ax.add_artist(species_legend)

#     if grid:
#         ax.grid(True, which="both", ls="--", alpha=0.4)

# def plot_snapshots(model, snapshots=[0], profiles='rho', xaxis=None, base_dir=None, filepath=None, show=False, grid=False, for_movie=False):
#     """
#     Plot up to three profiles at specified points in time for one simulation

#     Arguments
#     ---------
#     model : State object, Config object, or model_no
#         Each model can be a State, Config, or integer model number.
#     snapshots : int or list of int
#         Snapshot indices to plot
#     profiles : str or list of str, optional
#         Profiles to plot.  Options are 'rho', 'm', 'v2', 'p', 'trelax', 'eta'
#     xaxis : list of str, optional
#         X-axis for profiles to plot.  Default is 'r'.  Other option is 'm'.
#     base_dir : str, optional
#         Required if any model is passed as an integer.  The directory in which all ModelXXXXX subdirectories reside.
#     filepath : str, optional
#         If provided, save the plot to this file.
#     show : bool, optional
#         If True, show the plot even if saving.  Default is False.
#     grid : bool, optional
#         If True, shows grid on axes
#     for_movie : bool, should not be set by user
#         If True, then being called by make_movie()
#         This controls the colormap of the plots
#     """

#     if type(snapshots) != list:
#         snapshots = [snapshots]

#     if xaxis is None:
#         xaxis = ['r'] * len(profiles)
#     elif isinstance(xaxis, str):
#         xaxis = [xaxis]

#     def _resolve_dir(model, ind):
#         if hasattr(model, 'config'): # Passed state object
#             return os.path.join(model.config.io.base_dir, model.config.io.model_dir, f"profile_{ind}.dat")
#         elif hasattr(model, 'io'): # Passed config object
#             return os.path.join(model.io.base_dir, model.io.model_dir, f"profile_{ind}.dat")
#         elif isinstance(model, int): # Passed model number
#             if base_dir is None:
#                 raise ValueError("'base_dir' (base directory) must be specified if using model numbers.")
#             model_dir = f"Model{model:05d}"
#             return os.path.join(base_dir, model_dir, f"profile_{ind}.dat")
#         else:
#             raise TypeError(f"Unrecognized model type: {type(model)}. Must be a State object, Config object, or integer.")

#     # Change any '-1' entries to the last snapshot index
#     for ind, val in enumerate(snapshots):
#         if val == -1:
#             snapshot_indices_data = extract_snapshot_indices(os.path.dirname(_resolve_dir(model, 0)))
#             snapshots[ind] = snapshot_indices_data['index'][-1]

#     profiles_list = profiles if isinstance(profiles, list) else [profiles]
#     n = len(profiles_list) # Number of panels

#     data_list = [extract_snapshot_data(_resolve_dir(model,ind)) for ind in snapshots]

#     fig, axs = plt.subplots(1, n, figsize=(6*n, 5))

#     if n == 1:
#         no_spec = profiles_list[0] == 'eta'
#         plot_profile(axs, profiles_list[0], data_list, xaxis=xaxis[0], legend=True, no_spec=no_spec, grid=grid, for_movie=for_movie)
#     else:
#         for ind, ax in enumerate(axs):
#             # legend = False if ind < len(axs) - 1 else True
#             legend = False if ind > 0 else True
#             plot_profile(ax, profiles_list[ind], data_list, xaxis=xaxis[ind], legend=legend, grid=grid, for_movie=for_movie)

#     if filepath:
#         fig.savefig(filepath, dpi=300, bbox_inches='tight')
#         if show:
#             plt.show()
#         else:
#             plt.close(fig)
#     else:
#         plt.show()

# def make_movie(model, filepath=None, base_dir=None, profiles='rho', grid=False, fps=20):
#     """
#     Animate up to three profiles for one simulation

#     Arguments
#     ---------
#     model : State object, Config object, or model_no
#         Each model can be a State, Config, or integer model number.
#     filepath : str, optional
#         Save the plot to this file.  Defaults to '/base_dir/ModelXXXXX/movie_{profiles}.mp4'
#     base_dir : str, optional
#         Required if any model is passed as an integer.  The directory in which all ModelXXXXX subdirectories reside.
#     profiles : str or list of str, optional
#         Profiles to plot.  Options are 'rho', 'm', 'v2', 'p', 'trelax', 'kn'
#     grid : bool, optional
#         If True, shows grid on axes
#     fps : int, optional
#         Frames per second for the output movie. Default is 20

#     Returns
#     -------
#     None
#         Saves the movie as an MP4 file in the model directory.
#     """

#     n = 1 if type(profiles) != list else len(profiles) # number of panels

#     # Get the model directory
#     if hasattr(model, 'config'):        # Passed state object
#         model_dir = os.path.join(model.config.io.base_dir, model.config.io.model_dir)
#     elif hasattr(model, 'io'):          # Passed config object
#         model_dir = os.path.join(model.io.base_dir, model.io.model_dir)
#     elif isinstance(model, int):        # Passed model number
#         if base_dir is None:
#             raise ValueError("'base_dir' (base directory) must be specified if using model numbers.")
#         model_dir = f"Model{model:05d}"
#         model_dir = os.path.join(base_dir, model_dir)
#     else:
#         raise TypeError(f"Unrecognized model type: {type(model)}. Must be a State object, Config object, or integer.")
    
#     # Load snapshot indices
#     snapshot_indices_data = extract_snapshot_indices(model_dir)
#     indices = snapshot_indices_data['snapshot_index']

#     # Create a temporary directory for storing images
#     temp_dir = os.path.join(model_dir, "temp_images")
#     if os.path.exists(temp_dir):
#         shutil.rmtree(temp_dir)         # Delete the directory and all its contents
#     os.makedirs(temp_dir)

#     image_paths = []                    # List to store paths of generated images

#     print(f"Generating {len(indices)} frames...")
#     for ind in tqdm(indices, desc="Frames", unit="frame"):
#         snapshot_path = os.path.join(model_dir, f"profile_{ind}.dat")
#         if not os.path.isfile(snapshot_path):
#             continue                    # Skip if the snapshot file does not exist

#         # Define the output image path for the current frame
#         image_path = os.path.join(temp_dir, f"frame_{ind:04d}.png")

#         # Plot the profile, including the initial profile for comparison
#         if ind == 0:
#             plot_snapshots(model, profiles=profiles, base_dir=base_dir, filepath=image_path, grid=grid, for_movie=True)
#         else:
#             plot_snapshots(model, snapshots=[0,ind], profiles=profiles, base_dir=base_dir, filepath=image_path, grid=grid, for_movie=True)

#         image_paths.append(image_path)  # Add the image path to the list

#     print("Compiling into a movie using ffmpeg...")
#     # Define the output movie path
#     if isinstance(profiles, (list, tuple)):
#         profiles_str = "_".join(map(str, profiles))
#     else:
#         profiles_str = str(profiles)

#     output_movie_path = (
#         filepath if filepath is not None 
#         else os.path.join(model_dir, f"movie_{profiles_str}.mp4")
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

# def _deluxe_frame(args):
#     """
#     Worker function for rendering one movie frame.

#     Must be top-level so ProcessPoolExecutor can pickle it.
#     """
#     (
#         ind, model_dir, temp_dir, n, profiles, insets, xaxis, add_radii,
#         axislims, grid, index_t, tevo_t, time_data,
#     ) = args

#     import os
#     import numpy as np

#     # Safer for multiprocessing / headless rendering.
#     import matplotlib
#     matplotlib.use("Agg", force=True)
#     import matplotlib.pyplot as plt
#     from matplotlib import transforms

#     snapshot_path = os.path.join(model_dir, f"profile_{ind}.dat")
#     if not os.path.isfile(snapshot_path):
#         return None

#     image_path = os.path.join(temp_dir, f"frame_{ind:04d}.png")

#     # Extract data for current frame and initial frame
#     initial_snapshot_path = os.path.join(model_dir, "profile_0.dat")
#     data_list = [
#         extract_snapshot_data(initial_snapshot_path),
#         extract_snapshot_data(snapshot_path),
#     ]

#     # Plot profile and initial profile
#     fig, axs = plt.subplots(1, n, figsize=(6 * n, 5))
#     axs = np.atleast_1d(axs)

#     for i, ax in enumerate(axs):
#         profile = profiles[i]
#         inset = insets[i]
#         xax = xaxis[i]

#         legend = True if i == 0 else False

#         plot_profile(
#             ax,
#             profile,
#             data_list,
#             xaxis=xax,
#             axislims=axislims[profile],
#             legend=legend,
#             grid=grid,
#             for_movie=True,
#         )

#         if add_radii is not None:
#             frac_up = 0.15

#             for radius in add_radii:

#                 # One radius per species
#                 if radius in ["r01", "r05", "r10", "r20", "r50", "r90"]:

#                     for spec in time_data["species"]:
#                         r = np.interp(
#                             index_t[ind],
#                             tevo_t,
#                             time_data["species"][spec][radius],
#                         )

#                         if xax == "r":

#                             # If r is outside the x-axis limits, skip plotting
#                             if (
#                                 r < axislims[profile][0][0]
#                                 or r > axislims[profile][0][1]
#                             ):
#                                 continue

#                             ax.axvline(
#                                 r,
#                                 color="red",
#                                 ls="--",
#                                 zorder=-10,
#                             )

#                             trans = transforms.blended_transform_factory(
#                                 ax.transData,
#                                 ax.transAxes,
#                             )

#                             ax.text(
#                                 r,
#                                 frac_up,
#                                 f"{radius}[{spec}]",
#                                 rotation=90,
#                                 color="red",
#                                 fontsize=10,
#                                 ha="right",
#                                 va="bottom",
#                                 zorder=-10,
#                                 transform=trans,
#                             )

#                         elif xax == "m":

#                             m = np.interp(
#                                 r,
#                                 10 ** data_list[1]["log_r"],
#                                 data_list[1]["m_tot"],
#                             )

#                             # If m is outside the x-axis limits, skip plotting
#                             if (
#                                 m < axislims[profile][0][0]
#                                 or m > axislims[profile][0][1]
#                             ):
#                                 continue

#                             ax.axvline(
#                                 m,
#                                 color="red",
#                                 ls="--",
#                                 zorder=-10,
#                             )

#                             trans = transforms.blended_transform_factory(
#                                 ax.transData,
#                                 ax.transAxes,
#                             )

#                             ax.text(
#                                 m,
#                                 frac_up,
#                                 f"{radius}[{spec}]",
#                                 rotation=90,
#                                 color="red",
#                                 fontsize=10,
#                                 ha="right",
#                                 va="bottom",
#                                 zorder=-10,
#                                 transform=trans,
#                             )

#                 else:

#                     r = np.interp(
#                         index_t[ind],
#                         tevo_t,
#                         time_data[radius],
#                     )

#                     if xax == "r":

#                         # If r is outside the x-axis limits, skip plotting
#                         if (
#                             r < axislims[profile][0][0]
#                             or r > axislims[profile][0][1]
#                         ):
#                             continue

#                         ax.axvline(
#                             r,
#                             color="red",
#                             ls="--",
#                             zorder=-10,
#                         )

#                         trans = transforms.blended_transform_factory(
#                             ax.transData,
#                             ax.transAxes,
#                         )

#                         ax.text(
#                             r,
#                             frac_up,
#                             radius,
#                             rotation=90,
#                             color="red",
#                             fontsize=10,
#                             ha="right",
#                             va="bottom",
#                             zorder=-10,
#                             transform=trans,
#                         )

#                     elif xax == "m":

#                         m = np.interp(
#                             r,
#                             10 ** data_list[1]["log_r"],
#                             data_list[1]["m"],
#                         )

#                         # If m is outside the x-axis limits, skip plotting
#                         if (
#                             m < axislims[profile][0][0]
#                             or m > axislims[profile][0][1]
#                         ):
#                             continue

#                         ax.axvline(
#                             m,
#                             color="red",
#                             ls="--",
#                             zorder=-10,
#                         )

#                         trans = transforms.blended_transform_factory(
#                             ax.transData,
#                             ax.transAxes,
#                         )

#                         ax.text(
#                             m,
#                             frac_up,
#                             radius,
#                             rotation=90,
#                             color="red",
#                             fontsize=10,
#                             ha="right",
#                             va="bottom",
#                             zorder=-10,
#                             transform=trans,
#                         )

#         if inset is not None:
#             tevo_y = time_data[inset]

#             if profile != "trelax":
#                 axin = ax.inset_axes([0.55, 0.65, 0.45, 0.35])
#             else:
#                 axin = ax.inset_axes([0.0, 0.65, 0.45, 0.35])

#             axin.axvline(index_t[ind], color="grey")
#             axin.plot(tevo_t, tevo_y, color="black")

#             axin.scatter(
#                 index_t[ind],
#                 np.interp(index_t[ind], tevo_t, tevo_y),
#                 color="red",
#                 s=50,
#             )

#             axin.set_ylabel(inset, fontsize=12)
#             axin.set_xlabel("$t$", fontsize=12)
#             axin.set_yscale("log")

#             axin.tick_params(
#                 axis="both",
#                 which="both",
#                 labelbottom=False,
#                 labelleft=False,
#                 labeltop=False,
#                 labelright=False,
#                 top=True,
#                 bottom=True,
#                 left=True,
#                 right=True,
#                 direction="in",
#             )

#     fig.savefig(
#         image_path,
#         dpi=300,
#         bbox_inches="tight",
#     )

#     plt.close(fig)

#     return image_path

# def make_movie_deluxe_parallel(
#     model,
#     profiles=None,
#     insets=None,
#     xaxis=None,
#     add_radii=None,
#     filepath=None,
#     base_dir=None,
#     grid=False,
#     fps=20,
# ):
#     """
#     Animate profiles with constant scale and with inset for time evolution.
#     Scale stays constant throughout.

#     Arguments
#     ---------
#     model : State object, Config object, or model_no
#         Each model can be a State, Config, or integer model number.
#     profiles : list of str, optional
#         Profiles to plot. Options are 'rho', 'm', 'v2', 'trelax', 'eta'.
#     insets : list of str or None, optional
#         Inset plots to include. Options are any quantity in time_evolution.txt.
#     xaxis : list of str, optional
#         X-axis for profiles to plot. Default is 'r'. Other option is 'm'.
#     add_radii : list, optional
#         List of radii to add to profiles from time_evolution.txt.
#         Options: 'r_c', 'r01', 'r05', 'r10', 'r20', 'r50', 'r90'.
#     filepath : str, optional
#         Save the plot to this file.
#     base_dir : str, optional
#         Required if model is passed as an integer.
#     grid : bool, optional
#         If True, shows grid on axes.
#     fps : int, optional
#         Frames per second for the output movie. Default is 20.

#     Returns
#     -------
#     None
#         Saves the movie as an MP4 file in the model directory.
#     """

#     # Collect profiles and insets
#     if profiles is None:
#         profiles = ["rho", "v2"]
#     elif isinstance(profiles, str):
#         profiles = [profiles]

#     if insets is None:
#         insets = ["rho_c_tot"] + [None] * (len(profiles) - 1)
#     elif isinstance(insets, str) or insets is None:
#         insets = [insets]

#     if xaxis is None:
#         xaxis = ["r"] * len(profiles)
#     elif isinstance(xaxis, str):
#         xaxis = [xaxis]

#     # Validate profiles
#     valid_profiles = ["rho", "m", "v2", "trelax", "eta"]

#     if any(profile not in valid_profiles for profile in profiles):
#         raise ValueError(
#             f"Invalid profile specified. Valid options are: {valid_profiles}"
#         )

#     # Validate radii
#     valid_radii = ["r_c", "r01", "r05", "r10", "r20", "r50", "r90"]

#     if add_radii is not None:

#         if isinstance(add_radii, str):
#             add_radii = [add_radii]

#         if any(radius not in valid_radii for radius in add_radii):
#             raise ValueError(
#                 f"Invalid radius specified. Valid options are: {valid_radii}"
#             )

#     # Validate xaxis
#     valid_xaxis = ["r", "m"]

#     if any(x not in valid_xaxis for x in xaxis):
#         raise ValueError(
#             f"Invalid x-axis specified. Valid options are: {valid_xaxis}"
#         )

#     # Number of panels
#     n = len(profiles)

#     # Get the model directory
#     if hasattr(model, "config"):              # Passed state object

#         model_dir = os.path.join(
#             model.config.io.base_dir,
#             model.config.io.model_dir,
#         )

#     elif hasattr(model, "io"):                # Passed config object

#         model_dir = os.path.join(
#             model.io.base_dir,
#             model.io.model_dir,
#         )

#     elif isinstance(model, int):              # Passed model number

#         if base_dir is None:
#             raise ValueError(
#                 "'base_dir' (base directory) must be specified "
#                 "if using model numbers."
#             )

#         model_dir = f"Model{model:05d}"
#         model_dir = os.path.join(base_dir, model_dir)

#     else:

#         raise TypeError(
#             f"Unrecognized model type: {type(model)}. "
#             "Must be a State object, Config object, or integer."
#         )

#     # Load time evolution data
#     print("Getting time evolution data...")

#     time_evolution_path = os.path.join(
#         model_dir,
#         "time_evolution.txt",
#     )

#     time_data = extract_time_evolution_data(
#         time_evolution_path
#     )

#     tevo_t = time_data["time"]

#     # Validate insets
#     valid_insets = list(time_data.keys())

#     if any(
#         inset not in valid_insets
#         for inset in insets
#         if inset is not None
#     ):
#         raise ValueError(
#             f"Invalid inset specified. Valid options are: {valid_insets}"
#         )

#     if len(insets) != len(profiles):
#         raise ValueError(
#             "'insets' must have the same length as 'profiles'."
#         )

#     # Load snapshot indices
#     snapshot_indices_data = extract_snapshot_indices(model_dir)

#     indices = snapshot_indices_data["index"]
#     index_t = snapshot_indices_data["time"]

#     # Get axis limits
#     print("Getting axis limits...")

#     snapshot_data_list = []

#     for ind in indices:

#         snapshot_path = os.path.join(
#             model_dir,
#             f"profile_{ind}.dat",
#         )

#         if not os.path.isfile(snapshot_path):
#             continue

#         snapshot_data_list.append(
#             extract_snapshot_data(snapshot_path)
#         )

#     axislims = {}

#     for i, profile in enumerate(profiles):

#         xlim, ylim = get_profile_axis_limits(
#             profile,
#             snapshot_data_list,
#             xaxis=xaxis[i],
#         )

#         axislims[profile] = (xlim, ylim)

#     # Create a temporary directory for storing images
#     temp_dir = os.path.join(
#         model_dir,
#         "temp_images",
#     )

#     if os.path.exists(temp_dir):
#         shutil.rmtree(temp_dir)

#     os.makedirs(temp_dir)

#     image_paths = []

#     # Determine number of parallel processes
#     max_workers = max(
#         1,
#         min(os.cpu_count() - 2, 7),
#     )

#     print(
#         f"Generating {len(indices)} frames using "
#         f"{max_workers} parallel processes..."
#     )

#     frame_args = [
#         (
#             ind,
#             model_dir,
#             temp_dir,
#             n,
#             profiles,
#             insets,
#             xaxis,
#             add_radii,
#             axislims,
#             grid,
#             index_t,
#             tevo_t,
#             time_data,
#         )
#         for ind in indices
#     ]

#     with ProcessPoolExecutor(
#         max_workers=max_workers
#     ) as executor:

#         futures = [
#             executor.submit(
#                 _deluxe_frame,
#                 args,
#             )
#             for args in frame_args
#         ]

#         for future in tqdm(
#             as_completed(futures),
#             total=len(futures),
#             desc="Frames",
#             unit="frame",
#         ):

#             image_path = future.result()

#             if image_path is not None:
#                 image_paths.append(image_path)

#     # Keep list deterministic
#     # (although we never end up using it)
#     image_paths.sort()

#     print("Compiling into a movie using ffmpeg...")

#     if filepath is not None:
#         output_movie_path = filepath
#     else:
#         output_movie_path = os.path.join(
#             model_dir,
#             "movie_deluxe.mp4",
#         )

#     # Construct the ffmpeg command
#     movie_command = [
#         "ffmpeg",
#         "-y",
#         "-framerate",
#         str(fps),
#         "-i",
#         os.path.join(
#             temp_dir,
#             "frame_%04d.png",
#         ),
#         "-c:v",
#         "libx264",
#         "-pix_fmt",
#         "yuv420p",
#         "-vf",
#         "scale=trunc(iw/2)*2:trunc(ih/2)*2",
#         output_movie_path,
#     ]

#     subprocess.run(
#         movie_command,
#         stdout=subprocess.DEVNULL,
#         stderr=subprocess.STDOUT,
#         check=True,
#     )

#     print("Deleting frames...")

#     shutil.rmtree(
#         temp_dir,
#         ignore_errors=True,
#     )

#     print(
#         f"Movie saved to {output_movie_path}"
#     )

# def make_movie_deluxe_serial(model, profiles=None, insets=None, xaxis=None, add_radii=None, filepath=None, base_dir=None, grid=False, fps=20,):
#     """
#     Animate profiles wit constant scale and with inset for time evolution.
#     Scale stays constant throughout.

#     Arguments
#     ---------
#     model : State object, Config object, or model_no
#         Each model can be a State, Config, or integer model number.
#     profiles : list of str, optional
#         Profiles to plot.  Options are 'rho', 'm', 'v2', 'trelax', 'eta'.
#     insets : list of str or None, optional
#         Inset plots to include.  Options are any quantity in time_evolution.txt
#     xaxis : list of str, optional
#         X-axis for profiles to plot.  Default is 'r'.  Other option is 'm'.
#     add_radii : list, optional
#         List of radii to add to profiles from time_evolution.txt
#         Options: 'r_c', 'r_50[heavy]', etc
#     filepath : str, optional
#         Save the plot to this file.  Defaults to '/base_dir/ModelXXXXX/movie_deluxe.mp4'
#     base_dir : str, optional
#         Required if any model is passed as an integer.  The directory in which all ModelXXXXX subdirectories reside.
#     grid : bool, optional
#         If True, shows grid on axes
#     fps : int, optional
#         Frames per second for the output movie. Default is 20

#     Returns
#     -------
#     None
#         Saves the movie as an MP4 file in the model directory.
#     """
#     # Collect profiles and insets
#     if profiles is None:
#         profiles = ['rho', 'v2']
#     elif isinstance(profiles, str):
#         profiles = [profiles]
#     if insets is None:
#         insets = ['rho_c_tot'] + [None] * (len(profiles) - 1)
#     elif isinstance(insets, str) or insets is None:
#         insets = [insets]
#     if xaxis is None:
#         xaxis = ['r'] * len(profiles)
#     elif isinstance(xaxis, str):
#         xaxis = [xaxis]

#     # Validate profiles
#     valid_profiles = ['rho', 'm', 'v2', 'trelax', 'eta']
#     if any(profile not in valid_profiles for profile in profiles):
#         raise ValueError(f"Invalid profile specified. Valid options are: {valid_profiles}")
    
#     # Validate radii
#     valid_radii = ['r_c', 'r01', 'r05', 'r10', 'r20', 'r50', 'r90']
#     if add_radii is not None:
#         if isinstance(add_radii, str):
#             add_radii = [add_radii]
#         if any(radius not in valid_radii for radius in add_radii):
#             raise ValueError(f"Invalid radius specified. Valid options are: {valid_radii}")
        
#     # Validate xaxis
#     valid_xaxis = ['r', 'm']
#     if any(x not in valid_xaxis for x in xaxis):
#         raise ValueError(f"Invalid x-axis specified. Valid options are: {valid_xaxis}")

#     # Number of panels
#     n = len(profiles) 

#     # Get the model directory
#     if hasattr(model, 'config'):        # Passed state object
#         model_dir = os.path.join(model.config.io.base_dir, model.config.io.model_dir)
#     elif hasattr(model, 'io'):          # Passed config object
#         model_dir = os.path.join(model.io.base_dir, model.io.model_dir)
#     elif isinstance(model, int):        # Passed model number
#         if base_dir is None:
#             raise ValueError("'base_dir' (base directory) must be specified if using model numbers.")
#         model_dir = f"Model{model:05d}"
#         model_dir = os.path.join(base_dir, model_dir)
#     else:
#         raise TypeError(f"Unrecognized model type: {type(model)}. Must be a State object, Config object, or integer.")
    
#     # Load rhoc time evolution data
#     print(f"Getting time evolution data...")
#     time_evolution_path = os.path.join(model_dir, f"time_evolution.txt")
#     time_data = extract_time_evolution_data(time_evolution_path)
#     tevo_t = time_data['time']

#     # Validate insets
#     valid_insets = list(time_data.keys())
#     if any(inset not in valid_insets for inset in insets if inset is not None):
#         raise ValueError(f"Invalid inset specified. Valid options are: {valid_insets}")
#     if len(insets) != len(profiles):
#         raise ValueError("'insets' must have the same length as 'profiles'.")

#     # Load snapshot indices
#     snapshot_indices_data   = extract_snapshot_indices(model_dir)
#     indices                 = snapshot_indices_data['index']
#     index_t                 = snapshot_indices_data['time']

#     # Get axis limits
#     print(f"Getting axis limits...")
#     snapshot_data_list = []

#     for ind in indices:
#         snapshot_path = os.path.join(model_dir, f"profile_{ind}.dat")

#         if not os.path.isfile(snapshot_path):
#             continue

#         snapshot_data_list.append(extract_snapshot_data(snapshot_path))

#     axislims = {}

#     for i, profile in enumerate(profiles):
#         xlim, ylim = get_profile_axis_limits(profile, snapshot_data_list, xaxis=xaxis[i])
#         axislims[profile] = (xlim, ylim)

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
#         axs = np.atleast_1d(axs)

#         for i, ax in enumerate(axs):
#             profile = profiles[i]
#             inset   = insets[i]
#             xax     = xaxis[i]

#             legend = True if i == 0 else False
#             plot_profile(ax, profile, data_list, xaxis=xax, axislims=axislims[profile], legend=legend, grid=grid, for_movie=True)

#             if add_radii is not None:
#                 frac_up = 0.15
#                 for radius in add_radii:
#                     if radius in ['r01', 'r05', 'r10', 'r20', 'r50', 'r90']: # One per species
#                         for spec in time_data['species']:
#                             r = np.interp(index_t[ind], tevo_t, time_data['species'][spec][radius])
#                             if xax == 'r':
#                                 # If r is outside the x-axis limits, skip plotting
#                                 if r < axislims[profile][0][0] or r > axislims[profile][0][1]:
#                                     continue
#                                 ax.axvline(r, color='red', ls='--', zorder=-10)
#                                 trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
#                                 ax.text(r, frac_up, f"{radius}[{spec}]", rotation=90, color='red', fontsize=10, ha='right', va='bottom', zorder=-10, transform=trans)
#                             elif xax == 'm':
#                                 m = np.interp(r, 10**data_list[1]['log_r'], data_list[1]['m_tot'])
#                                 # If r is outside the x-axis limits, skip plotting
#                                 if m < axislims[profile][0][0] or m > axislims[profile][0][1]:
#                                     continue
#                                 ax.axvline(m, color='red', ls='--', zorder=-10)
#                                 trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
#                                 ax.text(m, frac_up, f"{radius}[{spec}]", rotation=90, color='red', fontsize=10, ha='right', va='bottom', zorder=-10, transform=trans)
#                     else:
#                         r = np.interp(index_t[ind], tevo_t, time_data[radius])
#                         if xax == 'r':
#                             # If r is outside the x-axis limits, skip plotting
#                             if r < axislims[profile][0][0] or r > axislims[profile][0][1]:
#                                 continue
#                             ax.axvline(r, color='red', ls='--', zorder=-10)
#                             trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
#                             ax.text(r, frac_up, radius, rotation=90, color='red', fontsize=10, ha='right', va='bottom', zorder=-10, transform=trans)
#                         elif xax == 'm':
#                             m = np.interp(r, 10**data_list[1]['log_r'], data_list[1]['m'])
#                             # If r is outside the x-axis limits, skip plotting
#                             if m < axislims[profile][0][0] or m > axislims[profile][0][1]:
#                                 continue
#                             ax.axvline(m, color='red', ls='--', zorder=-10)
#                             trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
#                             ax.text(m, frac_up, radius, rotation=90, color='red', fontsize=10, ha='right', va='bottom', zorder=-10, transform=trans)

#             if inset is not None:
#                 tevo_y = time_data[inset]
#                 if profile != 'trelax':
#                     axin = ax.inset_axes([0.55, 0.65, 0.45, 0.35])
#                 else:
#                     axin = ax.inset_axes([0.0, 0.65, 0.45, 0.35])
#                 axin.axvline(index_t[ind], color='grey')
#                 axin.plot(tevo_t, tevo_y, color='black')
#                 axin.scatter(index_t[ind], np.interp(index_t[ind], tevo_t, tevo_y),
#                             color='red', s=50)
#                 axin.set_ylabel(inset, fontsize=12)
#                 axin.set_xlabel('$t$', fontsize=12)
#                 axin.set_yscale('log')
#                 axin.tick_params(
#                     axis='both',
#                     which='both',
#                     labelbottom=False,
#                     labelleft=False,
#                     labeltop=False,
#                     labelright=False,
#                     top=True,
#                     bottom=True,
#                     left=True,
#                     right=True,
#                     direction='in'
#                 )
        
#         fig.savefig(image_path, dpi=300, bbox_inches='tight')
#         plt.close(fig)
#         image_paths.append(image_path)  # Add the image path to the list

#     print("Compiling into a movie using ffmpeg...")

#     if filepath is not None:
#         output_movie_path = filepath
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

# def make_movie_deluxe(model, parallel=True, **kwargs):
#     """
#     Top-level function for calling make_movie_deluxe,
#     either serial or parallel.
#     """
#     if parallel:
#         make_movie_deluxe_parallel(model, **kwargs)
#     else:
#         make_movie_deluxe_serial(model, **kwargs)