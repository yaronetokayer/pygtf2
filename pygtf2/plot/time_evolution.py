import os
import matplotlib.pyplot as plt
from itertools import cycle
from pygtf2.io.read import extract_time_evolution_data

def plot_time_evolution(models, quantity='rho_c', ylabel=None, logy=True, filepath=None, base_dir=None, show=False, grid=False):
    """
    Plot any time-evolution quantity vs. time for one or more simulations.

    Arguments
    ---------
    models : State | Config | int | list
        State or Config objects, or integer model numbers.
        If quantity is 'r_enc', only one model can be passed.
    quantity : {'rho_c','v2_c', 'r_c', 'eta_c','mintrel', 'r_enc'}, optional
        What to plot on y-axis. If 'rho_c', also plots per-species curves.  Default is rho_c
    ylabel : str, optional
        Custom y-axis label. Defaults to quantity.
    logy : bool, optional
        Use logarithmic scale on y-axis. Default is True.
    filepath : str, optional
        If specified, saves the figure to this path.
    base_dir : str, optional
        Required if any model is passed as an integer.  The directory in which all ModelXXX subdirectories reside.
    show : bool, optional
        If True, show the plot even if saving.  Default is False.
    grid : bool, optional
        If True, shows grid on axis
    """
    if not isinstance(models, list):
        models = [models]

    def _resolve_path(model):
        if hasattr(model, 'config'): # Passed state object
            return os.path.join(model.config.io.base_dir, model.config.io.model_dir, f"time_evolution.txt")
        elif hasattr(model, 'io'): # Passed config object
            return os.path.join(model.io.base_dir, model.io.model_dir, f"time_evolution.txt")
        elif isinstance(model, int): # Passed model number
            if base_dir is None:
                raise ValueError("'base_dir' must be specified when passing model numbers.")
            model_dir = f"Model{model:03d}"
            return os.path.join(base_dir, model_dir, "time_evolution.txt")
        else:
            raise TypeError(f"Unrecognized model type: {type(model)}. Must be a State object, Config object, or integer.")

    data_list = [extract_time_evolution_data(_resolve_path(m)) for m in models]

    if quantity not in {'rho_c', 'v2_c', 'r_c', 'eta_c', 'mintrel', 'r_enc'}:
        raise ValueError("quantity must be one of {'rho_c', 'v2_c', 'r_c', 'eta_c', 'mintrel', 'r_enc'}")
    
    # Special handling for r_enc: require exactly one model
    if quantity == 'r_enc' and len(models) != 1:
        raise ValueError("quantity='r_enc' expects exactly one model.")

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap('tab10')

    # Add equipartition line for eta_c
    if quantity == 'eta_c':
        ax.axhline(0.5, ls='--', color='grey')
        ax.text(x=0.1, y=0.49, s='equipartition', horizontalalignment='left', verticalalignment='top', color='grey', fontsize=14)

    if quantity in {'rho_c', 'v2_c', 'r_c', 'eta_c','mintrel'}:
        for i, data in enumerate(data_list):
            color = cmap(i % 10)
            model_label = f"{data.get('model_id', i):03d}"
            t = data['time']

            if quantity == 'rho_c':
                # total
                ax.plot(t, data['rho_c_tot'], lw=2, color=color, ls='solid', label=f"{model_label} total")

                # per-species (unique linestyle per species for this model)
                # reset style cycle for each model so species styles are consistent per-model
                # styles = cycle(['dashed', 'dashdot', 'dotted', (0, (1, 1)), (0, (3, 1, 1, 1))])
                styles = cycle([
                    (0, (5, 2)),              # long dash
                    (0, (1, 1)),              # fine dotted
                    (0, (3, 1, 1, 1)),        # dash-dot pattern
                    (0, (7, 3, 1, 3)),        # long dash, small dot, medium gap
                    (0, (5, 1)),              # medium dash, tight gaps
                    (0, (3, 5, 1, 5, 1, 5)),  # mixed dash/dot combo
                ])
                # iterate species in sorted name order for stable legends
                for sp_name in sorted(data['species'].keys()):
                    rho_c_k = data['species'][sp_name]['rho_c']
                    ax.plot(t, rho_c_k, lw=2, color=color, ls=next(styles), label=f"{model_label} {sp_name}")

            else:
                ax.plot(t, data[quantity], lw=2, ls='solid', color=color, label=model_label)

        ax.set_ylabel(ylabel if ylabel else quantity, fontsize=16)

    elif quantity == 'r_enc':
        data = data_list[0]
        t = data['time']

        color = cmap(0)

        # consistent linestyle per species
        # (map by species name so the same species always looks the same)
        # base_styles = ['dashed', 'dashdot', 'dotted', (0, (1, 1)), (0, (3, 1, 1, 1))]
        base_styles = [
            (0, (5, 2)),              # long dash
            (0, (1, 1)),              # fine dotted
            (0, (3, 1, 1, 1)),        # dash-dot pattern
            (0, (7, 3, 1, 3)),        # long dash, small dot, medium gap
            (0, (5, 1)),              # medium dash, tight gaps
            (0, (3, 5, 1, 5, 1, 5)),  # mixed dash/dot combo
        ]
        species_names = sorted(data['species'].keys())
        style_map = {name: base_styles[i % len(base_styles)] for i, name in enumerate(species_names)}

        keys = ['r01', 'r05', 'r10', 'r20', 'r50', 'r90']

        # build legend entries once per species
        for sp_name in species_names:
            sp = data['species'][sp_name]

            # plot all percentiles with the same linestyle & color
            ls = style_map[sp_name]
            for k in keys:
                y = sp[k]
                ax.plot(t, y, lw=1.8, color=color, ls=ls)

            # add one invisible handle with desired linestyle for legend
            ax.plot([], [], lw=2.2, color=color, ls=ls, label=sp_name)

        ax.set_ylabel(ylabel if ylabel else r"$r_\mathrm{enc}$", fontsize=14)

    # Cosmetics
    ax.set_xlabel(r'Time [$t_\mathrm{char}$]', fontsize=16)
    if logy:
        ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=12)
    if grid:
        ax.grid(True, which="both", ls="--")
    ax.legend(fontsize=10, frameon=False, ncol=1)

    if filepath:
        fig.savefig(filepath, bbox_inches='tight', dpi=300)
        if show:
            plt.show()
        else:
            plt.close(fig)
    else:
        plt.show()