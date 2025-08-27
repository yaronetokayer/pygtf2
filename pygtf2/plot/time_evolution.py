import os
import matplotlib.pyplot as plt
from pygtfcode.io.read import extract_time_evolution_data

def plot_time_evolution(models, quantity='rho_c', ylabel=None, logy=True, filepath=None, base_dir=None, show=False, grid=False):
    """
    Plot any time-evolution quantity vs. time for one or more simulations.

    Arguments
    ---------
    models : State object, Config object, or model_no (or list of the above)
        Each model can be a State, Config, or integer model number.
    quantity : str, optional
        Key from the time_evolution.txt file to plot on the y-axis.
        Default is 'rho_c'.
        Options are 't_phys', 'rho_c', 'rho_c_phys', 'v_max', 'v_max_phys', 'kn_min', 'mintrel', 'mintrel_phys'
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
    if type(models) != list:
        models = [models]

    def _resolve_path(model):
        if hasattr(model, 'config'): # Passed state object
            return os.path.join(model.config.io.base_dir, model.config.io.model_dir, f"time_evolution.txt")
        elif hasattr(model, 'io'): # Passed config object
            return os.path.join(model.io.base_dir, model.io.model_dir, f"time_evolution.txt")
        elif isinstance(model, int): # Passed model number
            if base_dir is None:
                raise ValueError("'base_dir' (base directory) must be specified if using model numbers.")
            model_dir = f"Model{model:03d}"
            return os.path.join(base_dir, model_dir, "time_evolution.txt")
        else:
            raise TypeError(f"Unrecognized model type: {type(model)}. Must be a State object, Config object, or integer.")

    data_list = [extract_time_evolution_data(_resolve_path(m)) for m in models]

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap('tab10')

    for i, data in enumerate(data_list):
        label = f"{data['model_id']:03d}"
        ax.plot(data['t'], data[quantity], lw=2, ls='solid', color=cmap(i % 10), label=label)

    ax.set_xlabel(r'Time [$t_\mathrm{char}$]', fontsize=16)
    ax.set_ylabel(ylabel if ylabel else quantity, fontsize=16)
    if logy:
        ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=12)
    if grid:
        ax.grid(True, which="both", ls="--")

    if filepath:
        fig.savefig(filepath, bbox_inches='tight', dpi=300)
        if show:
            plt.show()
        else:
            plt.close(fig)
    else:
        plt.show()