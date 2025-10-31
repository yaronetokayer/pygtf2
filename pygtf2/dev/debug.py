import matplotlib.pyplot as plt

def plot_r_markers(r_slice):
    """
    r_slice : array of shape (s, m)
        Radii for each species (s species, m points each).
    """
    s, m = r_slice.shape
    fig, axes = plt.subplots(s, 1, figsize=(10, 0.75*s), sharex=True)

    if s == 1:
        axes = [axes]

    for k, ax in enumerate(axes):
        ax.set_xscale("log")
        for j, rj in enumerate(r_slice[k]):
            ax.axvline(rj, color="k", lw=1)
            ax.text(rj, -0.05, str(j), ha="center", va="top",
                    transform=ax.get_xaxis_transform(), fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_xlim(3e-3, 1e-1)
        ax.set_yticks([])
        ax.set_ylabel(f"species {k+1}", rotation=0, labelpad=25, va="center")

    axes[-1].set_xlabel("r (log scale)")
    plt.tight_layout()
    plt.show()