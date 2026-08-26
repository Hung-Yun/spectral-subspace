import os
import numpy as np

BANDS = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta': (13.0, 30.0),
    'gamma': (30.0, 70.0),
}

##########################
# == GENERAL PRINTING == #
##########################

def section(title: str):
    print("\n" + "=" * (len(title)+5))
    print(title)
    print("=" * (len(title)+5))

def announce(title):
    print("\n" + "-"  * (len(title)+5))
    print(title)
    print("-"  * (len(title)+5))


##############################
# == GENERAL PREPROCSSING == #
##############################

def apply_transform(values, transform='raw', axis=None, eps=1e-12):
    """
    Apply a simple numeric transform to an array.

    `axis` controls where z-scoring/log shifting is estimated. For example,
    use axis=1 for a features x time matrix to transform each feature over time.
    """
    x = np.asarray(values, dtype=float)

    def zscore(arr):
        mean = np.nanmean(arr, axis=axis, keepdims=True)
        std = np.nanstd(arr, axis=axis, keepdims=True)
        std = np.where(std == 0, 1.0, std)
        return (arr - mean) / std

    def log_safe(arr):
        arr_min = np.nanmin(arr, axis=axis, keepdims=True)
        shift = np.where(arr_min <= 0, -arr_min + eps, 0.0)
        return np.log(arr + shift + eps)

    if transform == 'raw':
        return x
    if transform == 'zscore':
        return zscore(x)
    if transform == 'log':
        return log_safe(x)
    if transform == 'log_zscore':
        return zscore(log_safe(x))
    raise ValueError(
        f"Unsupported transform {transform!r}. "
        "Use one of: 'raw', 'zscore', 'log', 'log_zscore'."
    )


##########################
# == PLOTTING RELATED == #
##########################


def _get_plot_modules():
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    return matplotlib, plt, sns


def fig_set(font_size=8, linewidth=0.8):
    matplotlib, _, sns = _get_plot_modules()
    sns.set(style="ticks", context="paper",
            font="sans-serif",
            rc={"font.size": font_size,
                "figure.titlesize": font_size,
                "figure.labelweight": font_size,
                "axes.titlesize": font_size,
                "axes.labelsize": font_size,
                "axes.linewidth": linewidth,
                "lines.linewidth": 1,
                "lines.markersize": 3,
                "xtick.labelsize": font_size,
                "ytick.labelsize": font_size,
                "savefig.transparent": True,
                "xtick.major.size": 2.5,
                "ytick.major.size": 2.5,
                "xtick.major.width": linewidth,
                "ytick.major.width": linewidth,
                "xtick.minor.size": 2,
                "ytick.minor.size": 2,
                "xtick.minor.width": linewidth,
                "ytick.minor.width": linewidth,
                'legend.fontsize': font_size,
                'legend.title_fontsize': font_size,
                'legend.frameon': False,
                })
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42


def finish_plot(
    filename=None,
    save_dir=None,
    savefig=True,
    show=True,
    exts=('png', 'svg', 'pdf'),
):
    _, plt, sns = _get_plot_modules()
    sns.despine(trim=False)
    plt.tight_layout()

    if savefig and save_dir is not None and filename is not None:
        os.makedirs(save_dir, exist_ok=True)

        fig = plt.gcf()
        fig_patch_visible = fig.patch.get_visible()
        ax_patch_visible = [ax.patch.get_visible() for ax in fig.axes]

        fig.patch.set_visible(False)
        for ax in fig.axes:
            ax.patch.set_visible(False)

        try:
            for ext in exts:
                path = os.path.join(save_dir, f'{filename}.{ext}')
                plt.savefig(path, bbox_inches='tight')
                print(f'Saved: {path}')
        finally:
            fig.patch.set_visible(fig_patch_visible)
            for ax, visible in zip(fig.axes, ax_patch_visible):
                ax.patch.set_visible(visible)

    if show:
        plt.show()
    else:
        plt.close()
