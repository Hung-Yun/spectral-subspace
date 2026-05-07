import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

RAWDIR_CANDIDATES = (
    "/Volumes/stitched/EMU-18112",
    "/mnt/stitched/EMU-18112",
)


def _get_repo_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _get_repo_datadir():
    return os.path.join(_get_repo_dir(), 'data')


def _get_rawdir(rawdir=None):
    candidates = []

    if rawdir is not None:
        candidates.append(rawdir)

    env_rawdir = os.environ.get('SPECTRAL_SUBSPACE_RAWDIR')
    if env_rawdir:
        candidates.append(env_rawdir)

    candidates.extend(RAWDIR_CANDIDATES)

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not find the raw NS5 data directory. Checked: "
        + ", ".join(candidates)
        + ". Mount stitched locally at /Volumes/stitched/EMU-18112, "
        + "or use /mnt/stitched/EMU-18112 on Linux/SSH, "
        + "or set SPECTRAL_SUBSPACE_RAWDIR."
    )



LW = 0.8
def fig_set(font_size=8, linewidth=LW):
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
    matplotlib.rcParams['backend'] = 'QtAgg'
