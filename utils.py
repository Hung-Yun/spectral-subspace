import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

NS5DIR_CANDIDATES = (
    "/Volumes/stitched/EMU-18112",
    "/mnt/stitched/EMU-18112",
)

BEHAVDIR_CANDIDATES = (
    "/Volumes/projectworlds/EMU-18112",
    "/mnt/projectworlds/EMU-18112",
)


def _get_repo_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _get_repo_datadir():
    return os.path.join(_get_repo_dir(), 'data')


def _get_ns5dir(ns5dir=None):
    candidates = []

    if ns5dir is not None:
        candidates.append(ns5dir)

    candidates.extend(NS5DIR_CANDIDATES)

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not find the raw NS5 data directory. Checked: "
        + ", ".join(candidates)
        + ". Mount stitched locally at /Volumes/stitched/EMU-18112, "
        + "or use /mnt/stitched/EMU-18112 on Linux/SSH."
    )

def _get_behavdir(behavdir=None):
    candidates = []

    if behavdir is not None:
        candidates.append(behavdir)

    candidates.extend(BEHAVDIR_CANDIDATES)

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not find the raw behavioral data directory. Checked: "
        + ", ".join(candidates)
        + ". Mount projectworlds locally at /Volumes/projectworlds/EMU-18112, "
        + "or use /mnt/projectworlds/EMU-18112 on Linux/SSH."
    )

def _load_behavioral_data(behavdir=None):
    """
    Loop through the ns5 dir. For each mat file, check if there is a corresponding folder in the behav dir.
    If so, make a copy to the data/behavior folder here. 
    """
    behavdir = _get_behavdir(behavdir)
    ns5dir = _get_ns5dir()

    for filename in os.listdir(ns5dir):

    if not os.path.exists(behav_data_path):
        raise FileNotFoundError(f"Could not find behavioral data at {behav_data_path}.")
    return pd.read_csv(behav_data_path)



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
