import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

def _get_prefix():
    if sys.platform == 'linux':
        return os.environ.get(
            'SPECTRAL_SUBSPACE_PREFIX',
            os.path.join(os.path.expanduser('~'), 'hungyun-elias', 'data'),
        )
    elif sys.platform == 'darwin': # Data are stored on my MacBook locally. 
        return os.environ.get('SPECTRAL_SUBSPACE_PREFIX', 'data')
    

def _get_datadir(remote, prefix):
    """
    The local form (remote=False) is temporary because ideally 
    we don't want to store all ns5 files locally.
    Instead the neural folder should store the preprocessed LFP matrices.
    """
    if remote:
        result = os.environ.get('SPECTRAL_SUBSPACE_DATADIR', "/Volumes/stitched/EMU-18112")
        if not os.path.exists(result):
            raise FileNotFoundError(f"Data directory not found: {result}")
    else:
        result = os.path.join(prefix, 'neural')
    return result



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