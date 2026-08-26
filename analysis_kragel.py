"""Scratch analysis notebook for the Kragel et al. (2021) dataset.

Run this file one ``# %%`` cell at a time. Add exploratory analyses below the
shared imports and dataset paths.
"""

# %% Imports and paths
from pathlib import Path
import gc

import h5py
import numpy as np
import pandas as pd
from scipy import io, signal
import matplotlib.pyplot as plt
from utils import fig_set, finish_plot
from spectral import get_spectrogram, get_psd

REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = REPO_ROOT / "data" / "kragel"
EEG_DIR = DATA_ROOT / "eeg"
EYE_DIR = DATA_ROOT / "eye"
BEHAV_DIR = DATA_ROOT / "behav"
LOCALIZATION_DIR = DATA_ROOT / "localization"

ARCHIVE_ROOT = REPO_ROOT / "archive" / "kragel"
AUTHOR_CODE_ROOT = ARCHIVE_ROOT / "code"
PAPER_PATH = ARCHIVE_ROOT / "paper.pdf"


def section(title):
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92)


def warn(title):
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)


fig_set(font_size=8, linewidth=0.8)


# %% Start exploratory analysis here

