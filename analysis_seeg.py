#%%
"""
How to rerun this sEEG analysis
===============================

This file is the active scratch script for the human sEEG / Pacman analysis.
Run it cell-by-cell from the repo root after the processed data files are in
place. The main knobs to check before rerunning are:

- `session_name`: must match files in `data/neural/` and `data/behavior/`.
- `ch`: channel index or channel id/name passed into `LFPAnalyzer`.
- `SPEC_PARAMS`: frequency grid and wavelet settings for the spectrogram.
- `band`, `align_event`, and `window_s`: trial-aligned band-power summary.

Where things live:

- `processed.py`
    - `ProcessedLFP`: loads downsampled `.mat` LFP files from `data/neural/`.
    - `LFPAnalyzer`: computes per-channel PSD, spectrogram, and band power.
    - `Pacman`: loads behavioral trial files from `data/behavior/`.
    - `Comments`: loads NEV comments/events using the local `brpylib/` copy.
- `utils.py`
    - `get_data_path`: resolves neural, behavior, NEV, and temp paths.
    - `BANDS`: canonical band definitions used for band-power plots.
    - `fig_set`, `finish_plot`, `apply_transform`: plotting/preprocessing helpers.
- `spectral.py`
    - Low-level PSD, spectrogram, frequency-covariance, and autocorrelation
      helpers used by `LFPAnalyzer`.
- `decomposition.py`
    - PCA/FA helpers for spectral-subspace analyses once a power matrix is
      ready.
- `data/`
    - `data/neural/`: downsampled LFP `.mat` files and matching `.nev` files.
    - `data/behavior/`: Pacman behavioral session folders.
    - `data/temp/`: optional temporary raw/intermediate files.
- `archive/old_process_wrangell_pacman_sessions/`
    - Older Wrangell/downsampling/alignment scripts if the processed data need
      to be recreated.

Expected run order:

1. Resolve paths with `get_data_path(session_name)`.
2. Load `ProcessedLFP`, `Pacman`, and `Comments`.
3. Choose a channel and instantiate `LFPAnalyzer`.
4. Inspect band power with NEV event lines.
5. Run trial-aligned summaries or spectral-subspace analyses from the computed
   `lfp.spec_power` matrix.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from processed import ProcessedLFP, LFPAnalyzer, Pacman, Comments
from utils import fig_set, finish_plot, get_data_path, apply_transform, BANDS

SPEC_PARAMS = {
    'freqs_hz': np.arange(1, 101, dtype=float),
    'fwhm': 0.5,
    'wavelet_window_s': 2,
}

EVT_CLR = {
    'trialStart': "#EF1010",
    'itiStart': "#F38C07",
    'itiEnd': "#F4C700",
    'centralCueStart': "#99F518",
    'choiceStart': "#01F889",
    'choice2FeedbackStart': "#09BCE4",
    'feedbackStart': "#6D0DF4",
    'trialEnd': "#000000",
}

plt.figure()
plt.close()
fig_set(font_size=10, linewidth=0.8)

session_name = 'EMU-0044_subj-YFB_task-Pacman'

#%% Read data

paths = get_data_path(session_name)

print('=== READ DOWNSAMPLED LFP DATA ===')
recording = ProcessedLFP(paths['neural'])

print('=== READ BEHAVIORAL DATA ===')
behavior = Pacman(paths['behavior'], task_name=recording.task)

print('=== READ NEV DATA ===')
nev = Comments(paths['nev'])

ch = 0
print(f'=== READ LFP INFO FROM CHANNEL {ch} ===')
lfp = LFPAnalyzer(recording, channel=ch, spec_params=SPEC_PARAMS)

#%% plot band power with event lines

start, end = 10000, 70000
fig, ax = plt.subplots(5,1,figsize=(6,8), dpi=300, sharex=True)
ix = slice(start, end)
cmts = nev.df_comments[(nev.df_comments.timestamp > start) & (nev.df_comments.timestamp < end)]
for i, band in enumerate(BANDS.keys()):
    band_power = lfp.get_band_power(band)
    power_z = (band_power['power'] - np.nanmean(band_power['power'])) / np.nanstd(band_power['power'])
    ax[i].plot(band_power['time'][ix], power_z[ix])
    ax[i].set_ylabel(band)

    for row in cmts.itertuples(index=False):
        color = EVT_CLR.get(row.comment, None)
        if color is not None:
            ax[i].axvline(row.timestamp / 1000, c=color, lw=1.2, ls='--',)

plt.xlabel('Time (s)')
finish_plot()

#%% Trial-averaged theta power aligned to trial start

band = 'theta'
align_event = 'trialStart'
window_s = (-0.5, 0.5)
single_trial_zscore = False

band_power = lfp.get_band_power(band)
power = np.asarray(band_power['power'], dtype=float)
power_z = (power - np.nanmean(power)) / np.nanstd(power)
time_s = np.asarray(band_power['time'], dtype=float)

dt = np.nanmedian(np.diff(time_s))
relative_time = np.arange(window_s[0], window_s[1] + dt / 2, dt)
event_times_s = nev.df_trials[align_event].dropna().astype(float).to_numpy() / 1000

trial_power = []
for event_time_s in event_times_s:
    sample_time = event_time_s + relative_time
    if sample_time[0] < time_s[0] or sample_time[-1] > time_s[-1]:
        continue

    trial = np.interp(sample_time, time_s, power_z)
    if single_trial_zscore:
        baseline_mask = relative_time < 0
        baseline_mean = np.nanmean(trial[baseline_mask])
        baseline_std = np.nanstd(trial[baseline_mask])
        if baseline_std > 0:
            trial = (trial - baseline_mean) / baseline_std
    trial_power.append(trial)

trial_power = np.asarray(trial_power)
trial_avg = np.nanmean(trial_power, axis=0)
trial_std = np.nanstd(trial_power, axis=0)
grand_avg = np.nanmean(power_z)
grand_std = np.nanstd(power_z)

plt.figure(figsize=(3, 3), dpi=300)
plt.fill_between(
    relative_time,
    grand_avg - grand_std,
    grand_avg + grand_std,
    color='0.85',
    alpha=0.7,
    label='Grand avg +/- std',
)
plt.axhline(grand_avg, color='k', lw=1, label='Grand avg')
plt.fill_between(
    relative_time,
    trial_avg - trial_std,
    trial_avg + trial_std,
    color='#4C78A8',
    alpha=0.25,
    label='Trial avg +/- std',
)
plt.plot(relative_time, trial_avg, color='#4C78A8', lw=1.2, label='Trial avg')
plt.axvline(0, color='r', lw=1, ls='--')
plt.xlabel(f'Time from {align_event} (s)')
plt.ylabel(f'{band.capitalize()} power (z)')
plt.title(f'{band.capitalize()} power aligned to {align_event} (n={trial_power.shape[0]})')
plt.legend(frameon=False, loc='best')
finish_plot()

#%% Distribution of power
spec = lfp.spec_power
fig, ax = plt.subplots(1,1,figsize=(2,2), dpi=300)
for i in range(100):
    plt.hist(apply_transform(spec[i], transform='zscore'), bins=100, histtype='step', color=plt.cm.viridis(i/100))
plt.xlim([-5,5])
plt.xlabel('zscore power')
plt.yticks([])
plt.ylabel('Counts')
finish_plot()
# %%

# this is for illustration purpose.
# draw a line plot of two lines.
# first line is a list of 100 elements, where 13~30 elements are 1, and 0 otherwise.
# second line is more variable, with gradual increase from 5, peak at 20, plateau until 30, and gradually decrease till 37.
x = np.arange(100)

line_step = np.zeros_like(x, dtype=float)
line_step[13:31] = 1


line_theta_gamma = np.interp(
    x,
    [0, 4, 6, 8, 25, 30, 42, 58, 70, 80, 99],
    [0, 0, 1, 0, 0, 0, 0.75, 0.9, 0, 0, 0],
)
active = ((x >= 4) & (x <= 8)) | ((x >= 30) & (x <= 70))
line_theta_gamma[active] += (
    0.08 * np.sin(0.8 * x[active])
    + 0.05 * np.sin(1.9 * x[active])
)
line_theta_gamma = np.clip(line_theta_gamma, 0, None)
line_theta_gamma /= np.max(line_theta_gamma)

line_variable = np.interp(
    x,
    [0, 5, 20, 30, 45, 99],
    [0, 0, 1, 1, 0, 0],
)
active = (x >= 5) & (x <= 45)
line_variable[active] += (
    0.08 * np.sin(0.9 * x[active])
    + 0.04 * np.sin(2.2 * x[active])
)
line_variable = np.clip(line_variable, 0, None)

line_broadband = 0.92 + 0.04 * np.sin(0.23 * x) + 0.025 * np.sin(1.1 * x)
line_broadband = np.clip(line_broadband, 0, 1)

line_slanted = np.linspace(0.05, 1, len(x))
line_slanted += 0.035 * np.sin(0.3 * x) + 0.015 * np.sin(1.4 * x)
line_slanted = np.clip(line_slanted, 0, 1)

line_theta_alpha_beta_neg = np.interp(
    x,
    [0, 4, 8, 12, 13, 21, 30, 40, 99],
    [0, 0, 0.9, 0.7, 0, -0.9, 0, 0, 0],
)
active = ((x >= 4) & (x <= 12)) | ((x >= 13) & (x <= 30))
line_theta_alpha_beta_neg[active] += (
    0.06 * np.sin(0.75 * x[active])
    + 0.035 * np.sin(1.8 * x[active])
)
#%%
plt.figure(figsize=(1.5,1.5), dpi=300)
# plt.plot(x, line_step, label='Beta band')
plt.plot(x, line_slanted, c= 'k')
plt.yticks([0,1])
plt.xlabel('Frequency')
plt.ylabel('Weights')
# plt.legend(frameon=False)
finish_plot()

# %%
plt.figure(figsize=(3,3), dpi=300)
plt.plot(x, line_theta_gamma,c='k')
plt.xlabel('Frequency')
plt.ylabel('Weights')
finish_plot()

# %%


plt.figure(figsize=(3,3), dpi=300)
plt.plot(x, line_broadband, c='k', label='Broadband')
plt.plot(x, line_slanted, c='0.35', label='Slanted')
plt.plot(x, line_theta_alpha_beta_neg, c='0.65', label='Theta/alpha + beta -')
plt.axhline(0, c='0.8', lw=0.8)
plt.xlabel('Frequency')
plt.ylabel('Weights')
plt.legend(frameon=False)
finish_plot()

# %%
