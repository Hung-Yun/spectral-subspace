#%%
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import io
from processed import ProcessedLFP
from utils import fig_set, finish_plot
from decomposition import FAResults

plt.figure()
plt.close()
fig_set(font_size=10, linewidth=0.8)

#%%

DATA_PATH = 'data/simulation/sim_pinknoise_alpha-1_fs-500_dur-120_nseeds-15.mat'
mat = io.loadmat(DATA_PATH, squeeze_me=True, struct_as_record=False)
data = {key: value for key, value in mat.items() if not key.startswith('__')}
recording = ProcessedLFP(data)

#%%

recording.plot_psd(window_s=1.0, overlap_frac=0.5, max_hz=100)

recording.compute_spectrogram(
    channel=0,
    start_s=0,
    duration_s=350,
    freqs_hz=np.arange(1, 101, dtype=float),
    fwhm=0.5,
    wavelet_window_s=1.0,
)

display_step = max(int(round(recording.fs / 100)), 1)
plt.figure(figsize=(6, 3), dpi=300)
plt.pcolormesh(
    recording.spec_time[::display_step],
    recording.spec_freqs_hz,
    np.log(recording.spec_power[:, ::display_step] + 1e-12),
    cmap='viridis',
    shading='auto',
    # vmin=-10,
    # vmax=10,
)
plt.colorbar(label='Log power')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title(f'Spectrogram: {recording.channel_names[recording.spec_channel_idx]}')
finish_plot()

#%% Plot correlation matrix (freq_corr)
recording.compute_freq_corr()

plt.figure(figsize=(3,3), dpi=300)
plt.imshow(recording.freq_corr, aspect='auto', origin='lower', cmap='coolwarm')
plt.colorbar(label='Correlation')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Frequency (Hz)')
plt.title('Correlation matrix of spectral power across frequencies')
finish_plot()

#%% Dimensionality reduction with FA

fa_results = FAResults()
fa_results.fa_fit(
    X=recording.spec_power.T, # (time x freqs)
    shared_var_thresh=0.95,
    max_iter=int(1e6),
    tol=1e-6,
    verbose=True,
)
#%%

plt.figure(figsize=(3,3), dpi=300)
plt.plot(fa_results.explained_variance_ratio_, marker='o')
plt.xlabel('Number of components')
plt.ylabel('Proportion of shared variance explained')
plt.title('FA shared variance explained')
finish_plot()
# %%

plt.figure(figsize=(3,3), dpi=300)
plt.plot(fa_results.shared_var_per_unit, marker='o')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Proportion of shared variance')
plt.title('FA shared variance per frequency')
finish_plot()


