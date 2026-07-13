#%%
"""
Read the CRCNS hc-2 `ec013.527.eeg` LFP file.

This script reads the companion `.xml` file for session metadata. From the
CRCNS hc-2 data description:

- `.eeg` is the downsampled LFP file.
- Samples are stored as signed 16-bit integers.
- Data are multiplexed by sample:
  ch1_sample1, ch2_sample1, ..., chN_sample1, ch1_sample2, ...
- The XML stores `nChannels`, raw acquisition `samplingRate`, LFP
  `lfpSamplingRate`, and anatomical channel groups.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib.pyplot as plt
from neurodsp.spectral import compute_spectrum

from spectral import get_spectrogram
from decomposition import FAResults
from utils import fig_set, finish_plot, apply_transform


fig_set(font_size=10, linewidth=0.8)

session_name = 'ec013.527'
eeg_path = f'data/hc-2/{session_name}.eeg'
xml_path = f'data/hc-2/{session_name}.xml'
dtype = np.int16

#%% Read session metadata from XML

xml_root = ET.parse(xml_path).getroot()
n_channels = int(xml_root.findtext('acquisitionSystem/nChannels'))
raw_fs = int(xml_root.findtext('acquisitionSystem/samplingRate'))
fs = int(xml_root.findtext('fieldPotentials/lfpSamplingRate'))

channel_groups = []
for group in xml_root.findall('anatomicalDescription/channelGroups/group'):
    group_channels = [
        int(channel.text)
        for channel in group.findall('channel')
        if channel.attrib.get('skip', '0') == '0'
    ]
    channel_groups.append(group_channels)

print(f'Session: {session_name}')
print(f'Raw sampling rate: {raw_fs} Hz')
print(f'LFP sampling rate: {fs} Hz')
print(f'Channels: {n_channels}')
print(f'Channel groups: {channel_groups}')

#%% Memory-map the binary file as samples x channels

n_values = os.path.getsize(eeg_path) // np.dtype(dtype).itemsize
if n_values % n_channels != 0:
    raise ValueError(f'{eeg_path} size is not divisible by {n_channels} channels.')

n_samples = n_values // n_channels
duration_s = n_samples / fs

eeg = np.memmap(eeg_path, dtype=dtype, mode='r', shape=(n_samples, n_channels))

print(f'Loaded {eeg_path}')
print(f'Shape: {eeg.shape[0]:,} samples x {eeg.shape[1]} channels')
print(f'Sampling rate: {fs} Hz')
print(f'Duration: {duration_s:.1f} s ({duration_s / 60:.1f} min)')

# %% Show similarity across channels
eg = np.asarray(eeg[:10000], dtype=float)
corrs = np.corrcoef(eg, rowvar=False)
plt.figure(figsize=(3,2.5), dpi=300)
plt.pcolormesh(corrs, cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar(label='Correlation coefficient')
plt.xticks(np.arange(0, n_channels, 5))
plt.yticks(np.arange(0, n_channels, 5))
plt.xlabel('Channel')
plt.ylabel('Channel')
finish_plot()
#%% Plot a short trace preview

channel = 0
start_s = 0
window_s = 5
start = int(start_s * fs)
stop = int((start_s + window_s) * fs)

trace = np.asarray(eeg[start:stop, channel], dtype=float)
time_s = np.arange(start, stop) / fs

plt.figure(figsize=(3.5, 2), dpi=300)
plt.plot(time_s, trace, c='k')
plt.xlabel('Time (s)')
plt.ylabel('LFP amplitude')
finish_plot()

#%% Estimate PSD for one channel

psd_window_s = 120
psd_start = 0
psd_stop = int(psd_window_s * fs)
psd_trace = np.asarray(eeg[psd_start:psd_stop, channel], dtype=float)

freqs_hz, psd = compute_spectrum(
    psd_trace,
    fs,
    method='welch',
    f_range=(1, 30),
    nperseg=fs * 2,
)

plt.figure(figsize=(2,2), dpi=300)
plt.plot(freqs_hz, apply_transform(psd, 'log'), c='k')
plt.xlabel('Frequency (Hz)')
plt.ylabel('log power')
finish_plot()

#%% Compute a short wavelet spectrogram

spec_window_s = 60
spec_start = 0
spec_stop = int(spec_window_s * fs)
spec_trace = np.asarray(eeg[spec_start:spec_stop, channel], dtype=float)
spec_trace -= np.mean(spec_trace)
spec_trace /= np.std(spec_trace)

top_freq = 100.5

spec_freqs_hz = np.arange(1, top_freq, 0.5, dtype=float)
spec_time, spec_freqs_hz, spec_power = get_spectrogram(spec_trace,fs=fs,freqs_hz=spec_freqs_hz,fwhm=0.5,wavelet_window_s=2,)
#%%
plt.figure(figsize=(2.5,2), dpi=300)
cross_freq_cov = np.corrcoef(spec_power)
plt.pcolormesh(spec_freqs_hz,spec_freqs_hz,cross_freq_cov,shading='auto',cmap='RdBu_r',vmin=-1,vmax=1)
plt.xticks(np.arange(0, top_freq, top_freq//5))
plt.yticks(np.arange(0, top_freq, top_freq//5))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(label='Correlation coefficient')
finish_plot()

#%% Plot the spectrogram preview

log_spec_power = np.log10(spec_power + 1e-12)
vmin, vmax = np.nanpercentile(log_spec_power, [5, 95])

plt.figure(figsize=(5, 3), dpi=300)
plt.pcolormesh(spec_time,spec_freqs_hz,log_spec_power,shading='auto',cmap='viridis',vmin=vmin,vmax=vmax,)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title(f'Spectrogram channel {channel}')
plt.colorbar(label='log10 power')
finish_plot()

# %%

X = apply_transform(spec_power.T, 'log_zscore', axis=0)

fa_results = FAResults()
fa_results.fa_fit(X, n_components=5)
#%%
cumsum_var = np.cumsum(fa_results.explained_variance_ratio_)
plt.figure(figsize=(3, 3), dpi=300)
plt.plot(cumsum_var, marker='o', c='k')
plt.axvline(np.argmax(cumsum_var > 0.9), c='k', ls='--', lw=0.8, label=f'd shared = {np.argmax(cumsum_var > 0.9):d}')
plt.legend(frameon=False, loc='lower right', fontsize=8)
plt.xlabel('Number of factors')
plt.ylabel('Cumulative variance explained')
finish_plot()

#%% Plot FA frequency loadings

fig, ax = plt.subplots(2,3,figsize=(5,4), dpi=300, sharex=True, sharey=True)
ax = ax.flatten()
for j in range(5):
    ax[j].plot(spec_freqs_hz, fa_results.subspace[j], lw=0.8)
ax[4].set_xlabel('Frequency (Hz)')
finish_plot()

# %%

plt.figure(figsize=(3,3), dpi=300)
plt.plot(fa_results.shared_var_per_unit, c='k')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Shared Variance per frequency')
finish_plot()
# %%

psd_window_s = 120
psd_start = 0
psd_stop = int(psd_window_s * fs)
psd_trace = np.asarray(eeg[psd_start:psd_stop, channel], dtype=float)

freqs_hz, psd = compute_spectrum(
    psd_trace,
    fs,
    method='welch',
    f_range=(1, 100),
    nperseg=fs * 2,
)
agg = spec_power.sum(1)
psd = psd - psd.mean() + agg.mean() 

plt.figure(figsize=(2,2), dpi=300)
plt.plot(spec_freqs_hz, apply_transform(agg, 'log'), c='r')
plt.plot(freqs_hz, apply_transform(psd, 'log'), c='k')
plt.xlabel('Frequency (Hz)')
plt.ylabel('log power')
finish_plot()

# %%
