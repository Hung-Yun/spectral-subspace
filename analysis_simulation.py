#%%

import numpy as np
import matplotlib.pyplot as plt
from neurodsp.spectral import compute_spectrum

from decomposition import FAResults
from spectral import get_spectrogram
from simulation import target_psd, signal_from_psd
from utils import fig_set, finish_plot, apply_transform
from neurodsp.sim import sim_combined, sim_powerlaw, sim_peak_oscillation


fig_set(font_size=10, linewidth=0.8)

fs = 500
n_seconds = 30
n_samples = fs * n_seconds
rng = np.random.default_rng(0)

#%% Simulate one baseline LFP-like signal from a known PSD

freqs_fft = np.fft.rfftfreq(n_samples, 1 / fs)
baseline_psd = target_psd(
    freqs_fft,
    exponent=-1.5,
    peak_center_hz=10,
    peak_sd_hz=2,
    peak_height=1.0,
)
baseline_sig = signal_from_psd(
    baseline_psd, 
    n_samples=n_samples, 
    rng=rng
)

components = {
    'sim_powerlaw': {'exponent': -1.5, 'f_range': (2, None)},
    'sim_oscillation': {'freq': 10},
}

baseline_sig = sim_combined(
    n_seconds=n_seconds,
    fs=fs,
    components=components,
    component_variances=[1.0, 0.5],
)

# freqs, psd = compute_spectrum(sig, fs, method='welch', nperseg=fs*2)

#%% Estimate the PSD back from the simulated signal

freqs_hz, estimated_psd = compute_spectrum(
    baseline_sig,
    fs,
    method='welch',
    f_range=(1, 100),
    nperseg=fs * 2,
)

#%% Plot the simulated signal

time_s = np.arange(n_samples) / fs

plt.figure(figsize=(5, 2), dpi=300)
plt.plot(time_s, baseline_sig, c='k', lw=0.6)
plt.xlim(0, 5)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
finish_plot()

#%% Plot target PSD and estimated PSD

target_psd_welch = np.interp(freqs_hz, freqs_fft, baseline_psd)
target_log_psd = np.log10(target_psd_welch + 1e-12)
estimated_log_psd = np.log10(estimated_psd + 1e-12)
target_log_psd = target_log_psd - target_log_psd.mean() + estimated_log_psd.mean()

plt.figure(figsize=(3, 3), dpi=300)
plt.plot(freqs_hz, target_log_psd, c='0.7', label='target')
plt.plot(freqs_hz, estimated_log_psd, c='k', label='estimated')
plt.xlim(1, 100)
# plt.xscale('log')
plt.xlabel('Frequency (Hz)')
plt.ylabel('log10 power')
plt.legend(frameon=False)
finish_plot()

#%% Compute wavelet spectrogram

spec_freqs_hz = np.arange(1, 100, dtype=float)
spec_time, spec_freqs_hz, spec_power = get_spectrogram(
    baseline_sig,
    fs=fs,
    freqs_hz=spec_freqs_hz,
    fwhm=0.5,
    wavelet_window_s=2,
)

#%% Plot wavelet spectrogram

log_spec_power = np.log10(spec_power + 1e-12)
vmin, vmax = np.nanpercentile(log_spec_power, [5, 95])

plt.figure(figsize=(5, 3), dpi=300)
plt.pcolormesh(
    spec_time,
    spec_freqs_hz,
    log_spec_power,
    shading='auto',
    cmap='viridis',
    vmin=vmin,
    vmax=vmax,
)
plt.xlim(0, n_seconds)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(label='log10 power')
finish_plot()

#%% Cross-freq cov
plt.figure(figsize=(3,2.5), dpi=300)
cross_freq_cov = np.corrcoef(spec_power)
plt.pcolormesh(spec_freqs_hz,spec_freqs_hz,cross_freq_cov,shading='auto',cmap='RdBu_r',vmin=-1,vmax=1)
plt.xlim(1, 100)
plt.ylim(1, 100)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(label='Covariance')
finish_plot()

#%% Factor analysis on spectrogram power

X = apply_transform(spec_power.T, 'log_zscore', axis=0)

fa_results = FAResults()
fa_results.fa_fit(X)
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
fa_results.fa_fit(X, n_components=5)

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
from neurodsp.utils.download import load_ndsp_data
"""
- `sample_data_1`: a segment of human primary motor cortex (M1)
    - Data sample is from a DBS device implanted in a patient with Parkinson's Disease
    - The data sample is 10 seconds from a single channel, sampled at 1000 Hz
    - For more information on the data, see Cole et al, 2017 (https://doi.org/10.1523/JNEUROSCI.2208-16.2017)
- `sample_data_2`: a segment of rat hippocampal LFP data
    - The data sample is 150 seconds from a single channel, sampled at 1000 Hz
    - Data file is from the publicly available 'hc2' dataset from CRCNS (https://crcns.org/)
    - For more information on the data, see Mizuseki et al, 2012 (https://doi.org/10.1038/nn.2894)

"""
sig_m1 = load_ndsp_data('sample_data_1.npy', folder='data')  # human M1 DBS, 10 s, 1000 Hz
sig_hpc = load_ndsp_data('sample_data_2.npy', folder='data') # rat hippocampal LFP, 150 s, 1000 Hz
# %%

freqs_hz, estimated_psd = compute_spectrum(
    sig_m1,
    1000,
    method='welch',
    f_range=(1, 100),
    nperseg=fs * 2,
)
# %%
estimated_log_psd = np.log10(estimated_psd + 1e-12)

plt.figure(figsize=(3, 3), dpi=300)
plt.plot(freqs_hz, estimated_log_psd, c='k')
plt.xlim(1, 100)
plt.xlabel('Frequency (Hz)')
plt.ylabel('log10 power')
finish_plot()
# %%
