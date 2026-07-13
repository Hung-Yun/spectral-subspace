"""Archived exploratory simulation workflow.

This script preserves the older notebook-style simulation pipeline that mixes
general plotting, dimensionality reduction, reconstruction diagnostics, the
smooth-window sensitivity test, and the X-pattern investigation.
"""

#%%
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# import fa
from archive.old_handrolled_simulation.simulate_core import generate_oscillation_trace
from spectral import get_autocorr, get_power_cov, get_psd, get_spectrogram
from utils import fig_set, finish_plot


class FactorAnalysisResult:
    def __init__(self, fit_result: dict):
        for key, value in fit_result.items():
            setattr(self, key, value)

    def __repr__(self):
        return f'{self.subspace.shape[0]} components from {self.subspace.shape[1]} channels'


plt.figure()
plt.close()
fig_set(font_size=10, linewidth=0.8)


class SimulationResult:
    def __init__(self, envelope_mode, sim_kwargs, psd_kwargs, spectrogram_kwargs):
        self.envelope_mode = envelope_mode
        self.trace, self.envelopes = generate_oscillation_trace(
            **sim_kwargs,
            envelope_mode=envelope_mode,
        )
        self.psd_f, self.psd = get_psd(self.trace, **psd_kwargs)
        self.spec_time, self.spec_freqs_hz, self.spec_power = get_spectrogram(
            self.trace,
            **spectrogram_kwargs,
        )
        self.truth_power = build_ground_truth_spectrogram(
            spec_freqs_hz=self.spec_freqs_hz,
            active_freqs_hz=sim_kwargs['freqs_hz'],
            envelopes=self.envelopes,
        )
        self.spec_power_demean = self.spec_power - self.spec_power.mean(axis=1, keepdims=True)
        self.z_power, self.freq_corr = get_power_cov(self.spec_power)
        self.truth_z_power, self.truth_freq_corr = get_power_cov(self.truth_power)
        self.demean_power, self.demean_freq_corr = get_power_cov(
            self.spec_power_demean,
            z_scored=False,
        )

    def PCA(self):
        self.pca = PCA().fit(self.spec_power.T)
        self.pca_truth = PCA().fit(self.truth_power.T)
        self.pca_demean = PCA().fit(self.spec_power_demean.T)
        self.z_pca = PCA().fit(self.z_power.T)

    def FA(self, shared_var_thresh=0.95):
        self.fa = FactorAnalysisResult(
            fa.fa_fit(self.spec_power.T, shared_var_thresh=shared_var_thresh)
        )
        self.z_fa = FactorAnalysisResult(
            fa.fa_fit(self.z_power.T, shared_var_thresh=shared_var_thresh)
        )

    def __repr__(self):
        return f'{self.envelope_mode} envelope'


def build_ground_truth_spectrogram(spec_freqs_hz, active_freqs_hz, envelopes):
    spec_freqs_hz = np.asarray(spec_freqs_hz, dtype=float)
    active_freqs_hz = np.asarray(active_freqs_hz, dtype=float)
    envelopes = np.asarray(envelopes, dtype=float)

    if active_freqs_hz.shape[0] != envelopes.shape[0]:
        raise ValueError('active_freqs_hz and envelopes must have the same number of rows.')

    power_by_freq_time = np.zeros((spec_freqs_hz.shape[0], envelopes.shape[1]), dtype=float)
    for freq_idx, freq_hz in enumerate(active_freqs_hz):
        match_idx = np.where(np.isclose(spec_freqs_hz, freq_hz))[0]
        if match_idx.size != 1:
            raise ValueError(f'Expected exactly one matching spectrogram bin for {freq_hz} Hz.')
        power_by_freq_time[match_idx[0]] = envelopes[freq_idx] ** 2

    return power_by_freq_time


def get_wavelet_frequency_responses(fs, freqs_hz, fwhm, wavelet_window_s, response='power', n_fft=None):
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    wavetime = np.arange(-wavelet_window_s, wavelet_window_s, 1 / fs)
    gaussian = np.exp(-(4 * np.log(2) * wavetime**2) / fwhm**2)
    if n_fft is None:
        n_fft = 2 ** int(np.ceil(np.log2(wavetime.size * 16)))

    fft_freqs = np.fft.fftfreq(n_fft, d=1 / fs)
    keep = fft_freqs >= 0
    response_by_wavelet = np.zeros((freqs_hz.size, np.sum(keep)), dtype=float)

    for freq_idx, freq_hz in enumerate(freqs_hz):
        wavelet = np.exp(1j * 2 * np.pi * freq_hz * wavetime) * gaussian
        wavelet_fft = np.abs(np.fft.fft(wavelet, n_fft))[keep]
        wavelet_fft /= wavelet_fft.max()
        if response == 'power':
            wavelet_fft = wavelet_fft ** 2
        response_by_wavelet[freq_idx] = wavelet_fft

    return fft_freqs[keep], response_by_wavelet


def max_off_center_response(response_freqs_hz, response_by_wavelet, center_freqs_hz):
    overlap = np.zeros((center_freqs_hz.size, center_freqs_hz.size), dtype=float)
    for wavelet_idx in range(center_freqs_hz.size):
        overlap[wavelet_idx] = np.interp(
            center_freqs_hz,
            response_freqs_hz,
            response_by_wavelet[wavelet_idx],
        )
    np.fill_diagonal(overlap, 0)
    return overlap.max()


def plot_wavelet_frequency_responses(freqs_hz, fs, fwhm, wavelet_window_s, max_hz=5, response='power'):
    response_freqs_hz, response_by_wavelet = get_wavelet_frequency_responses(
        fs=fs,
        freqs_hz=freqs_hz,
        fwhm=fwhm,
        wavelet_window_s=wavelet_window_s,
        response=response,
    )
    max_overlap = max_off_center_response(
        response_freqs_hz,
        response_by_wavelet,
        np.asarray(freqs_hz, dtype=float),
    )
    keep = response_freqs_hz <= max_hz

    plt.figure(figsize=(4.5, 3), dpi=300)
    for response_idx in range(response_by_wavelet.shape[0]):
        plt.plot(response_freqs_hz[keep], response_by_wavelet[response_idx, keep], lw=1)
    plt.axhline(0.01, color='k', ls='--', lw=1, alpha=0.5, label='1% response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel(f'Normalized {response} response')
    plt.title(f'FWHM={fwhm:g}s, window=+/-{wavelet_window_s:g}s, max overlap={max_overlap:.3g}')
    plt.legend(frameon=False, loc='upper right', fontsize=7)
    finish_plot()
    return response_freqs_hz, response_by_wavelet, max_overlap


def relative_error(estimate, target):
    return np.linalg.norm(estimate - target) / np.linalg.norm(target)


def participation_ratio(values):
    values = np.asarray(values, dtype=float)
    values = values[values > 0]
    if values.size == 0:
        return np.nan
    return values.sum() ** 2 / np.sum(values ** 2)


def summarize_simulation(sim, n_summary_freqs=3, acf_max_lag_s=60):
    log_power = np.log(sim.spec_power + 1e-12)
    fs = 1 / np.median(np.diff(sim.spec_time))
    summary_freq_idx = np.argsort(log_power.mean(axis=1))[-n_summary_freqs:]
    summary_freq_idx = np.sort(summary_freq_idx)
    summary_freqs_hz = sim.spec_freqs_hz[summary_freq_idx]

    acf_max_lag_samples = min(int(round(acf_max_lag_s * fs)), log_power.shape[1] - 1)
    acf_by_freq = np.vstack(
        [get_autocorr(log_power[freq_idx], acf_max_lag_samples) for freq_idx in summary_freq_idx]
    )
    mean_acf = acf_by_freq.mean(axis=0)
    acf_lags_s = np.arange(mean_acf.shape[0]) / fs
    decay_idx = np.where(mean_acf < np.exp(-1))[0]
    acf_decay_s = acf_lags_s[decay_idx[0]] if decay_idx.size else np.nan

    pca_cumulative = np.cumsum(sim.pca.explained_variance_ratio_)
    freq_corr_upper = sim.freq_corr[np.triu_indices_from(sim.freq_corr, k=1)]

    return dict(
        summary_freq_idx=summary_freq_idx,
        summary_freqs_hz=summary_freqs_hz,
        acf_lags_s=acf_lags_s,
        mean_acf=mean_acf,
        acf_decay_s=acf_decay_s,
        pc1_var_explained=sim.pca.explained_variance_ratio_[0],
        n_pcs_95=np.searchsorted(pca_cumulative, 0.95) + 1,
        effective_rank=participation_ratio(sim.pca.explained_variance_),
        freq_corr_upper=freq_corr_upper,
    )


def mean_and_sem(values):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if not np.any(valid):
        return np.nan, np.nan
    mean = np.nanmean(values)
    if np.sum(valid) == 1:
        return mean, 0.0
    sem = np.nanstd(values, ddof=1) / np.sqrt(np.sum(valid))
    return mean, sem


def matrix_similarity(a, b):
    iu = np.triu_indices_from(a, k=1)
    a_flat = a[iu]
    b_flat = b[iu]
    valid = np.isfinite(a_flat) & np.isfinite(b_flat)
    if np.sum(valid) < 2:
        return np.nan
    return np.corrcoef(a_flat[valid], b_flat[valid])[0, 1]


def pca_rank_reconstruction(pca, features_by_time, rank):
    rank = min(rank, pca.components_.shape[0])
    scores = pca.transform(features_by_time)
    return scores[:, :rank] @ pca.components_[:rank, :] + pca.mean_


#%% General simulation parameters
SIM_FS = 500
SIM_DURATION_S = 150
SIM_TIME = np.arange(0, SIM_DURATION_S, 1 / SIM_FS)
SIM_FREQS_HZ = np.array([12, 30, 70], dtype=float)
SMOOTH_WINDOW_S = 0.5

PSD_WINDOW_S = 1.0
PSD_OVERLAP_FRAC = 0.5

FWHM = 0.5
WAVELET_WINDOW_S = 1.0
SPEC_FREQS_HZ = np.arange(1, 101, dtype=float)
STP = max(int(round(SIM_FS / 200)), 1)

sim_kwargs = dict(
    sim_time=SIM_TIME,
    rng=np.random.default_rng(0),
    freqs_hz=SIM_FREQS_HZ,
    smooth_window_s=SMOOTH_WINDOW_S,
    base_amplitudes=np.array([30, 28, 25], dtype=float),
    phases_rad=np.array([0.0, 0.6, 1.1], dtype=float),
    envelope_scales=np.array([12, 8, 4], dtype=float),
)
psd_kwargs = dict(fs=SIM_FS, window_s=PSD_WINDOW_S, overlap_frac=PSD_OVERLAP_FRAC)
spectrogram_kwargs = dict(
    fs=SIM_FS,
    freqs_hz=SPEC_FREQS_HZ,
    fwhm=FWHM,
    wavelet_window_s=WAVELET_WINDOW_S,
)

ind_noise = SimulationResult('ind', sim_kwargs, psd_kwargs, spectrogram_kwargs)
shared_noise = SimulationResult('shared', sim_kwargs, psd_kwargs, spectrogram_kwargs)
simulations = [ind_noise, shared_noise]

for sim in simulations:
    sim.PCA()
    # sim.FA()


#%% Envelopes
for sim in simulations:
    plt.figure(figsize=(3, 3), dpi=300)
    for freq_idx, freq_hz in enumerate(SIM_FREQS_HZ):
        plt.plot(
            SIM_TIME[1000:3000],
            sim.envelopes[freq_idx, 1000:3000],
            lw=1,
            label=f'{freq_hz:.0f} Hz',
        )
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(sim)
    finish_plot()


#%% PSDs
for sim in simulations:
    mask = sim.psd_f <= 100
    plt.figure(figsize=(3, 3), dpi=300)
    plt.plot(sim.psd_f[mask], sim.psd[mask], lw=1)
    plt.yscale('log')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD')
    plt.title(f'{sim} PSD')
    finish_plot()


#%% Spectrograms
for sim in simulations:
    ix = np.max(np.where(sim.spec_freqs_hz <= 100))
    plt.figure(figsize=(3, 3), dpi=300)
    plt.pcolormesh(
        sim.spec_time[::STP],
        sim.spec_freqs_hz[:ix],
        np.log(sim.spec_power[:ix, ::STP]),
        cmap='coolwarm',
        vmin=-20,
        vmax=10,
    )
    plt.colorbar()
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(f'{sim} log power')
    finish_plot()


#%% Ground truth spectrograms
for sim in simulations:
    ix = np.max(np.where(sim.spec_freqs_hz <= 100))
    plt.figure(figsize=(3, 3), dpi=300)
    plt.pcolormesh(
        sim.spec_time[::STP],
        sim.spec_freqs_hz[:ix],
        sim.truth_power[:ix, ::STP],
        cmap='Blues',
    )
    plt.colorbar()
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(f'{sim} truth')
    finish_plot()


#%% Frequency correlation matrices
for sim in simulations:
    ix = np.max(np.where(sim.spec_freqs_hz <= 100))
    plt.figure(figsize=(3, 3), dpi=300)
    plt.imshow(
        sim.demean_freq_corr[:ix, :ix],
        aspect='auto',
        origin='lower',
        extent=[sim.spec_freqs_hz[0], sim.spec_freqs_hz[ix], sim.spec_freqs_hz[0], sim.spec_freqs_hz[ix]],
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
    )
    plt.colorbar()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Frequency (Hz)')
    plt.title(sim)
    finish_plot()


#%% Ground truth frequency correlation matrices
for sim in simulations:
    ix = np.max(np.where(sim.spec_freqs_hz <= 100))
    plt.figure(figsize=(3, 3), dpi=300)
    plt.imshow(
        sim.truth_freq_corr[:ix, :ix],
        aspect='auto',
        origin='lower',
        extent=[sim.spec_freqs_hz[0], sim.spec_freqs_hz[ix], sim.spec_freqs_hz[0], sim.spec_freqs_hz[ix]],
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
    )
    plt.colorbar()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Frequency (Hz)')
    plt.title(sim)
    finish_plot()


#%% PCA: dimensionality
plt.figure(figsize=(3, 3), dpi=300)
for sim in simulations:
    dims = np.where(np.cumsum(sim.pca.explained_variance_ratio_) >= 0.99)[0][0] + 1
    plt.plot(np.cumsum(sim.pca.explained_variance_ratio_), lw=1, label=f'{sim} ({dims} PCs)')
    dims = np.where(np.cumsum(sim.pca_truth.explained_variance_ratio_) >= 0.99)[0][0] + 1
    plt.plot(
        np.cumsum(sim.pca_truth.explained_variance_ratio_),
        lw=1,
        ls='--',
        label=f'{sim} truth ({dims} PCs)',
    )
plt.xlabel('Number of PCs')
plt.ylabel('Cumulative Explained Variance')
plt.legend(fontsize=6, frameon=False, loc='lower right')
plt.xlim([-0.5, 10])
finish_plot()


#%% Plot the first FA component loadings
for sim in simulations:
    plt.figure(figsize=(3, 3), dpi=300)
    plt.plot(sim.fa.subspace[0])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Loading')
    plt.title(f'{sim} FA first component')
    finish_plot()


#%% Projection to the latent subspace
ix = 0
sim = shared_noise
pca_proj = sim.pca_demean.transform(sim.spec_power_demean.T)[:, ix]
fa_proj, _ = fa.fa_transform(sim.z_power.T, sim.z_fa.fa)
fa_proj = fa_proj[:, ix]
envelope = sim.envelopes[0]

pca_proj = (pca_proj - pca_proj.mean()) / pca_proj.std()
fa_proj = (fa_proj - fa_proj.mean()) / fa_proj.std()
envelope = (envelope - envelope.mean()) / envelope.std()

plt.figure(figsize=(3.6, 3), dpi=300)
plt.plot(SIM_TIME, envelope, lw=1, label='True shared envelope')
plt.plot(SIM_TIME, pca_proj, lw=1, label='PC1 score')
plt.plot(SIM_TIME, fa_proj, lw=1, label='FA1 score')
plt.xlabel('Time (s)')
plt.ylabel('Normalized amplitude')
plt.title('Shared envelope vs recovered latent signal')
plt.legend(frameon=False, loc='best', fontsize=7)
finish_plot()


#%% Smooth-window sensitivity
SWEEP_SMOOTH_WINDOWS_S = np.array([0.5, 5, 50], dtype=float)
SWEEP_SEEDS = np.arange(3)
SWEEP_ENVELOPE_MODES = ['ind', 'shared']

sweep_simulations = {
    envelope_mode: {smooth_window_s: [] for smooth_window_s in SWEEP_SMOOTH_WINDOWS_S}
    for envelope_mode in SWEEP_ENVELOPE_MODES
}

for smooth_window_s in SWEEP_SMOOTH_WINDOWS_S:
    print(f'=== Smooth window: {smooth_window_s} s ===')
    for seed in SWEEP_SEEDS:
        print(seed)
        sim_kwargs_sweep = {
            **sim_kwargs,
            'smooth_window_s': smooth_window_s,
            'rng': np.random.default_rng(seed),
        }
        for envelope_mode in SWEEP_ENVELOPE_MODES:
            sim = SimulationResult(envelope_mode, sim_kwargs_sweep, psd_kwargs, spectrogram_kwargs)
            sim.smooth_window_s = smooth_window_s
            sim.seed = seed
            sim.PCA()
            sweep_simulations[envelope_mode][smooth_window_s].append(sim)

for sims_by_window in sweep_simulations.values():
    for sim_list in sims_by_window.values():
        for sim in sim_list:
            sim.summary_metrics = summarize_simulation(sim)

sweep_summary = {envelope_mode: {} for envelope_mode in SWEEP_ENVELOPE_MODES}
for envelope_mode in SWEEP_ENVELOPE_MODES:
    for smooth_window_s in SWEEP_SMOOTH_WINDOWS_S:
        sim_list = sweep_simulations[envelope_mode][smooth_window_s]
        acf_decay_s = np.array([sim.summary_metrics['acf_decay_s'] for sim in sim_list], dtype=float)
        pc1_var_explained = np.array(
            [sim.summary_metrics['pc1_var_explained'] for sim in sim_list],
            dtype=float,
        )
        n_pcs_95 = np.array([sim.summary_metrics['n_pcs_95'] for sim in sim_list], dtype=float)
        effective_rank = np.array([sim.summary_metrics['effective_rank'] for sim in sim_list], dtype=float)

        acf_decay_s_mean, acf_decay_s_sem = mean_and_sem(acf_decay_s)
        pc1_var_explained_mean, pc1_var_explained_sem = mean_and_sem(pc1_var_explained)
        n_pcs_95_mean, n_pcs_95_sem = mean_and_sem(n_pcs_95)
        effective_rank_mean, effective_rank_sem = mean_and_sem(effective_rank)

        sweep_summary[envelope_mode][smooth_window_s] = dict(
            smooth_window_s=smooth_window_s,
            n_seeds=len(sim_list),
            acf_lags_s=sim_list[0].summary_metrics['acf_lags_s'],
            acf_decay_s=acf_decay_s,
            acf_decay_s_mean=acf_decay_s_mean,
            acf_decay_s_sem=acf_decay_s_sem,
            pc1_var_explained=pc1_var_explained,
            pc1_var_explained_mean=pc1_var_explained_mean,
            pc1_var_explained_sem=pc1_var_explained_sem,
            n_pcs_95=n_pcs_95,
            n_pcs_95_mean=n_pcs_95_mean,
            n_pcs_95_sem=n_pcs_95_sem,
            effective_rank=effective_rank,
            effective_rank_mean=effective_rank_mean,
            effective_rank_sem=effective_rank_sem,
            summary_freqs_hz=np.vstack([sim.summary_metrics['summary_freqs_hz'] for sim in sim_list]),
            mean_acf=np.vstack([sim.summary_metrics['mean_acf'] for sim in sim_list]),
            freq_corr_upper=np.vstack([sim.summary_metrics['freq_corr_upper'] for sim in sim_list]),
        )

metric_specs = [
    ('acf_decay_s', 'ACF decay (s)'),
    ('pc1_var_explained', 'PC1 variance explained'),
    ('n_pcs_95', 'PCs for 95% variance'),
    ('effective_rank', 'Effective rank'),
]
mode_colors = {'ind': 'tab:blue', 'shared': 'tab:orange'}

fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.5), dpi=300)
for ax, (metric_key, ylabel) in zip(axes.flat, metric_specs):
    for envelope_mode in SWEEP_ENVELOPE_MODES:
        means = np.array(
            [sweep_summary[envelope_mode][window][f'{metric_key}_mean'] for window in SWEEP_SMOOTH_WINDOWS_S],
            dtype=float,
        )
        sems = np.array(
            [sweep_summary[envelope_mode][window][f'{metric_key}_sem'] for window in SWEEP_SMOOTH_WINDOWS_S],
            dtype=float,
        )
        ax.errorbar(
            SWEEP_SMOOTH_WINDOWS_S,
            means,
            yerr=sems,
            lw=1,
            marker='o',
            ms=3,
            capsize=2,
            color=mode_colors[envelope_mode],
            label=envelope_mode,
        )
    ax.set_xscale('log')
    ax.set_xlabel('Smooth window (s)')
    ax.set_ylabel(ylabel)
axes[0, 0].legend(frameon=False, loc='best', fontsize=7)
finish_plot()


#%% Reconstruction diagnostics across rank in z-power space
RANKS = [1, 3, 5, 7, 9, 60]

for sim in simulations:
    sim.rank_spectrogram_error = []
    sim.rank_freq_corr_error = []
    sim.rank_freq_corr_similarity = []
    sim.rank_recon_z_power = {}
    sim.rank_recon_freq_corr = {}

    for rank in RANKS:
        recon_z_power = pca_rank_reconstruction(sim.z_pca, sim.z_power.T, rank).T
        recon_freq_corr = np.corrcoef(recon_z_power)

        sim.rank_recon_z_power[rank] = recon_z_power
        sim.rank_recon_freq_corr[rank] = recon_freq_corr
        sim.rank_spectrogram_error.append(relative_error(recon_z_power, sim.z_power))
        sim.rank_freq_corr_error.append(relative_error(recon_freq_corr, sim.truth_freq_corr))
        sim.rank_freq_corr_similarity.append(matrix_similarity(recon_freq_corr, sim.truth_freq_corr))


#%% Plot reconstructed frequency correlation matrices
for sim in simulations:
    ix = np.max(np.where(sim.spec_freqs_hz <= 100))
    plt.figure(figsize=(15, 3), dpi=300)
    for plot_idx, rank in enumerate([1, 3, 5, 7, 9], start=1):
        plt.subplot(1, 5, plot_idx)
        plt.imshow(
            sim.rank_recon_freq_corr[rank][:ix, :ix],
            aspect='auto',
            origin='lower',
            extent=[sim.spec_freqs_hz[0], sim.spec_freqs_hz[ix], sim.spec_freqs_hz[0], sim.spec_freqs_hz[ix]],
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
        )
        plt.colorbar()
        plt.xlabel('Frequency (Hz)')
        plt.title(f'Rank {rank}')
    plt.ylabel('Frequency (Hz)')
    plt.suptitle(f'{sim} reconstructed frequency correlation', y=1.02)
    finish_plot()


#%% Plot reconstructed z-power
for sim in simulations:
    ix = np.max(np.where(sim.spec_freqs_hz <= 100))
    plt.figure(figsize=(15, 3), dpi=300)
    for plot_idx, rank in enumerate([1, 3, 5, 7, 9], start=1):
        plt.subplot(1, 5, plot_idx)
        plt.pcolormesh(
            sim.spec_time[::STP * 3],
            sim.spec_freqs_hz[:ix],
            sim.rank_recon_z_power[rank][:ix, ::STP * 3],
            cmap='coolwarm',
            vmin=-3,
            vmax=3,
        )
        plt.colorbar()
        plt.xlabel('Time (s)')
        plt.title(f'Rank {rank}')
    plt.ylabel('Frequency (Hz)')
    plt.suptitle(f'{sim} reconstructed z-power', y=1.02)
    finish_plot()


#%% Reconstruction error vs rank
plt.figure(figsize=(3, 3), dpi=300)
for sim in simulations:
    plt.plot(RANKS, sim.rank_spectrogram_error, lw=1, marker='o', label=sim)
plt.xlabel('Rank')
plt.ylabel('Relative error')
plt.title('z-power recon error')
plt.legend(frameon=False, loc='best', fontsize=7)
finish_plot()

plt.figure(figsize=(3, 3), dpi=300)
for sim in simulations:
    plt.plot(RANKS, sim.rank_freq_corr_error, lw=1, marker='o', label=sim)
plt.xlabel('Rank')
plt.ylabel('Relative error')
plt.title('Correlation matrix recon error')
plt.legend(frameon=False, loc='best', fontsize=7)
finish_plot()


#%% Frequency-wise variance after reconstruction
for sim in simulations:
    plt.figure(figsize=(3, 3), dpi=300)
    for rank_idx, rank in enumerate(RANKS):
        recon_z_power = sim.rank_recon_z_power[rank]
        plt.plot(
            sim.spec_freqs_hz,
            recon_z_power.var(axis=1),
            lw=1,
            label=f'Rank-{rank}',
            c=plt.cm.rainbow((rank_idx + 1) / len(RANKS)),
        )
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Temporal variance')
    plt.title(f'{sim} variance after recon')
    plt.legend(frameon=False, loc='best', fontsize=4)
    finish_plot()


#%% X pattern in the frequency correlation
TEST_FWHM = 3
TEST_WAVELET_WINDOW_S = 5
TEST_SPEC_FREQS_HZ = np.arange(1, 6, dtype=float)

plot_wavelet_frequency_responses(
    freqs_hz=TEST_SPEC_FREQS_HZ,
    fs=SIM_FS,
    fwhm=TEST_FWHM,
    wavelet_window_s=TEST_WAVELET_WINDOW_S,
    max_hz=5,
)

test_spectrogram_kwargs = dict(
    fs=SIM_FS,
    freqs_hz=np.arange(1, 101, dtype=float),
    fwhm=TEST_FWHM,
    wavelet_window_s=TEST_WAVELET_WINDOW_S,
)
for sim in simulations:
    spec_time, spec_freqs_hz, spec_power = get_spectrogram(sim.trace, **test_spectrogram_kwargs)
    freq_corr = np.corrcoef(np.log(spec_power + 1e-12))

    plt.figure(figsize=(3, 3), dpi=300)
    plt.imshow(
        freq_corr,
        aspect='auto',
        origin='lower',
        extent=[spec_freqs_hz[0], spec_freqs_hz[-1], spec_freqs_hz[0], spec_freqs_hz[-1]],
        cmap='Blues',
    )
    plt.colorbar()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Frequency (Hz)')
    plt.title(f'{sim} freq corr')
    finish_plot()

xpattern_sim = shared_noise
_, xpattern_freqs_hz, xpattern_power = get_spectrogram(xpattern_sim.trace, **test_spectrogram_kwargs)
xpattern_corr = np.corrcoef(np.log(xpattern_power + 1e-12))


#%% Visualize scatter powers at nearby frequencies, colored by their correlation
base = 3
n_freq = 2 * base + 1
ix = 11 - base
fig, ax = plt.subplots(n_freq, n_freq, figsize=(n_freq, n_freq), dpi=300)

for i in range(n_freq):
    for j in range(n_freq):
        ax[i, j].scatter(
            xpattern_power[i + ix],
            xpattern_power[j + ix],
            alpha=0.5,
            s=10,
            color=plt.cm.Blues(xpattern_corr[i + ix, j + ix] + 0.1),
        )
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])
        ax[i, j].set_xlabel(f'{xpattern_freqs_hz[i + ix]:.0f} Hz', fontsize=6)
        ax[i, j].set_ylabel(f'{xpattern_freqs_hz[j + ix]:.0f} Hz', fontsize=6)
finish_plot()
