#%%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA

import fa
from simulate_core import (
    build_ground_truth_spectrogram,
    mean_and_sem,
    simulate_trace,
    summarize_simulation,
)
from spectral import get_freq_cov_from_power, get_psd, get_spectrogram
from utils import fig_set, finish_plot

class FactorAnalysisResult:
    def __init__(self, fit_result: dict):
        for key, value in fit_result.items():
            setattr(self, key, value)

    def __repr__(self):
        return f'{self.subspace.shape[0]} components from {self.subspace.shape[1]} channels'

def sanity_check_raw_and_downsampled_lfp():
    """Sanity-check raw and saved downsampled LFP for one session."""
    from scipy import io
    from preprocess import LFP_processor, get_ns5_path, get_output_mat_path

    regions = ['HPC']
    raw_window_s = 5
    ds_window_s = 5

    ns5_path = get_ns5_path('YFT', 32)
    ds_mat_path = get_output_mat_path(ns5_path)

    lfp = LFP_processor(ns5_path, regions=regions)
    if len(lfp.chosen_channel_ids) == 0:
        raise ValueError(f'No channels found for regions={regions} in {ns5_path}')

    channel_id = lfp.chosen_channel_ids[0]

    raw_end = min(int(raw_window_s * lfp.fs), lfp.n_samples)//100
    raw_trace = lfp.recording.get_traces(
        channel_ids=[channel_id],
        segment_index=lfp.segment_index,
        start_frame=0,
        end_frame=raw_end,
    ).squeeze()
    raw_fs = lfp.fs
    raw_time = np.arange(raw_trace.shape[0]) / raw_fs

    ds_mat = io.loadmat(ds_mat_path, squeeze_me=True)
    ds_lfp = np.asarray(ds_mat['lfp_ds'])
    ds_fs = float(ds_mat['fs'])
    ds_end = min(int(ds_window_s * ds_fs), ds_lfp.shape[0])//100
    if ds_lfp.ndim == 1:
        ds_trace = ds_lfp[:ds_end]
    else:
        ds_trace = ds_lfp[:ds_end, 0]
    ds_time = np.arange(ds_trace.shape[0]) / ds_fs

    fig, ax = plt.subplots(1,1, figsize=(6,3), dpi=300)
    ax.plot(raw_time, raw_trace, lw=0.8, label='Raw')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')

    ax.plot(ds_time, ds_trace, lw=0.8, color='tab:orange', label='Downsampled')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')

    ax.legend(frameon=False, loc='upper right')
    sns.despine(trim=False)
    fig.tight_layout()
    plt.show()

    psd_window_s = 2.0
    psd_overlap_frac = 0.5
    f_raw, psd_raw = get_psd(raw_trace,fs=raw_fs,window_s=psd_window_s,overlap_frac=psd_overlap_frac)
    f_ds, psd_ds = get_psd(ds_trace,fs=ds_fs,window_s=psd_window_s,overlap_frac=psd_overlap_frac)

    plt.figure(figsize=(3,3), dpi=300)
    plt.plot(f_raw, psd_raw, lw=1, label=f'Raw ({lfp.fs//1000:.0f} kHz)')
    plt.plot(f_ds, psd_ds, lw=1, label=f'Downsampled ({ds_fs//1000:.0f} kHz)')
    plt.axvline(400, ls='--', c='k', lw=1, alpha=0.4) # Low passed at 400 Hz.
    plt.yscale('log')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD')
    plt.legend(frameon=False, loc='lower left')
    sns.despine(trim=False)
    plt.tight_layout()
    plt.show()

plt.figure()
plt.close()
fig_set(font_size=10, linewidth=0.8)

class SimulationResult:
    def __init__(self, envelope_mode, sim_kwargs, psd_kwargs, spectrogram_kwargs):
        self.envelope_mode = envelope_mode
        self.trace, self.envelopes, self.carriers = simulate_trace(
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
        self.z_power, self.freq_corr = get_freq_cov_from_power(self.spec_power)
        self.truth_z_power, self.truth_freq_corr = get_freq_cov_from_power(self.truth_power)
        self.demean_power, self.demean_freq_corr = get_freq_cov_from_power(self.spec_power_demean, z_scored=False)

    def PCA(self):
        self.pca = PCA().fit(self.spec_power.T)
        self.pca_truth = PCA().fit(self.truth_power.T)
        self.pca_demean = PCA().fit(self.spec_power_demean.T)
        self.z_pca = PCA().fit(self.z_power.T)

    def FA(self, shared_var_thresh=0.95):
        self.fa = FactorAnalysisResult(fa.fa_fit(self.spec_power.T, shared_var_thresh=shared_var_thresh))
        self.z_fa = FactorAnalysisResult(fa.fa_fit(self.z_power.T, shared_var_thresh=shared_var_thresh))

    def __repr__(self):
        return f'{self.envelope_mode} envelope'

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
        overlap[wavelet_idx] = np.interp(center_freqs_hz, response_freqs_hz, response_by_wavelet[wavelet_idx])
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
    max_overlap = max_off_center_response(response_freqs_hz, response_by_wavelet, np.asarray(freqs_hz, dtype=float))
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

def pca_rank_reconstruction(pca, features_by_time, rank):
    rank = min(rank, pca.components_.shape[0])
    scores = pca.transform(features_by_time)
    return scores[:, :rank] @ pca.components_[:rank, :] + pca.mean_

#%% General simulation parameters
SIM_FS = 500
SIM_DURATION_S = 150
SIM_TIME = np.arange(0, SIM_DURATION_S, 1 / SIM_FS)
SIM_FREQS_HZ = np.array([12, 30, 70], dtype=float)
SMOOTH_WINDOW_S = 10.0
 
# PSD parameters
PSD_WINDOW_S = 1.0
PSD_OVERLAP_FRAC = 0.5

# Spectrogram parameters
FWHM = 0.5
WAVELET_WINDOW_S = 1.0
SPEC_FREQS_HZ = np.arange(1, 101, dtype=float)
STP = max(int(round(SIM_FS / 200)), 1)

sim_kwargs = dict(sim_time=SIM_TIME,rng=np.random.default_rng(0),freqs_hz=SIM_FREQS_HZ,smooth_window_s=SMOOTH_WINDOW_S,
                  base_amplitudes=np.array([30, 28, 25], dtype=float),
                  phases_rad=np.array([0.0, 0.6, 1.1], dtype=float),
                  envelope_scales=np.array([12, 8, 4], dtype=float),)
psd_kwargs = dict(fs=SIM_FS, window_s=PSD_WINDOW_S, overlap_frac=PSD_OVERLAP_FRAC)
spectrogram_kwargs = dict(fs=SIM_FS, freqs_hz=SPEC_FREQS_HZ, fwhm=FWHM, wavelet_window_s=WAVELET_WINDOW_S)

ind_noise = SimulationResult('ind', sim_kwargs, psd_kwargs, spectrogram_kwargs)
shared_noise = SimulationResult('shared', sim_kwargs, psd_kwargs, spectrogram_kwargs)
simulations = [ind_noise, shared_noise]

for sim in simulations:
    sim.PCA()

#%% Envelopes
for sim in simulations:
    plt.figure(figsize=(3, 3), dpi=300)
    for freq_idx, freq_hz in enumerate(SIM_FREQS_HZ):
        plt.plot(SIM_TIME[1000:3000], sim.envelopes[freq_idx, 1000:3000], lw=1, label=f'{freq_hz:.0f} Hz')
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
    plt.pcolormesh(sim.spec_time[::STP], sim.spec_freqs_hz[:ix], np.log(sim.spec_power[:ix, ::STP]), cmap='coolwarm', vmin=-20, vmax=10)
    plt.colorbar()
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(f'{sim} log power')
    finish_plot()

#%% Spectrograms where wavelets are definitely not overlapping

#%% Ground truth spectrograms
for sim in simulations:
    ix = np.max(np.where(sim.spec_freqs_hz <= 100))
    plt.figure(figsize=(3, 3), dpi=300)
    plt.pcolormesh(sim.spec_time[::STP], sim.spec_freqs_hz[:ix], sim.truth_power[:ix, ::STP], cmap='Blues')
    plt.colorbar()
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(f'{sim} truth')
    finish_plot()

#%% Frequency correlation matrices
for sim in simulations:
    ix = np.max(np.where(sim.spec_freqs_hz <= 100))
    plt.figure(figsize=(3, 3), dpi=300)
    plt.imshow(sim.demean_freq_corr[:ix, :ix],aspect='auto',origin='lower',
               extent=[sim.spec_freqs_hz[0], sim.spec_freqs_hz[ix], sim.spec_freqs_hz[0], sim.spec_freqs_hz[ix]],
               cmap='coolwarm',vmin=-1,vmax=1)
    plt.colorbar()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Frequency (Hz)')
    plt.title(sim)
    finish_plot()
#%% Ground truth frequency correlation matrices
for sim in simulations:
    ix = np.max(np.where(sim.spec_freqs_hz <= 100))
    plt.figure(figsize=(3, 3), dpi=300)
    plt.imshow(sim.truth_freq_corr[:ix, :ix],aspect='auto',origin='lower',
               extent=[sim.spec_freqs_hz[0], sim.spec_freqs_hz[ix], sim.spec_freqs_hz[0], sim.spec_freqs_hz[ix]],
               cmap='coolwarm',vmin=-1,vmax=1)
    plt.colorbar()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Frequency (Hz)')
    plt.title(sim)
    finish_plot()

#%% PCA: dimensionality

plt.figure(figsize=(3, 3), dpi=300)
for sim in simulations:
    dims = np.where(np.cumsum(sim.pca.explained_variance_ratio_) >= 0.99)[0][0]+1
    plt.plot(np.cumsum(sim.pca.explained_variance_ratio_), lw=1, label=f'{sim} ({dims} PCs)')
    dims = np.where(np.cumsum(sim.pca_truth.explained_variance_ratio_) >= 0.99)[0][0]+1
    plt.plot(np.cumsum(sim.pca_truth.explained_variance_ratio_), lw=1, ls='--', label=f'{sim} truth, ({dims} PCs)')
plt.xlabel('Number of PCs')
plt.ylabel('Cumulative Explained Variance')
plt.legend(fontsize=6, frameon=False, loc='lower right')
plt.xlim([-0.5,10])
finish_plot()

#%% Plot the first FA component loadings.

for sim in simulations: 
    plt.figure(figsize=(3, 3), dpi=300)
    plt.plot(sim.fa.subspace[0])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Loading')
    plt.title(f'{sim} FA first component')
    finish_plot()

#%% [Example] Projection to the latent subspace

ix = 0 # Which latent component to inspect
sim = shared_noise
pca_proj = sim.pca_demean.transform(sim.spec_power_demean.T)[:, ix]
fa_proj, _ = fa.fa_transform(sim.z_power.T, sim.z_fa.fa)
fa_proj = fa_proj[:, ix]
envelope = sim.envelopes[0]

# Normalize for shape comparison only.
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

#%% Smooth-window sensitivity: simulation sweep
SWEEP_SMOOTH_WINDOWS_S = np.array([0.5, 5, 50], dtype=float)
SWEEP_SEEDS = np.arange(3)
SWEEP_ENVELOPE_MODES = ['ind', 'shared']

# Build a bank of simulations across smoothing windows and random seeds.
# This lets us quantify sensitivity without relying on a single realization.
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


# Smooth-window sensitivity: summary metrics
for sims_by_window in sweep_simulations.values():
    for sim_list in sims_by_window.values():
        for sim in sim_list:
            sim.summary_metrics = summarize_simulation(sim)

sweep_summary = {envelope_mode: {} for envelope_mode in SWEEP_ENVELOPE_MODES}
for envelope_mode in SWEEP_ENVELOPE_MODES:
    for smooth_window_s in SWEEP_SMOOTH_WINDOWS_S:
        sim_list = sweep_simulations[envelope_mode][smooth_window_s]
        acf_decay_s = np.array([sim.summary_metrics['acf_decay_s'] for sim in sim_list], dtype=float)
        pc1_var_explained = np.array([sim.summary_metrics['pc1_var_explained'] for sim in sim_list], dtype=float)
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

#%% Smooth-window sensitivity: scalar summaries
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
        means = np.array([sweep_summary[envelope_mode][window][f'{metric_key}_mean'] for window in SWEEP_SMOOTH_WINDOWS_S], dtype=float)
        sems = np.array([sweep_summary[envelope_mode][window][f'{metric_key}_sem'] for window in SWEEP_SMOOTH_WINDOWS_S], dtype=float)
        ax.errorbar(SWEEP_SMOOTH_WINDOWS_S, means, yerr=sems, lw=1, marker='o', ms=3,
                    capsize=2, color=mode_colors[envelope_mode], label=envelope_mode)
    ax.set_xscale('log')
    ax.set_xlabel('Smooth window (s)')
    ax.set_ylabel(ylabel)
axes[0, 0].legend(frameon=False, loc='best', fontsize=7)
finish_plot()


#%% Rank-N Reconstruction Diagnostics


RANKS = [1, 3, 5, 7, 9]

for sim in simulations:
    if not hasattr(sim, 'z_pca'):
        sim.z_pca = PCA().fit(sim.z_power.T)
    sim.rank_spectrogram_error = []
    sim.rank_freq_corr_error = []
    sim.rank_freq_corr_similarity = []
    sim.rank_recon_z_power = {}
    sim.rank_recon_freq_corr = {}

    for rank in RANKS:
        recon_z_power = pca_rank_reconstruction(sim.z_pca, sim.z_power.T, rank).T
        recon_freq_corr = np.cov(recon_z_power)

        sim.rank_recon_z_power[rank] = recon_z_power
        sim.rank_recon_freq_corr[rank] = recon_freq_corr

#%% Reconstructed Frequency Correlation Matrices

NONOVERLAP_FWHM = 3
NONOVERLAP_WAVELET_WINDOW_S = 5
NONOVERLAP_SPEC_FREQS_HZ = np.arange(1, 6, dtype=float)

plot_wavelet_frequency_responses(freqs_hz=NONOVERLAP_SPEC_FREQS_HZ, fs=SIM_FS, fwhm=NONOVERLAP_FWHM, wavelet_window_s=NONOVERLAP_WAVELET_WINDOW_S, max_hz=5,)

nonoverlap_spectrogram_kwargs = dict(fs=SIM_FS,freqs_hz=np.arange(1,101,dtype=float),fwhm=NONOVERLAP_FWHM,wavelet_window_s=NONOVERLAP_WAVELET_WINDOW_S,)
for sim in simulations:
    spec_time, spec_freqs_hz, spec_power = get_spectrogram(sim.trace, **nonoverlap_spectrogram_kwargs)
    freq_corr = np.cov(spec_power.T)

    plt.figure(figsize=(3, 3), dpi=300)
    plt.imshow(freq_corr,aspect='auto',origin='lower',
        extent=[spec_freqs_hz[0], spec_freqs_hz[-1], spec_freqs_hz[0], spec_freqs_hz[-1]],
        cmap='Blues',
        # vmin=-1,
        # vmax=1,
    )
    plt.colorbar()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Frequency (Hz)')
    plt.title(f'{sim} freq corr')
    finish_plot()


plt.figure(figsize=(15, 3), dpi=300)
for i, rank in enumerate([1, 3, 5, 7, 9]):
    pca = PCA().fit(np.log(spec_power.T + 1e-12))

    recon_z_power = pca_rank_reconstruction(pca, spec_power.T, rank).T
    recon_freq_corr = np.cov(recon_z_power)
    plt.subplot(1, 5, i + 1)
    plt.imshow( recon_freq_corr, aspect='auto', origin='lower',
        extent=[sim.spec_freqs_hz[0], sim.spec_freqs_hz[-1], sim.spec_freqs_hz[0], sim.spec_freqs_hz[-1]],
        cmap='coolwarm',
        # vmin=-1,
        # vmax=1,
    )
    plt.colorbar()
    plt.xlabel('Frequency (Hz)')
    plt.title(f'Rank {rank}')
plt.ylabel('Frequency (Hz)')
plt.suptitle(f'{sim} rank-N recon freq corr', y=1.02)
finish_plot()

# %%
