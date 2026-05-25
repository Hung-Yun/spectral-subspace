#%%
import numpy as np
from utils import fig_set, finish_plot
import fa
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

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

def get_psd(trace, fs, window_s, overlap_frac, window='hann'):
    from scipy import signal

    nperseg = min(int(window_s * fs), trace.shape[0])
    if nperseg < 1:
        raise ValueError('PSD window is too short for the provided trace.')

    noverlap = min(int(nperseg * overlap_frac), max(nperseg - 1, 0))
    freqs, psd = signal.welch(trace,fs=fs,window=window,nperseg=nperseg,noverlap=noverlap)
    return freqs, psd


def get_spectrogram(trace, fs, freqs_hz, fwhm=0.3, wavelet_window_s=1.0):
    trace = np.asarray(trace, dtype=float)
    freqs_hz = np.asarray(freqs_hz, dtype=float)

    wavetime = np.arange(-wavelet_window_s, wavelet_window_s, 1 / fs)
    gaussian = np.exp(-(4 * np.log(2) * wavetime**2) / fwhm**2)
    wavelets = np.zeros((len(freqs_hz), len(wavetime)), dtype=complex)
    for freq_idx, freq_hz in enumerate(freqs_hz):
        wavelets[freq_idx] = np.exp(1j * 2 * np.pi * freq_hz * wavetime) * gaussian

    data_len = trace.shape[0]
    nconv = data_len + len(wavetime) - 1
    halfk = int(np.floor(len(wavetime) / 2))
    data_fft = np.fft.fft(trace, nconv)
    tf_power = np.zeros((len(freqs_hz), data_len))

    for freq_idx in range(len(freqs_hz)):
        wavelet_fft = np.fft.fft(wavelets[freq_idx], nconv)
        wavelet_fft /= np.max(np.abs(wavelet_fft))
        convres = np.fft.ifft(wavelet_fft * data_fft)
        convres = convres[halfk - 1:-halfk]
        tf_power[freq_idx] = np.abs(convres) ** 2

    tf_time = np.arange(data_len) / fs
    return tf_time, freqs_hz, tf_power


def get_freq_cov_from_power(power_by_freq_time, z_scored=True, eps=1e-12):
    power_by_freq_time = np.log(power_by_freq_time + eps)
    power_mean = power_by_freq_time.mean(axis=1, keepdims=True)
    power_std = power_by_freq_time.std(axis=1, keepdims=True)
    variable_rows = (power_std.squeeze() > 0)
    power_std[power_std == 0] = 1

    if z_scored:
        z_power = (power_by_freq_time - power_mean) / power_std
        freq_corr = np.zeros((z_power.shape[0], z_power.shape[0]), dtype=float)
        if np.any(variable_rows):
            if np.sum(variable_rows) == 1:
                freq_corr[np.ix_(variable_rows, variable_rows)] = 1.0
            else:
                freq_corr[np.ix_(variable_rows, variable_rows)] = np.corrcoef(z_power[variable_rows])
        return z_power, freq_corr
    else: # just compute freq_corr
        freq_corr = np.zeros((power_by_freq_time.shape[0], power_by_freq_time.shape[0]), dtype=float)
        if np.any(variable_rows):
            if np.sum(variable_rows) == 1:
                freq_corr[np.ix_(variable_rows, variable_rows)] = 1.0
            else:
                freq_corr[np.ix_(variable_rows, variable_rows)] = np.corrcoef(power_by_freq_time[variable_rows])
        return None, freq_corr


def simulate_trace(sim_time,rng,freqs_hz,base_amplitudes,phases_rad,envelope_mode,
                   envelope_scales,smooth_window_s,additive_noise_sd=0,):
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    base_amplitudes = np.asarray(base_amplitudes, dtype=float)
    phases_rad = np.asarray(phases_rad, dtype=float)
    envelope_scales = np.asarray(envelope_scales, dtype=float)

    def build_smooth_noise(sim_time, rng, scale, smooth_window_s):
        if scale == 0:
            return np.zeros(sim_time.shape[0])

        dt = sim_time[1] - sim_time[0]
        n_samples = sim_time.shape[0]
        noise = rng.standard_normal(sim_time.shape[0])
        smooth_window_samples = max(int(round(smooth_window_s / dt)), 1)
        kernel = np.ones(smooth_window_samples, dtype=float)
        kernel /= kernel.sum()
        pad = smooth_window_samples // 2
        noise_pad = np.pad(noise, pad_width=pad, mode='wrap')
        smooth_noise = np.convolve(noise_pad, kernel, mode='valid')
        smooth_noise = smooth_noise[:n_samples]
        smooth_noise -= smooth_noise.mean()

        noise_std = smooth_noise.std()
        if noise_std > 0:
            smooth_noise /= noise_std

        return scale * smooth_noise

    if len({arr.shape for arr in (freqs_hz, base_amplitudes, phases_rad, envelope_scales)}) != 1:
        raise ValueError('All per-frequency parameter arrays must have the same shape.')

    envelopes = np.repeat(base_amplitudes[:, None], sim_time.shape[0], axis=1)
    if envelope_mode not in {'constant', 'ind', 'shared'}:
        raise ValueError("envelope_mode must be 'constant', 'ind', or 'shared'.")

    if envelope_mode != 'constant':
        if envelope_mode == 'ind':
            for freq_idx, scale in enumerate(envelope_scales):
                envelopes[freq_idx] += build_smooth_noise(sim_time, rng, scale, smooth_window_s)
        else:
            shared_noise = build_smooth_noise(sim_time, rng, scale=1, smooth_window_s=smooth_window_s)
            envelopes += envelope_scales[:, None] * shared_noise[None, :]

    envelopes = np.clip(envelopes, a_min=0, a_max=None)
    carriers = np.sin(2 * np.pi * freqs_hz[:, None] * sim_time[None, :] + phases_rad[:, None])
    sim_trace = np.sum(envelopes * carriers, axis=0)
    if additive_noise_sd > 0:
        sim_trace += additive_noise_sd * rng.standard_normal(sim_time.shape[0])
    return sim_trace, envelopes, carriers


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
        self.z_power, self.freq_corr = get_freq_cov_from_power(self.spec_power)
        self.truth_z_power, self.truth_freq_corr = get_freq_cov_from_power(self.truth_power)

    def PCA(self):
        from sklearn.decomposition import PCA

        self.pca = PCA().fit(self.spec_power.T)
        self.truth_pca = PCA().fit(self.truth_power.T)
        self.z_pca = PCA().fit(self.z_power.T)

    def FA(self, shared_var_thresh=0.95):
        import fa

        self.fa = FactorAnalysisResult(fa.fa_fit(self.spec_power.T, shared_var_thresh=shared_var_thresh))
        self.z_fa = FactorAnalysisResult(fa.fa_fit(self.z_power.T, shared_var_thresh=shared_var_thresh))

    def __repr__(self):
        return f'{self.envelope_mode} envelope'

def get_autocorr(trace, max_lag_samples):
    from scipy import signal

    trace = np.asarray(trace, dtype=float)
    trace = trace - trace.mean()
    if np.allclose(trace, 0):
        return np.zeros(max_lag_samples + 1, dtype=float)

    full_corr = signal.correlate(trace, trace, mode='full')
    acf = full_corr[trace.shape[0] - 1:trace.shape[0] + max_lag_samples]
    return acf / acf[0]


def participation_ratio(values):
    values = np.asarray(values, dtype=float)
    values = values[values > 0]
    if values.size == 0:
        return np.nan
    return values.sum() ** 2 / np.sum(values ** 2)


def summarize_simulation(sim, n_summary_freqs=3, acf_max_lag_s=60, fluct_psd_window_s=20):
    log_power = np.log(sim.spec_power + 1e-12)
    fs = 1 / np.median(np.diff(sim.spec_time))
    summary_freq_idx = np.argsort(log_power.mean(axis=1))[-n_summary_freqs:]
    summary_freq_idx = np.sort(summary_freq_idx)
    summary_freqs_hz = sim.spec_freqs_hz[summary_freq_idx]

    acf_max_lag_samples = min(int(round(acf_max_lag_s * fs)), log_power.shape[1] - 1)
    acf_by_freq = np.vstack([get_autocorr(log_power[freq_idx], acf_max_lag_samples) for freq_idx in summary_freq_idx])
    mean_acf = acf_by_freq.mean(axis=0)
    acf_lags_s = np.arange(mean_acf.shape[0]) / fs
    decay_idx = np.where(mean_acf < np.exp(-1))[0]
    acf_decay_s = acf_lags_s[decay_idx[0]] if decay_idx.size else np.nan

    mean_log_power = log_power[summary_freq_idx].mean(axis=0)
    fluct_psd_f, fluct_psd = get_psd(mean_log_power, fs=fs, window_s=fluct_psd_window_s, overlap_frac=0.5)

    pca_cumulative = np.cumsum(sim.pca.explained_variance_ratio_)
    freq_corr_upper = sim.freq_corr[np.triu_indices_from(sim.freq_corr, k=1)]

    return dict(summary_freq_idx=summary_freq_idx,
                summary_freqs_hz=summary_freqs_hz,
                acf_lags_s=acf_lags_s,
                mean_acf=mean_acf,
                acf_decay_s=acf_decay_s,
                fluct_psd_f=fluct_psd_f,
                fluct_psd=fluct_psd,
                pc1_var_explained=sim.pca.explained_variance_ratio_[0],
                n_pcs_95=np.searchsorted(pca_cumulative, 0.95) + 1,
                effective_rank=participation_ratio(sim.pca.explained_variance_),
                freq_corr_upper=freq_corr_upper,)


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

#%%
# General simulation parameters
SIM_FS = 500
SIM_DURATION_S = 600
SIM_TIME = np.arange(0, SIM_DURATION_S, 1 / SIM_FS)
SIM_FREQS_HZ = np.array([12, 30, 70], dtype=float)
 
# PSD parameters
PSD_WINDOW_S = 1.0
PSD_OVERLAP_FRAC = 0.5

# Spectrogram parameters
FWHM = 0.5
WAVELET_WINDOW_S = 1.0
SPEC_FREQS_HZ = np.arange(1, 101, dtype=float)
STP = max(int(round(SIM_FS / 200)), 1)

sim_kwargs = dict(sim_time=SIM_TIME,rng=np.random.default_rng(0),freqs_hz=SIM_FREQS_HZ,smooth_window_s=10,
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
    plt.imshow(sim.freq_corr[:ix, :ix],aspect='auto',origin='lower',
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
    dims = np.where(np.cumsum(sim.truth_pca.explained_variance_ratio_) >= 0.99)[0][0]+1
    plt.plot(np.cumsum(sim.truth_pca.explained_variance_ratio_), lw=1, ls='--', label=f'{sim} truth, ({dims} PCs)')
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
pca_proj = sim.z_pca.transform(sim.z_power.T)[:, ix]
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

#%%
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
            fluct_psd_f=sim_list[0].summary_metrics['fluct_psd_f'],
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
            fluct_psd=np.vstack([sim.summary_metrics['fluct_psd'] for sim in sim_list]),
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

#%% Smooth-window sensitivity: temporal summaries
selected_windows_s = np.array([SWEEP_SMOOTH_WINDOWS_S[0], SWEEP_SMOOTH_WINDOWS_S[3], SWEEP_SMOOTH_WINDOWS_S[-1]])
window_colors = plt.cm.viridis(np.linspace(0.15, 0.85, selected_windows_s.size))

fig, axes = plt.subplots(2, 2, figsize=(7, 5.5), dpi=300)
for row_idx, envelope_mode in enumerate(SWEEP_ENVELOPE_MODES):
    ax_acf = axes[row_idx, 0]
    ax_psd = axes[row_idx, 1]
    for color, smooth_window_s in zip(window_colors, selected_windows_s):
        summary = sweep_summary[envelope_mode][smooth_window_s]
        ax_acf.plot(summary['acf_lags_s'], summary['mean_acf'].mean(axis=0), lw=1,
                    color=color, label=f'{smooth_window_s:g} s')

        mean_fluct_psd = summary['fluct_psd'].mean(axis=0)
        psd_mask = summary['fluct_psd_f'] <= 2
        ax_psd.plot(summary['fluct_psd_f'][psd_mask], mean_fluct_psd[psd_mask], lw=1,
                    color=color, label=f'{smooth_window_s:g} s')

    ax_acf.set_xlim([0, 60])
    ax_acf.set_ylim([-0.1, 1.02])
    ax_acf.set_xlabel('Lag (s)')
    ax_acf.set_ylabel(f'{envelope_mode} mean ACF')

    ax_psd.set_xlabel('Fluctuation frequency (Hz)')
    ax_psd.set_ylabel(f'{envelope_mode} fluctuation PSD')
    ax_psd.set_yscale('log')
    ax_psd.set_xlim([0, 2])

axes[0, 0].legend(frameon=False, loc='upper right', fontsize=7, title='Window')
finish_plot()

# %%
