#%%
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from utils import fig_set


def get_psd(trace, fs, window_s, overlap_frac, window='hann'):
    nperseg = min(int(window_s * fs), trace.shape[0])
    if nperseg < 1:
        raise ValueError('PSD window is too short for the provided trace.')

    noverlap = min(int(nperseg * overlap_frac), max(nperseg - 1, 0))
    freqs, psd = signal.welch(trace, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
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

    freq_corr = np.zeros((power_by_freq_time.shape[0], power_by_freq_time.shape[0]), dtype=float)
    if np.any(variable_rows):
        if np.sum(variable_rows) == 1:
            freq_corr[np.ix_(variable_rows, variable_rows)] = 1.0
        else:
            freq_corr[np.ix_(variable_rows, variable_rows)] = np.corrcoef(power_by_freq_time[variable_rows])
    return None, freq_corr


def simulate_trace(sim_time, rng, freqs_hz, base_amplitudes, phases_rad, envelope_mode,
                   envelope_scales, smooth_window_s, additive_noise_sd=0):
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

    def dim_reduction(self):
        self.z_pca = PCA().fit(self.z_power.T)

    def __repr__(self):
        return f'{self.envelope_mode} envelope'


def finish_plot():
    sns.despine(trim=False)
    plt.tight_layout()
    plt.show()
    plt.close()


def relative_error(estimate, target):
    return np.linalg.norm(estimate - target) / np.linalg.norm(target)


def matrix_similarity(a, b):
    iu = np.triu_indices_from(a, k=1)
    a_flat = a[iu]
    b_flat = b[iu]
    valid = np.isfinite(a_flat) & np.isfinite(b_flat)
    if np.sum(valid) < 2:
        return np.nan
    return np.corrcoef(a_flat[valid], b_flat[valid])[0, 1]


def pca_rank_reconstruction(pca, features_by_time, rank):
    scores = pca.transform(features_by_time)
    components = pca.components_
    return scores[:, :rank] @ components[:rank, :] + pca.mean_


plt.figure()
plt.close()
fig_set(font_size=10, linewidth=0.8)

#%% Simulation Setup
SIM_FS = 500
SIM_DURATION_S = 600
SIM_TIME = np.arange(0, SIM_DURATION_S, 1 / SIM_FS)
SIM_FREQS_HZ = np.array([12, 30, 70], dtype=float)
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
    smooth_window_s=10,
    base_amplitudes=np.array([30, 28, 25], dtype=float),
    phases_rad=np.array([0.0, 0.6, 1.1], dtype=float),
    envelope_scales=np.array([12, 8, 4], dtype=float),
)
psd_kwargs = dict(fs=SIM_FS, window_s=PSD_WINDOW_S, overlap_frac=PSD_OVERLAP_FRAC)
spectrogram_kwargs = dict(fs=SIM_FS, freqs_hz=SPEC_FREQS_HZ, fwhm=FWHM, wavelet_window_s=WAVELET_WINDOW_S)

ind_noise = SimulationResult('ind', sim_kwargs, psd_kwargs, spectrogram_kwargs)
shared_noise = SimulationResult('shared', sim_kwargs, psd_kwargs, spectrogram_kwargs)
simulations = [ind_noise, shared_noise]

#%% Dimensionality Reduction
for sim in simulations:
    sim.dim_reduction()

#%% Reconstruction Diagnostics Across Rank In z-Power Space
RANKS = [1, 3, 5, 7, 9, 60]
DIAGNOSTIC_RANK = 3

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

#%% Plot Reconstructed Frequency Correlation Matrices
for sim in simulations:
    ix = np.max(np.where(sim.spec_freqs_hz <= 100))
    fig, ax = plt.subplots(1, 5, figsize=(15, 3), dpi=300)
    for i, rank in enumerate([1, 3, 5, 7, 9]):
        plt.subplot(1, 5, i + 1)
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
    finish_plot()

#%% Plot Reconstructed z-Power
for sim in simulations:
    ix = np.max(np.where(sim.spec_freqs_hz <= 100))
    fig, ax = plt.subplots(1, 5, figsize=(15, 3), dpi=300)
    for i, rank in enumerate([1, 3, 5, 7, 9]):
        plt.subplot(1, 5, i + 1)
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
    finish_plot()

#%% Reconstruction Error vs Rank: z-Power
plt.figure(figsize=(3, 3), dpi=300)
for sim in simulations:
    plt.plot(RANKS, sim.rank_spectrogram_error, lw=1, marker='o', label=sim)
plt.xlabel('Rank')
plt.ylabel('Relative error')
plt.title('z-power recon error')
plt.legend(frameon=False, loc='best', fontsize=7)
finish_plot()

#%% Reconstruction Error vs Rank: Correlation Matrix
plt.figure(figsize=(3, 3), dpi=300)
for sim in simulations:
    plt.plot(RANKS, sim.rank_freq_corr_error, lw=1, marker='o', label=sim)
plt.xlabel('Rank')
plt.ylabel('Relative error')
plt.title('Correlation matrix recon error')
plt.legend(frameon=False, loc='best', fontsize=7)
finish_plot()

#%% Frequency-Wise Variance After Reconstruction
for sim in simulations:
    plt.figure(figsize=(3, 3), dpi=300)
    for j, rank in enumerate(RANKS):
        recon_z_power = sim.rank_recon_z_power[rank]
        plt.plot(sim.spec_freqs_hz, recon_z_power.var(axis=1), lw=1, label=f'Rank-{rank}', c=plt.cm.rainbow((j + 1) / len(RANKS)))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Temporal variance')
    plt.title(f'{sim} variance after recon')
    plt.legend(frameon=False, loc='best', fontsize=4)
    finish_plot()
