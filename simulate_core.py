import numpy as np

from spectral import get_autocorr
from scipy import signal

def simulate_trace(sim_time, rng, freqs_hz, base_amplitudes, phases_rad, envelope_mode,
                   envelope_scales, smooth_window_s, additive_noise_sd=0):
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    base_amplitudes = np.asarray(base_amplitudes, dtype=float)
    phases_rad = np.asarray(phases_rad, dtype=float)
    envelope_scales = np.asarray(envelope_scales, dtype=float)

    def build_smooth_noise(time, random_state, scale, window_s):
        if scale == 0:
            return np.zeros(time.shape[0])

        dt = time[1] - time[0]
        n_samples = time.shape[0]
        noise = random_state.standard_normal(n_samples)
        smooth_window_samples = max(int(round(window_s / dt)), 1)
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

        # sos = signal.butter(5, 0.1, btype='lowpass', fs=100, output='sos')
        # smooth_noise = signal.sosfiltfilt(sos, smooth_noise)

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
            shared_noise = build_smooth_noise(sim_time, rng, scale=1, window_s=smooth_window_s)
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
    acf_by_freq = np.vstack([get_autocorr(log_power[freq_idx], acf_max_lag_samples) for freq_idx in summary_freq_idx])
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


def init_summary_bucket():
    return dict(
        n_seeds=0,
        acf_lags_s=None,
        acf_decay_s=[],
        pc1_var_explained=[],
        n_pcs_95=[],
        effective_rank=[],
        summary_freqs_hz=[],
        mean_acf=[],
        freq_corr_upper=[],
    )


def update_summary_bucket(bucket, metrics):
    if bucket['acf_lags_s'] is None:
        bucket['acf_lags_s'] = metrics['acf_lags_s']
    bucket['n_seeds'] += 1
    bucket['acf_decay_s'].append(metrics['acf_decay_s'])
    bucket['pc1_var_explained'].append(metrics['pc1_var_explained'])
    bucket['n_pcs_95'].append(metrics['n_pcs_95'])
    bucket['effective_rank'].append(metrics['effective_rank'])
    bucket['summary_freqs_hz'].append(metrics['summary_freqs_hz'])
    bucket['mean_acf'].append(metrics['mean_acf'])
    bucket['freq_corr_upper'].append(metrics['freq_corr_upper'])
    return bucket


def finalize_summary_bucket(bucket):
    for key in ('acf_decay_s', 'pc1_var_explained', 'n_pcs_95', 'effective_rank'):
        bucket[key] = np.asarray(bucket[key], dtype=float)
        bucket[f'{key}_mean'], bucket[f'{key}_sem'] = mean_and_sem(bucket[key])

    bucket['summary_freqs_hz'] = np.vstack(bucket['summary_freqs_hz'])
    bucket['mean_acf'] = np.vstack(bucket['mean_acf'])
    bucket['freq_corr_upper'] = np.vstack(bucket['freq_corr_upper'])
    return bucket
