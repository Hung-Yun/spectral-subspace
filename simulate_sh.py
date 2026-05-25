#%%
import argparse
import csv
import os

print('Starting simulate_sh.py; loading analysis libraries...', flush=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(SCRIPT_DIR, '.cache', 'matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', os.path.join(SCRIPT_DIR, '.cache', 'xdg'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import signal
from sklearn.decomposition import PCA
from utils import fig_set, finish_plot


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


def get_freq_cov_from_power(power_by_freq_time, eps=1e-12):
    log_power = np.log(power_by_freq_time + eps)
    power_mean = log_power.mean(axis=1, keepdims=True)
    power_std = log_power.std(axis=1, keepdims=True)
    variable_rows = power_std.squeeze() > 0
    power_std[power_std == 0] = 1

    z_power = (log_power - power_mean) / power_std
    freq_corr = np.zeros((z_power.shape[0], z_power.shape[0]), dtype=float)
    if np.any(variable_rows):
        if np.sum(variable_rows) == 1:
            freq_corr[np.ix_(variable_rows, variable_rows)] = 1.0
        else:
            freq_corr[np.ix_(variable_rows, variable_rows)] = np.corrcoef(z_power[variable_rows])
    return z_power, freq_corr


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


class SimulationResult:
    def __init__(self, envelope_mode, sim_kwargs, spectrogram_kwargs):
        self.envelope_mode = envelope_mode
        self.trace, self.envelopes, _ = simulate_trace(**sim_kwargs, envelope_mode=envelope_mode)
        self.spec_time, self.spec_freqs_hz, self.spec_power = get_spectrogram(self.trace, **spectrogram_kwargs)
        self.z_power, self.freq_corr = get_freq_cov_from_power(self.spec_power)

    def PCA(self):
        self.pca = PCA().fit(self.spec_power.T)


def get_autocorr(trace, max_lag_samples):
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

    return dict(
        summary_freq_idx=summary_freq_idx,
        summary_freqs_hz=summary_freqs_hz,
        acf_lags_s=acf_lags_s,
        mean_acf=mean_acf,
        acf_decay_s=acf_decay_s,
        fluct_psd_f=fluct_psd_f,
        fluct_psd=fluct_psd,
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
        fluct_psd_f=None,
        acf_decay_s=[],
        pc1_var_explained=[],
        n_pcs_95=[],
        effective_rank=[],
        summary_freqs_hz=[],
        mean_acf=[],
        fluct_psd=[],
        freq_corr_upper=[],
    )


def finalize_summary_bucket(bucket):
    for key in ('acf_decay_s', 'pc1_var_explained', 'n_pcs_95', 'effective_rank'):
        bucket[key] = np.asarray(bucket[key], dtype=float)
        bucket[f'{key}_mean'], bucket[f'{key}_sem'] = mean_and_sem(bucket[key])

    bucket['summary_freqs_hz'] = np.vstack(bucket['summary_freqs_hz'])
    bucket['mean_acf'] = np.vstack(bucket['mean_acf'])
    bucket['fluct_psd'] = np.vstack(bucket['fluct_psd'])
    bucket['freq_corr_upper'] = np.vstack(bucket['freq_corr_upper'])
    return bucket


def write_summary_csv(sweep_summary, smooth_windows_s, envelope_modes, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'smooth_window_scalar_summary.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'envelope_mode', 'smooth_window_s', 'n_seeds',
            'acf_decay_s_mean', 'acf_decay_s_sem',
            'pc1_var_explained_mean', 'pc1_var_explained_sem',
            'n_pcs_95_mean', 'n_pcs_95_sem',
            'effective_rank_mean', 'effective_rank_sem',
        ])
        for envelope_mode in envelope_modes:
            for smooth_window_s in smooth_windows_s:
                summary = sweep_summary[envelope_mode][smooth_window_s]
                writer.writerow([
                    envelope_mode,
                    smooth_window_s,
                    summary['n_seeds'],
                    summary['acf_decay_s_mean'],
                    summary['acf_decay_s_sem'],
                    summary['pc1_var_explained_mean'],
                    summary['pc1_var_explained_sem'],
                    summary['n_pcs_95_mean'],
                    summary['n_pcs_95_sem'],
                    summary['effective_rank_mean'],
                    summary['effective_rank_sem'],
                ])
    print(f'Saved: {path}', flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Smooth-window sensitivity analysis for simulated spectrograms.')
    parser.add_argument('--n-seeds', type=int, default=10)
    parser.add_argument('--sim-fs', type=float, default=500)
    parser.add_argument('--sim-duration-s', type=float, default=600)
    parser.add_argument('--output-subdir', default='')
    parser.add_argument('--savefig', action='store_true', default=True)
    parser.add_argument('--no-savefig', dest='savefig', action='store_false')
    return parser.parse_args()


#%%
def main():
    args = parse_args()
    fig_set(font_size=10, linewidth=0.8)

    save_dir = os.path.join('plots', 'simulate', args.output_subdir)
    smooth_windows_s = np.array([0.5, 1, 2, 5, 10, 20, 40], dtype=float)
    sweep_seeds = np.arange(args.n_seeds)
    envelope_modes = ['ind', 'shared']

    sim_fs = args.sim_fs
    sim_duration_s = args.sim_duration_s
    sim_time = np.arange(0, sim_duration_s, 1 / sim_fs)
    sim_freqs_hz = np.array([12, 30, 70], dtype=float)
    spec_freqs_hz = np.arange(1, 101, dtype=float)

    sim_kwargs = dict(
        sim_time=sim_time,
        freqs_hz=sim_freqs_hz,
        base_amplitudes=np.array([30, 28, 25], dtype=float),
        phases_rad=np.array([0.0, 0.6, 1.1], dtype=float),
        envelope_scales=np.array([12, 8, 4], dtype=float),
    )
    spectrogram_kwargs = dict(fs=sim_fs, freqs_hz=spec_freqs_hz, fwhm=0.5, wavelet_window_s=1.0)

    print(
        f'Starting Python sweep: {len(smooth_windows_s)} windows x {args.n_seeds} seeds '
        f'x {len(envelope_modes)} modes; duration={sim_duration_s:g}s, fs={sim_fs:g}Hz',
        flush=True,
    )
    print(f'Output directory: {save_dir}', flush=True)

    sweep_summary = {
        envelope_mode: {smooth_window_s: init_summary_bucket() for smooth_window_s in smooth_windows_s}
        for envelope_mode in envelope_modes
    }

    for smooth_window_s in smooth_windows_s:
        print(f'Starting smooth_window_s={smooth_window_s:g}', flush=True)
        for seed in sweep_seeds:
            print(f'  Running seed={seed}', flush=True)
            sim_kwargs_sweep = {
                **sim_kwargs,
                'smooth_window_s': smooth_window_s,
                'rng': np.random.default_rng(seed),
            }
            for envelope_mode in envelope_modes:
                sim = SimulationResult(envelope_mode, sim_kwargs_sweep, spectrogram_kwargs)
                sim.PCA()
                metrics = summarize_simulation(sim)

                bucket = sweep_summary[envelope_mode][smooth_window_s]
                if bucket['acf_lags_s'] is None:
                    bucket['acf_lags_s'] = metrics['acf_lags_s']
                    bucket['fluct_psd_f'] = metrics['fluct_psd_f']
                bucket['n_seeds'] += 1
                bucket['acf_decay_s'].append(metrics['acf_decay_s'])
                bucket['pc1_var_explained'].append(metrics['pc1_var_explained'])
                bucket['n_pcs_95'].append(metrics['n_pcs_95'])
                bucket['effective_rank'].append(metrics['effective_rank'])
                bucket['summary_freqs_hz'].append(metrics['summary_freqs_hz'])
                bucket['mean_acf'].append(metrics['mean_acf'])
                bucket['fluct_psd'].append(metrics['fluct_psd'])
                bucket['freq_corr_upper'].append(metrics['freq_corr_upper'])

                del sim
        print(f'Finished smooth_window_s={smooth_window_s:g}', flush=True)

    for envelope_mode in envelope_modes:
        for smooth_window_s in smooth_windows_s:
            sweep_summary[envelope_mode][smooth_window_s] = finalize_summary_bucket(
                sweep_summary[envelope_mode][smooth_window_s]
            )

    write_summary_csv(sweep_summary, smooth_windows_s, envelope_modes, save_dir)

    metric_specs = [
        ('acf_decay_s', 'ACF decay (s)'),
        ('pc1_var_explained', 'PC1 variance explained'),
        ('n_pcs_95', 'PCs for 95% variance'),
        ('effective_rank', 'Effective rank'),
    ]
    mode_colors = {'ind': 'tab:blue', 'shared': 'tab:orange'}

    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.5), dpi=300)
    for ax, (metric_key, ylabel) in zip(axes.flat, metric_specs):
        for envelope_mode in envelope_modes:
            means = np.array(
                [sweep_summary[envelope_mode][window][f'{metric_key}_mean'] for window in smooth_windows_s],
                dtype=float,
            )
            sems = np.array(
                [sweep_summary[envelope_mode][window][f'{metric_key}_sem'] for window in smooth_windows_s],
                dtype=float,
            )
            ax.errorbar(
                smooth_windows_s,
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
    finish_plot('smooth_window_scalar_summaries', save_dir=save_dir, savefig=args.savefig)

    selected_windows_s = smooth_windows_s
    window_colors = plt.cm.viridis(np.linspace(0.15, 0.85, selected_windows_s.size))

    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5), dpi=300)
    for row_idx, envelope_mode in enumerate(envelope_modes):
        ax_acf = axes[row_idx, 0]
        ax_psd = axes[row_idx, 1]
        for color, smooth_window_s in zip(window_colors, selected_windows_s):
            summary = sweep_summary[envelope_mode][smooth_window_s]
            ax_acf.plot(
                summary['acf_lags_s'],
                summary['mean_acf'].mean(axis=0),
                lw=1,
                color=color,
                label=f'{smooth_window_s:g} s',
            )

            mean_fluct_psd = summary['fluct_psd'].mean(axis=0)
            psd_mask = summary['fluct_psd_f'] <= 2
            ax_psd.plot(
                summary['fluct_psd_f'][psd_mask],
                mean_fluct_psd[psd_mask],
                lw=1,
                color=color,
                label=f'{smooth_window_s:g} s',
            )

        ax_acf.set_xlim([0, 60])
        ax_acf.set_ylim([-0.1, 1.02])
        ax_acf.set_xlabel('Lag (s)')
        ax_acf.set_ylabel(f'{envelope_mode} mean ACF')

        ax_psd.set_xlabel('Fluctuation frequency (Hz)')
        ax_psd.set_ylabel(f'{envelope_mode} fluctuation PSD')
        ax_psd.set_yscale('log')
        ax_psd.set_xlim([0, 2])

    axes[0, 0].legend(frameon=False, loc='upper right', fontsize=7, title='Window')
    finish_plot('smooth_window_temporal_summaries', save_dir=save_dir, savefig=args.savefig)
    print('Finished smooth-window sensitivity analysis.', flush=True)


if __name__ == '__main__':
    main()
