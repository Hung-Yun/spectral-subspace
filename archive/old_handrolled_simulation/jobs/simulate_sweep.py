#%%
import argparse
import csv
import os
import sys

"""

This script performs a sensitivity analysis of spectrogram summary
metrics to the choice of smoothing window. It simulates spectrograms 
with known properties, applies different smoothing windows, and evaluates 
how the summary metrics change. The results are saved to CSV and plotted.

This script was meant to be implemented on Wrangell with a bash script in /jobs folder.
See /jobs/simulate_sweep.sh for the intended usage.

"""

print('Starting simulate_sweep.py; loading analysis libraries...', flush=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

os.environ.setdefault('MPLCONFIGDIR', os.path.join(REPO_DIR, '.cache', 'matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', os.path.join(REPO_DIR, '.cache', 'xdg'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from archive.old_handrolled_simulation.simulate_core import generate_oscillation_trace
from spectral import get_autocorr, get_power_cov, get_spectrogram
from utils import fig_set, finish_plot


class SimulationResult:
    def __init__(self, envelope_mode, sim_kwargs, spectrogram_kwargs):
        self.envelope_mode = envelope_mode
        self.trace, _ = generate_oscillation_trace(**sim_kwargs, envelope_mode=envelope_mode)
        self.spec_time, self.spec_freqs_hz, self.spec_power = get_spectrogram(self.trace, **spectrogram_kwargs)
        self.z_power, self.freq_corr = get_power_cov(self.spec_power)

    def PCA(self):
        self.pca = PCA().fit(self.spec_power.T)


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
                update_summary_bucket(bucket, metrics)

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

    fig, axes = plt.subplots(2, 1, figsize=(4, 5.5), dpi=300)
    for row_idx, envelope_mode in enumerate(envelope_modes):
        ax_acf = axes[row_idx]
        for color, smooth_window_s in zip(window_colors, selected_windows_s):
            summary = sweep_summary[envelope_mode][smooth_window_s]
            ax_acf.plot(
                summary['acf_lags_s'],
                summary['mean_acf'].mean(axis=0),
                lw=1,
                color=color,
                label=f'{smooth_window_s:g} s',
            )

        ax_acf.set_xlim([0, 60])
        ax_acf.set_ylim([-0.1, 1.02])
        ax_acf.set_xlabel('Lag (s)')
        ax_acf.set_ylabel(f'{envelope_mode} mean ACF')

    axes[0].legend(frameon=False, loc='upper right', fontsize=7, title='Window')
    finish_plot('smooth_window_temporal_summaries', save_dir=save_dir, savefig=args.savefig)
    print('Finished smooth-window sensitivity analysis.', flush=True)


if __name__ == '__main__':
    main()
