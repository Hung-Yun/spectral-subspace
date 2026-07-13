import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from spectral import get_psd
from utils import _get_repo_dir, finish_plot


def robust_zscore(values, eps=1e-12):
    """
    TODO: write docstring
    """
    values = np.asarray(values, dtype=float)
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < eps:
        return np.zeros_like(values, dtype=float)
    return (values - median) / scale


def _window_starts(n_samples, fs, window_s, step_s):
    window_n = max(int(round(window_s * fs)), 1)
    step_n = max(int(round(step_s * fs)), 1)
    if n_samples <= window_n:
        return np.array([0], dtype=int), window_n
    starts = np.arange(0, n_samples - window_n + 1, step_n, dtype=int)
    if starts.size == 0 or starts[-1] != n_samples - window_n:
        starts = np.append(starts, n_samples - window_n)
    return starts, window_n


def _merge_intervals(intervals, gap_s=0.0):
    if len(intervals) == 0:
        return []

    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [list(intervals[0])]

    for start_s, stop_s in intervals[1:]:
        if start_s <= merged[-1][1] + gap_s:
            merged[-1][1] = max(merged[-1][1], stop_s)
        else:
            merged.append([start_s, stop_s])
    return [(start_s, stop_s) for start_s, stop_s in merged]


class QC:

    """
    TODO: write docstring especially for each DEFAULT_CHANNEL_THRESHOLDS and DEFAULT_INTERVAL_THRESHOLDS
    """
    DEFAULT_CHANNEL_THRESHOLDS = {
        'rms_z_thresh': 5.0,
        'mad_z_thresh': 5.0,
        'ptp_z_thresh': 5.0,
        'kurtosis_z_thresh': 5.0,
        'flat_frac_thresh': 0.05,
        'nan_frac_thresh': 0.0,
        'line_noise_ratio_z_thresh': 5.0,
        'hf_ratio_z_thresh': 5.0,
        'corr_low_thresh': 0.1,
    }

    DEFAULT_INTERVAL_THRESHOLDS = {
        'window_s': 1.0,
        'step_s': 0.25,
        'global_rms_z_thresh': 6.0,
        'channel_frac_thresh': 0.3,
        'min_interval_s': 0.5,
        'merge_gap_s': 0.25,
    }

    def __init__(
        self,
        lfp,
        session_name=None,
        line_freq=60.0,
        line_bandwidth_hz=2.0,
        line_harmonics=(1, 2, 3),
        flat_abs_thresh=1e-6,
    ):
        self.lfp = lfp
        self.session_name = session_name or getattr(lfp, 'subject', None) or 'session'
        self.line_freq = float(line_freq)
        self.line_bandwidth_hz = float(line_bandwidth_hz)
        self.line_harmonics = tuple(line_harmonics)
        self.flat_abs_thresh = float(flat_abs_thresh)

        self.psd_f = None
        self.psd = None
        self.channel_metrics = None
        self.bad_channels = None
        self.bad_intervals = None
        self.interval_metrics = None

    def summary(self):
        n_samples, n_channels = self.lfp.traces.shape
        duration_s = n_samples / self.lfp.fs
        summary = {
            'session_name': self.session_name,
            'n_samples': int(n_samples),
            'n_channels': int(n_channels),
            'fs': float(self.lfp.fs),
            'duration_s': float(duration_s),
            'duration_min': float(duration_s / 60),
            'n_nan': int(np.isnan(self.lfp.traces).sum()),
            'n_inf': int(np.isinf(self.lfp.traces).sum()),
            'n_zero_var_channels': int(np.sum(np.nanstd(self.lfp.traces, axis=0) == 0)),
        }
        return summary

    def compute_psd(self, window_s=2.0, overlap_frac=0.5):
        self.psd_f, self.psd = get_psd(self.lfp.traces, fs=self.lfp.fs, window_s=window_s, overlap_frac=overlap_frac)
        return self.psd_f, self.psd

    def compute_channel_metrics(self, psd_window_s=2.0, psd_overlap_frac=0.5):
        traces = np.asarray(self.lfp.traces, dtype=float)
        n_samples, n_channels = traces.shape

        if self.psd_f is None or self.psd is None:
            self.compute_psd(window_s=psd_window_s, overlap_frac=psd_overlap_frac)

        channel_rms = np.sqrt(np.nanmean(traces**2, axis=0))
        channel_mad = stats.median_abs_deviation(traces, axis=0, nan_policy='omit', scale=1.0)
        channel_ptp = np.nanmax(traces, axis=0) - np.nanmin(traces, axis=0)
        channel_kurtosis = stats.kurtosis(traces, axis=0, fisher=True, bias=False, nan_policy='omit')
        nan_fraction = np.mean(~np.isfinite(traces), axis=0)
        flat_fraction = np.mean(np.abs(traces) < self.flat_abs_thresh, axis=0)

        channel_corr = np.corrcoef(np.nan_to_num(traces, nan=0.0), rowvar=False)
        if channel_corr.ndim == 0:
            median_corr_to_others = np.array([1.0])
        else:
            median_corr_to_others = np.zeros(n_channels, dtype=float)
            for channel_idx in range(n_channels):
                others = np.delete(channel_corr[channel_idx], channel_idx)
                if others.size == 0:
                    median_corr_to_others[channel_idx] = 1.0
                else:
                    median_corr_to_others[channel_idx] = np.nanmedian(others)

        line_noise_ratio = self._compute_line_noise_ratio(self.psd_f, self.psd)
        hf_ratio = self._compute_high_frequency_ratio(self.psd_f, self.psd)

        metrics = pd.DataFrame({
            'channel_idx': np.arange(n_channels),
            'channel_id': np.asarray(self.lfp.channel_ids).astype(str),
            'channel_name': np.asarray(self.lfp.channel_names).astype(str),
            'rms': channel_rms,
            'mad': channel_mad,
            'ptp': channel_ptp,
            'kurtosis': channel_kurtosis,
            'nan_fraction': nan_fraction,
            'flat_fraction': flat_fraction,
            'median_corr_to_others': median_corr_to_others,
            'line_noise_ratio': line_noise_ratio,
            'hf_ratio': hf_ratio,
        })

        for metric_name in ['rms', 'mad', 'ptp', 'kurtosis', 'line_noise_ratio', 'hf_ratio']:
            metrics[f'{metric_name}_robust_z'] = robust_zscore(metrics[metric_name].to_numpy())

        self.channel_metrics = metrics
        return self.channel_metrics

    def flag_bad_channels(self, thresholds=None):
        if self.channel_metrics is None:
            self.compute_channel_metrics()

        thresholds = {**self.DEFAULT_CHANNEL_THRESHOLDS, **(thresholds or {})}
        metrics = self.channel_metrics.copy()
        reasons = []

        for _, row in metrics.iterrows():
            channel_reasons = []
            if np.abs(row['rms_robust_z']) > thresholds['rms_z_thresh']:
                channel_reasons.append('rms_outlier')
            if np.abs(row['mad_robust_z']) > thresholds['mad_z_thresh']:
                channel_reasons.append('mad_outlier')
            if np.abs(row['ptp_robust_z']) > thresholds['ptp_z_thresh']:
                channel_reasons.append('ptp_outlier')
            if np.abs(row['kurtosis_robust_z']) > thresholds['kurtosis_z_thresh']:
                channel_reasons.append('kurtosis_outlier')
            if row['flat_fraction'] > thresholds['flat_frac_thresh']:
                channel_reasons.append('flat_signal')
            if row['nan_fraction'] > thresholds['nan_frac_thresh']:
                channel_reasons.append('nonfinite_samples')
            if row['line_noise_ratio_robust_z'] > thresholds['line_noise_ratio_z_thresh']:
                channel_reasons.append('line_noise')
            if row['hf_ratio_robust_z'] > thresholds['hf_ratio_z_thresh']:
                channel_reasons.append('hf_contamination')
            if row['median_corr_to_others'] < thresholds['corr_low_thresh']:
                channel_reasons.append('low_cohort_corr')
            reasons.append(channel_reasons)

        metrics['bad_reasons'] = reasons
        metrics['is_bad'] = metrics['bad_reasons'].apply(lambda x: len(x) > 0)
        self.channel_metrics = metrics
        self.bad_channels = metrics.loc[metrics['is_bad']].copy()
        return self.bad_channels

    def compute_interval_metrics(self, window_s=1.0, step_s=0.25):
        traces = np.asarray(self.lfp.traces, dtype=float)
        fs = float(self.lfp.fs)
        starts, window_n = _window_starts(traces.shape[0], fs, window_s, step_s)

        rows = []
        for start in starts:
            stop = min(start + window_n, traces.shape[0])
            window = traces[start:stop]
            channel_rms = np.sqrt(np.nanmean(window**2, axis=0))
            global_rms = np.nanmedian(channel_rms)
            rows.append({
                'start_sample': int(start),
                'stop_sample': int(stop),
                'start_s': float(start / fs),
                'stop_s': float(stop / fs),
                'global_rms': float(global_rms),
                'max_abs': float(np.nanmax(np.abs(window))),
                'channel_rms_median': float(np.nanmedian(channel_rms)),
                'channel_rms_max': float(np.nanmax(channel_rms)),
            })

        interval_metrics = pd.DataFrame(rows)
        interval_metrics['global_rms_robust_z'] = robust_zscore(interval_metrics['global_rms'].to_numpy())

        channel_rms_by_window = []
        for start in interval_metrics['start_sample'].to_numpy(dtype=int):
            stop = min(start + window_n, traces.shape[0])
            window = traces[start:stop]
            channel_rms = np.sqrt(np.nanmean(window**2, axis=0))
            channel_rms_by_window.append(channel_rms)
        channel_rms_by_window = np.asarray(channel_rms_by_window, dtype=float)

        channel_baseline = np.nanmedian(channel_rms_by_window, axis=0)
        channel_scale = 1.4826 * np.nanmedian(np.abs(channel_rms_by_window - channel_baseline), axis=0)
        channel_scale[channel_scale == 0] = 1.0
        channel_rms_z = (channel_rms_by_window - channel_baseline) / channel_scale

        interval_metrics['channel_frac_high_rms'] = np.mean(channel_rms_z > 5.0, axis=1)
        self.interval_metrics = interval_metrics
        return self.interval_metrics

    def detect_bad_intervals(self, thresholds=None):
        thresholds = {**self.DEFAULT_INTERVAL_THRESHOLDS, **(thresholds or {})}

        if self.interval_metrics is None:
            self.compute_interval_metrics(
                window_s=thresholds['window_s'],
                step_s=thresholds['step_s'],
            )

        interval_metrics = self.interval_metrics.copy()
        mask = (
            (interval_metrics['global_rms_robust_z'] > thresholds['global_rms_z_thresh'])
            | (interval_metrics['channel_frac_high_rms'] > thresholds['channel_frac_thresh'])
        )

        candidate_intervals = [
            (row.start_s, row.stop_s)
            for row in interval_metrics.loc[mask].itertuples()
            if (row.stop_s - row.start_s) >= thresholds['min_interval_s']
        ]
        merged = _merge_intervals(candidate_intervals, gap_s=thresholds['merge_gap_s'])

        rows = []
        for start_s, stop_s in merged:
            interval_subset = interval_metrics[
                (interval_metrics['start_s'] < stop_s) & (interval_metrics['stop_s'] > start_s)
            ]
            rows.append({
                'start_s': start_s,
                'stop_s': stop_s,
                'duration_s': stop_s - start_s,
                'max_global_rms_z': float(interval_subset['global_rms_robust_z'].max()),
                'max_channel_frac_high_rms': float(interval_subset['channel_frac_high_rms'].max()),
            })

        self.bad_intervals = pd.DataFrame(rows)
        return self.bad_intervals

    def plot_trace_preview(self, start_s=0.0, dur_s=10.0, n_channels=10, filename=None, save_dir=None):
        traces = np.asarray(self.lfp.traces, dtype=float)
        n_channels = min(int(n_channels), traces.shape[1])
        start = max(int(round(start_s * self.lfp.fs)), 0)
        stop = min(int(round((start_s + dur_s) * self.lfp.fs)), traces.shape[0])
        if start >= stop:
            raise ValueError('Requested trace window is outside the recording.')

        preview = traces[start:stop, :n_channels]
        time = np.arange(start, stop) / self.lfp.fs
        channel_sd = np.nanstd(preview, axis=0)
        offset = 4 * np.nanmedian(channel_sd[channel_sd > 0]) if np.any(channel_sd > 0) else 1.0

        plt.figure(figsize=(8, 4), dpi=200)
        for channel_idx in range(n_channels):
            plt.plot(
                time,
                preview[:, channel_idx] + channel_idx * offset,
                lw=0.6,
                label=str(self.lfp.channel_names[channel_idx]),
            )
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude + offset')
        plt.title(f'LFP trace preview: {self.session_name}')
        plt.legend(frameon=False, fontsize=6, ncol=2, loc='upper right')
        finish_plot(filename=filename, save_dir=save_dir, savefig=filename is not None, show=filename is None)

    def plot_psd(self, window_s=2.0, overlap_frac=0.5, max_hz=150.0, filename=None, save_dir=None):
        if self.psd_f is None or self.psd is None:
            self.compute_psd(window_s=window_s, overlap_frac=overlap_frac)

        mask = self.psd_f <= max_hz
        plt.figure(figsize=(6, 4), dpi=200)
        for channel_idx in range(self.psd.shape[1]):
            plt.plot(self.psd_f[mask], self.psd[mask, channel_idx], color='0.75', lw=0.6)
        plt.plot(self.psd_f[mask], np.nanmedian(self.psd[mask], axis=1), color='k', lw=1.5)
        for harmonic in self.line_harmonics:
            line_hz = harmonic * self.line_freq
            if line_hz <= max_hz:
                plt.axvline(line_hz, color='tab:red', ls='--', lw=0.8, alpha=0.8)
        plt.yscale('log')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD')
        plt.title(f'PSD QC: {self.session_name}')
        finish_plot(filename=filename, save_dir=save_dir, savefig=filename is not None, show=filename is None)

    def plot_channel_metrics(self, metrics=('rms_robust_z', 'line_noise_ratio_robust_z', 'hf_ratio_robust_z'),
                             filename=None, save_dir=None):
        if self.channel_metrics is None:
            self.compute_channel_metrics()

        metrics = list(metrics)
        fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 2.2 * len(metrics)), dpi=200, sharex=True)
        axes = np.atleast_1d(axes)
        x = np.arange(self.channel_metrics.shape[0])

        for ax, metric_name in zip(axes, metrics):
            ax.bar(x, self.channel_metrics[metric_name].to_numpy(), color='0.35', width=0.8)
            ax.axhline(0, color='k', lw=0.8)
            ax.set_ylabel(metric_name)

        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(self.channel_metrics['channel_name'].to_list(), rotation=90)
        axes[0].set_title(f'Channel QC metrics: {self.session_name}')
        finish_plot(filename=filename, save_dir=save_dir, savefig=filename is not None, show=filename is None)

    def plot_artifact_timeline(self, filename=None, save_dir=None):
        if self.bad_intervals is None:
            self.detect_bad_intervals()

        duration_s = self.lfp.traces.shape[0] / self.lfp.fs
        plt.figure(figsize=(8, 1.8), dpi=200)
        plt.hlines(1, 0, duration_s, color='0.75', lw=4)
        for row in self.bad_intervals.itertuples():
            plt.hlines(1, row.start_s, row.stop_s, color='tab:red', lw=6)
        plt.xlim(0, duration_s)
        plt.yticks([])
        plt.xlabel('Time (s)')
        plt.title(f'Artifact timeline: {self.session_name}')
        finish_plot(filename=filename, save_dir=save_dir, savefig=filename is not None, show=filename is None)

    def save_outputs(self, plot_subdir='plots/lfp_qc', stats_subdir='stats/lfp_qc'):
        repo_dir = _get_repo_dir()
        plot_dir = os.path.join(repo_dir, plot_subdir)
        stats_dir = os.path.join(repo_dir, stats_subdir)
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(stats_dir, exist_ok=True)

        if self.channel_metrics is not None:
            self.channel_metrics.to_csv(
                os.path.join(stats_dir, f'{self.session_name}_channel_metrics.csv'),
                index=False,
            )

        if self.bad_intervals is not None:
            self.bad_intervals.to_csv(
                os.path.join(stats_dir, f'{self.session_name}_bad_intervals.csv'),
                index=False,
            )

        self.plot_trace_preview(filename=f'{self.session_name}_trace_preview', save_dir=plot_dir)
        self.plot_psd(filename=f'{self.session_name}_psd', save_dir=plot_dir)
        self.plot_channel_metrics(filename=f'{self.session_name}_channel_metrics', save_dir=plot_dir)
        self.plot_artifact_timeline(filename=f'{self.session_name}_artifact_timeline', save_dir=plot_dir)

    def _compute_line_noise_ratio(self, freqs_hz, psd):
        ratios = np.zeros(psd.shape[1], dtype=float)
        for channel_idx in range(psd.shape[1]):
            channel_psd = psd[:, channel_idx]
            numerators = []
            denominators = []
            for harmonic in self.line_harmonics:
                center_hz = harmonic * self.line_freq
                band_mask = np.abs(freqs_hz - center_hz) <= self.line_bandwidth_hz / 2
                surround_mask = (
                    (np.abs(freqs_hz - center_hz) > self.line_bandwidth_hz / 2)
                    & (np.abs(freqs_hz - center_hz) <= self.line_bandwidth_hz * 2)
                )
                if np.any(band_mask):
                    numerators.append(np.nanmean(channel_psd[band_mask]))
                if np.any(surround_mask):
                    denominators.append(np.nanmean(channel_psd[surround_mask]))
            numerator = np.nanmean(numerators) if len(numerators) > 0 else np.nan
            denominator = np.nanmean(denominators) if len(denominators) > 0 else np.nan
            if not np.isfinite(denominator) or denominator <= 0:
                ratios[channel_idx] = 0.0
            else:
                ratios[channel_idx] = numerator / denominator
        return ratios

    def _compute_high_frequency_ratio(self, freqs_hz, psd):
        nyquist = self.lfp.fs / 2
        hf_low = min(100.0, max(0.0, nyquist * 0.6))
        hf_high = min(0.9 * nyquist, nyquist)
        lf_low = 1.0
        lf_high = min(40.0, nyquist)

        hf_mask = (freqs_hz >= hf_low) & (freqs_hz <= hf_high)
        lf_mask = (freqs_hz >= lf_low) & (freqs_hz <= lf_high)

        hf_power = np.nanmean(psd[hf_mask], axis=0) if np.any(hf_mask) else np.zeros(psd.shape[1], dtype=float)
        lf_power = np.nanmean(psd[lf_mask], axis=0) if np.any(lf_mask) else np.ones(psd.shape[1], dtype=float)
        lf_power[lf_power <= 0] = 1.0
        return hf_power / lf_power
