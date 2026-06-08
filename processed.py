import numpy as np
from spectral import get_psd, get_spectrogram
from utils import finish_plot
import matplotlib.pyplot as plt


def _maybe_scalar(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.item() if value.size == 1 else value
    if isinstance(value, bytes):
        return value.decode()
    return value


class ProcessedLFP:
    def __init__(self, data):
        """
        Processed LFP recording loaded from a downsampled MAT file.
        """
        self.traces = np.asarray(data['lfp_ds'], dtype=float)
        self.fs = float(data['fs'])
        self.source_fs = float(data['source_fs'])
        self.subject = _maybe_scalar(data.get('subject'))
        self.emu_id = _maybe_scalar(data.get('emu_id'))
        self.task = _maybe_scalar(data.get('task'))
        self.channel_ids = np.atleast_1d(data['channel_ids']).astype(str)
        self.channel_names = np.atleast_1d(data['channel_names']).astype(str)
        self.regions = np.atleast_1d(data['regions']).astype(str)
        self.psd_f = None
        self.psd = None
        self.spec_time = None
        self.spec_freqs_hz = None
        self.spec_power = None
        self.spec_trace = None
        self.spec_channel_idx = None

    def summary(self):
        n_samples, n_channels = self.traces.shape
        duration_s = n_samples / self.fs
        print(f'LFP Summary:')
        print(f'  Regions: {", ".join(self.regions)}')
        print(f'  LFP shape: {n_samples:,} samples x {n_channels} channels')
        print(f'  Sampling rate: {self.fs:g} Hz (source: {self.source_fs:g} Hz)')
        print(f'  Duration: {duration_s:.2f} s ({duration_s / 60:.2f} min)')
        print('Channels:')
        for channel_id, channel_name in zip(self.channel_ids, self.channel_names):
            print(f'  {channel_id}: {channel_name}')

    def get_channel_idx(self, channel):
        if isinstance(channel, (int, np.integer)):
            if channel < 0 or channel >= self.traces.shape[1]:
                raise IndexError(f'Channel index {channel} is out of bounds.')
            return int(channel)

        channel = str(channel)
        matches = np.where((self.channel_ids == channel) | (self.channel_names == channel))[0]
        if matches.size != 1:
            raise ValueError(f'Expected one matching channel for {channel!r}; found {matches.size}.')
        return int(matches[0])

    def compute_psd(self, **psd_kwargs):
        self.psd_f, self.psd = get_psd(self.traces, fs=self.fs, **psd_kwargs)
        return self.psd_f, self.psd

    def compute_spectrogram(self, channel, start_s=0, duration_s=None, **spectrogram_kwargs):
        self.spec_channel_idx = self.get_channel_idx(channel)
        start = int(round(start_s * self.fs))
        stop = self.traces.shape[0] if duration_s is None else int(round((start_s + duration_s) * self.fs))
        stop = min(stop, self.traces.shape[0])
        if start < 0 or start >= stop:
            raise ValueError('Requested spectrogram window is outside the recording.')

        self.spec_trace = self.traces[start:stop, self.spec_channel_idx]
        self.spec_time, self.spec_freqs_hz, self.spec_power = get_spectrogram(
            self.spec_trace,
            fs=self.fs,
            **spectrogram_kwargs,
        )
        self.spec_time += start_s
        return self.spec_time, self.spec_freqs_hz, self.spec_power

    def transform_data(self, source='traces', transform='raw', eps=1e-12):
        if source == 'traces':
            data = np.asarray(self.traces, dtype=float)
            data = data.T if data.ndim == 2 else np.atleast_2d(data)
        elif source == 'spec_power':
            if self.spec_power is None:
                raise ValueError('Must compute spectrogram before requesting spec_power.')
            data = np.atleast_2d(np.asarray(self.spec_power, dtype=float))
        elif source == 'psd':
            if self.psd is None:
                raise ValueError('Must compute PSD before requesting psd.')
            data = np.atleast_2d(np.asarray(self.psd, dtype=float))
        else:
            raise ValueError(f'Unsupported source {source!r}.')

        def zscore_rows(matrix):
            row_mean = np.nanmean(matrix, axis=1, keepdims=True)
            row_std = np.nanstd(matrix, axis=1, keepdims=True)
            row_std[row_std == 0] = 1
            return (matrix - row_mean) / row_std

        def log_rows(matrix):
            row_min = np.nanmin(matrix, axis=1, keepdims=True)
            shift = np.where(row_min <= 0, -row_min + eps, 0.0)
            return np.log(matrix + shift + eps)

        if transform == 'raw':
            return data
        if transform == 'zscore':
            return zscore_rows(data)
        if transform == 'log':
            return log_rows(data)
        if transform == 'log_zscore':
            return zscore_rows(log_rows(data))
        if transform == 'zscore_log':
            return log_rows(zscore_rows(data))
        raise ValueError(f'Unsupported transform {transform!r}.')

    def compute_corr(self, source='spec_power', transform='raw'):
        data = self.transform_data(source=source, transform=transform)
        variable_rows = np.nanstd(data, axis=1) > 0
        corr = np.zeros((data.shape[0], data.shape[0]), dtype=float)

        if np.any(variable_rows):
            if np.sum(variable_rows) == 1:
                corr[np.ix_(variable_rows, variable_rows)] = 1.0
            else:
                corr[np.ix_(variable_rows, variable_rows)] = np.corrcoef(data[variable_rows])
        return corr

    def plot_lfp_traces(self, start_s, dur_s, n_channels=5):
        n_channels = min(int(n_channels), self.traces.shape[1])
        start = int(round(start_s * self.fs))
        stop = min(int(round((start_s + dur_s) * self.fs)), self.traces.shape[0])
        if start < 0 or start >= stop:
            raise ValueError('Requested trace window is outside the recording.')

        time = np.arange(self.traces.shape[0]) / self.fs
        channel_sd = np.nanstd(self.traces, axis=0)
        channel_names = self.channel_names[:n_channels]
        preview_traces = self.traces[start:stop, :n_channels]
        preview_time = time[start:stop]
        offset = 4 * np.nanmedian(channel_sd[:n_channels])

        plt.figure(figsize=(3,3), dpi=300)
        for channel_idx in range(n_channels):
            plt.plot(
                preview_time,
                preview_traces[:, channel_idx] + channel_idx * offset,
                lw=0.6,
                label=channel_names[channel_idx],
            )
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude + offset')
        plt.title('LFP traces (preview)')
        plt.legend(frameon=False, loc='best', fontsize=5)
        finish_plot()

    def plot_psd(self, window_s, overlap_frac, max_hz=100):
        self.compute_psd(window_s=window_s, overlap_frac=overlap_frac)
        mask = self.psd_f <= max_hz
        plt.figure(figsize=(3,3), dpi=300)
        for channel_idx in range(self.traces.shape[1]):
            plt.plot(self.psd_f[mask], self.psd[mask, channel_idx], lw=0.7, alpha=0.85,
                     color=plt.cm.viridis(channel_idx / self.traces.shape[1]))
        plt.yscale('log')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD')
        finish_plot()
