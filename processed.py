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
