import numpy as np
from spectral import get_psd, get_spectrogram
from utils import finish_plot
import matplotlib.pyplot as plt

class ProcessedLFP:
    def __init__(self, data):
        """
        Processed LFP recording loaded from a downsampled MAT file.
        """
        self.traces = np.asarray(data['lfp_ds'], dtype=float)
        self.fs = float(data['fs'])
        self.source_fs = float(data['source_fs'])
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
        self.freq_corr = None

        # Derived data
        self.traces_z_scored = None
        self.traces_log = None
        self.traces_z_scored_log = None
        self.get_traces()

    def get_traces(self):
        self.traces_z_scored = (self.traces - np.nanmean(self.traces, axis=0)) / np.nanstd(self.traces, axis=0)
        self.traces_log = np.log(self.traces_z_scored - np.nanmin(self.traces_z_scored) + 1e-12)
        self.traces_z_scored_log = (self.traces_log - np.nanmean(self.traces_log, axis=0)) / np.nanstd(self.traces_log, axis=0)

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
    
    def compute_freq_corr(self):
        if self.spec_power is not None:
            self.freq_corr = np.corrcoef(np.log(self.spec_power + 1e-12))
        else:
            raise Exception('Must compute spectrogram before computing frequency correlation.')
        return self.freq_corr
    

    def plot_lfp_traces(self, start_s, dur_s, n_channels=5):
                
        time = np.arange(self.traces.shape[0]) / self.fs
        channel_sd = np.nanstd(self.traces, axis=0)
        channel_names = self.channel_names[:n_channels]
        preview_traces = self.traces[:int(dur_s * self.fs), :n_channels]
        fs = preview_traces.shape[0] / dur_s

        start = int(start_s * fs)
        stop = int((start_s + dur_s) * fs)
        offset = 4 * np.nanmedian(channel_sd[:n_channels])

        plt.figure(figsize=(3,3), dpi=300)
        for channel_idx in range(n_channels):
            plt.plot(
                time[start:stop],
                preview_traces[start:stop, channel_idx] + channel_idx * offset,
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