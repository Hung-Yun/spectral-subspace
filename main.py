#%%
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import io

from utils import fig_set, finish_plot

plt.figure()
plt.close()
fig_set(font_size=10, linewidth=0.8)

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

        trace = self.traces[start:stop, self.spec_channel_idx]
        self.spec_time, self.spec_freqs_hz, self.spec_power = get_spectrogram(
            trace,
            fs=self.fs,
            **spectrogram_kwargs,
        )
        self.spec_time += start_s
        return self.spec_time, self.spec_freqs_hz, self.spec_power

    def PCA(self):
        from sklearn.decomposition import PCA

        if self.spec_power is None:
            raise ValueError('Compute a channel spectrogram before running PCA.')
        self.pca = PCA().fit(self.spec_power.T)
        return self.pca


def get_psd(trace, fs, window_s, overlap_frac, window='hann'):
    from scipy import signal

    nperseg = min(int(window_s * fs), trace.shape[0])
    if nperseg < 1:
        raise ValueError('PSD window is too short for the provided trace.')

    noverlap = min(int(nperseg * overlap_frac), max(nperseg - 1, 0))
    freqs, psd = signal.welch(trace, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, axis=0)
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


DATA_PATH = os.path.join(
    'data',
    'neural',
    'EMU-0130_subj-YFA_task-WheelOfFortune_run-01_NSP-2_ds_lfp.mat',
)

mat = io.loadmat(DATA_PATH, squeeze_me=True, struct_as_record=False)
data = {key: value for key, value in mat.items() if not key.startswith('__')}
recording = ProcessedLFP(data)

#%% Preview a short LFP segment
PREVIEW_START_S = 0
PREVIEW_DURATION_S = 5
PREVIEW_CHANNELS = min(3, recording.traces.shape[1])

time = np.arange(recording.traces.shape[0]) / recording.fs
channel_sd = np.nanstd(recording.traces, axis=0)
channel_names = recording.channel_names[:PREVIEW_CHANNELS]
channel_ids = recording.channel_ids[:PREVIEW_CHANNELS]
preview_traces = recording.traces[:int(PREVIEW_DURATION_S * recording.fs), :PREVIEW_CHANNELS]
fs = preview_traces.shape[0] / PREVIEW_DURATION_S

start = int(PREVIEW_START_S * fs)
stop = int((PREVIEW_START_S + PREVIEW_DURATION_S) * fs)
offset = 4 * np.nanmedian(channel_sd[:PREVIEW_CHANNELS])

plt.figure(figsize=(3,3), dpi=300)
for channel_idx in range(PREVIEW_CHANNELS):
    plt.plot(
        time[start:stop],
        preview_traces[start:stop, channel_idx] + channel_idx * offset,
        lw=0.6,
        label=channel_names[channel_idx],
    )
plt.xlabel('Time (s)')
plt.ylabel('Amplitude + offset')
plt.title('EMU-0130 processed LFP preview')
plt.legend(frameon=False, loc='best', fontsize=5)
finish_plot()

#%% PSDs for all channels
PSD_WINDOW_S = 2.0
PSD_OVERLAP_FRAC = 0.5
PSD_MAX_HZ = 100

recording.compute_psd(window_s=PSD_WINDOW_S, overlap_frac=PSD_OVERLAP_FRAC)
channel_indices = np.arange(recording.traces.shape[1])
cmap = plt.cm.rainbow
norm = plt.Normalize(vmin=channel_indices[0], vmax=channel_indices[-1])
colors = cmap(norm(channel_indices))
plt.figure(figsize=(3,3), dpi=300)
for channel_idx, color in enumerate(colors):
    mask = recording.psd_f <= PSD_MAX_HZ
    plt.plot(recording.psd_f[mask], recording.psd[mask, channel_idx], lw=0.7, alpha=0.85, color=color)
plt.yscale('log')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD')
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=plt.gca(), label='Channel index')
cbar.ax.tick_params(labelsize=7) 
cbar.set_label('Channel index', size=7, weight='bold')
plt.title('PSD of all channels')
finish_plot()

#%% Spectrogram for one selected channel

spec_time, spec_freqs_hz, spec_power = recording.compute_spectrogram(
    channel=4,
    start_s=0,
    duration_s=60,
    freqs_hz=np.arange(1, 101, dtype=float),
    fwhm=0.5,
    wavelet_window_s=1.0,
)

display_step = max(int(round(recording.fs / 100)), 1)
plt.figure(figsize=(6, 3), dpi=300)
plt.pcolormesh(
    spec_time[::display_step],
    spec_freqs_hz,
    np.log(spec_power[:, ::display_step] + 1e-12),
    cmap='viridis',
    shading='auto',
)
plt.colorbar(label='Log power')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title(f'Spectrogram: {recording.channel_names[recording.spec_channel_idx]}')
finish_plot()


#%% PCA of the selected channel spectrogram
recording.PCA()

# print cumulative variance explained by the PCA components
cumulative_variance = np.cumsum(recording.pca.explained_variance_ratio_)
plt.figure(figsize=(3,3), dpi=300)
plt.plot(cumulative_variance[:30], lw=1)
plt.xlabel('Number of PCA components')
plt.ylabel('Cumulative variance explained')
plt.title('PCA of spectrogram power')
finish_plot()
# %% QC

class LFP_QC:
    """
    Lightweight QC helper that depends on an DS_LFP instance.
    """

    def __init__(self,lfp,window_s=1.0,
                 overlap_s=0.5, z_thresh=5.0, line_freq=60.0,
                 chunk_duration_s=30.0, max_psd_chunks=12, psd_nperseg_s=4.0,
                 down_sampled=True, new_fs=1000):
        
        self.lfp = lfp
        self.window_s = window_s
        self.overlap_s = overlap_s
        self.z_thresh = z_thresh
        self.line_freq = line_freq
        self.chunk_duration_s = chunk_duration_s
        self.max_psd_chunks = max_psd_chunks
        self.psd_nperseg_s = psd_nperseg_s
        self.new_fs = new_fs

        # Initial empty results that will be populated by QC methods
        self.channel_metrics = pd.DataFrame()
        self.interval_metrics = pd.DataFrame()
        self.bad_channel_ids = []
        self.bad_channel_names = []
        self.bad_intervals = pd.DataFrame(
            columns=['start_sample', 'end_sample', 'start_time', 'end_time', 'reason']
        )

        # Downsample for QC purpose
        if down_sampled:
            self.is_downsampled = True
            self.lfp_ds = downsample(self.lfp.raw_lfp, 
                                    fs=self.lfp.fs, 
                                    target_fs=self.new_fs, 
                                    lowpass_hz=400)
            
        else:
            self.is_downsampled = False
            self.lfp_ds = None

    @property
    def params(self):

        return {
            'window_s': self.window_s,
            'overlap_s': self.overlap_s,
            'z_thresh': self.z_thresh,
            'line_freq': self.line_freq,
            'chunk_duration_s': self.chunk_duration_s,
            'max_psd_chunks': self.max_psd_chunks,
            'psd_nperseg_s': self.psd_nperseg_s,
            'sampling_frequency': self.fs,
        }
    
    @property
    def fs(self):
        """sampling frequency"""
        return self.lfp.fs if not self.is_downsampled else self.new_fs
    
    def compute_channel_metrics(self):
        """
        Compute per-channel QC metrics such as RMS, variance, line noise.
        """

        n_channels = len(self.lfp.chosen_channel_ids)
        if n_channels == 0:
            raise ValueError('No neural channels were selected for QC.')



        return self

    # def compute_channel_metrics(self):
    #     """
    #     Compute per-channel QC metrics such as RMS, variance, line noise,
    #     clipping fraction, and peer correlation.
    #     """
    #     print(' - Computing channel-level QC metrics')
    #     fs = float(self.lfp.fs)
    #     n_samples = int(self.lfp.n_samples)
    #     n_channels = len(self.lfp.chosen_channel_ids)
    #     if n_channels == 0:
    #         raise ValueError('No neural channels were selected for QC.')

    #     sum_x = np.zeros(n_channels, dtype=np.float64)
    #     sum_x2 = np.zeros(n_channels, dtype=np.float64)
    #     sum_abs_dev = np.zeros(n_channels, dtype=np.float64)
    #     mins = np.full(n_channels, np.inf, dtype=np.float64)
    #     maxs = np.full(n_channels, -np.inf, dtype=np.float64)
    #     min_counts = np.zeros(n_channels, dtype=np.int64)
    #     max_counts = np.zeros(n_channels, dtype=np.int64)
    #     repeated_counts = np.zeros(n_channels, dtype=np.int64)
    #     prev_last = None
    #     psd_sum, freqs = None, None
    #     psd_count = 0

    #     chunk_frames = max(1, int(self.chunk_duration_s * fs))
    #     n_chunks = max(1, int(np.ceil(n_samples / chunk_frames)))
    #     psd_stride = max(1, int(np.ceil(n_chunks / self.max_psd_chunks)))

    #     for i, (_, _, traces) in enumerate(self.iter_lfp_chunks()):
    #         sum_x += traces.sum(axis=0, dtype=np.float64)
    #         sum_x2 += np.sum(traces * traces, axis=0, dtype=np.float64)
    #         mins = np.minimum(mins, traces.min(axis=0))
    #         maxs = np.maximum(maxs, traces.max(axis=0))
    #         if traces.shape[0] > 1:
    #             repeated_counts += np.sum(np.diff(traces, axis=0) == 0, axis=0)
    #         if prev_last is not None:
    #             repeated_counts += traces[0] == prev_last
    #         prev_last = traces[-1].copy()

    #         if i % psd_stride != 0:
    #             continue
    #         nperseg = min(int(self.psd_nperseg_s * fs), traces.shape[0])
    #         if nperseg < 8:
    #             continue
    #         freqs, psd = signal.welch(traces, fs=fs, axis=0, nperseg=nperseg)
    #         psd_sum = psd if psd_sum is None else psd_sum + psd
    #         psd_count += 1

    #     mean = sum_x / n_samples
    #     std = np.sqrt(np.maximum(sum_x2 / n_samples - mean ** 2, 0))
    #     rms = np.sqrt(sum_x2 / n_samples)
    #     peak_to_peak = maxs - mins

    #     for _, _, traces in self.iter_lfp_chunks():
    #         sum_abs_dev += np.sum(np.abs(traces - mean), axis=0, dtype=np.float64)
    #         min_counts += np.sum(traces == mins, axis=0)
    #         max_counts += np.sum(traces == maxs, axis=0)

    #     mean_abs_dev = sum_abs_dev / n_samples
    #     repeated_fraction = repeated_counts / max(n_samples - 1, 1)
    #     clipping_fraction = np.maximum(min_counts, max_counts) / n_samples

    #     if psd_sum is None or freqs is None or psd_count == 0:
    #         line_noise_ratio = np.full(n_channels, np.nan)
    #     else:
    #         mean_psd = psd_sum / psd_count
    #         band_mask = (freqs >= 1.0) & (freqs <= min(200.0, fs / 2.0))
    #         line_mask = band_mask & (np.abs(freqs - self.line_freq) <= 1.0)
    #         total_power = np.trapezoid(mean_psd[band_mask], freqs[band_mask], axis=0)
    #         line_power = np.trapezoid(mean_psd[line_mask], freqs[line_mask], axis=0)
    #         line_noise_ratio = np.divide(line_power, total_power, out=np.full(n_channels, np.nan), where=total_power > 0)

    #     metrics = self.lfp.chosen_chan.reset_index(drop=True).copy()
    #     metrics['n_samples'] = n_samples
    #     metrics['duration_s'] = n_samples / fs
    #     metrics['mean'] = mean
    #     metrics['std'] = std
    #     metrics['rms'] = rms
    #     metrics['mean_abs_dev'] = mean_abs_dev
    #     metrics['peak_to_peak'] = peak_to_peak
    #     metrics['repeated_fraction'] = repeated_fraction
    #     metrics['clipping_fraction'] = clipping_fraction
    #     metrics['line_noise_ratio'] = line_noise_ratio

    #     self.channel_metrics = metrics
    #     return self

    def detect_bad_channels(self):
        """
        Flag globally bad channels based on channel-level QC metrics.
        """
        return self

    def compute_interval_metrics(self):
        """
        Compute windowed QC metrics for detecting transient bad time intervals.
        """
        return self

    def detect_bad_intervals(self):
        """
        Flag bad time intervals from windowed QC metrics.
        """
        return self

    def update_lfp_metadata(self):
        """
        Write compact QC results back onto the linked LFP_processor instance.
        """
        return self

    def plot_channel_summary(self):
        """
        Plot summary figures for channel-level QC metrics and flags.
        """
        return self

    def plot_interval_summary(self):
        """
        Plot summary figures for bad-interval detection over time.
        """
        return self

    def plot_summary(self):
        """
        Convenience wrapper for generating the main QC summary figures.
        """
        self.plot_channel_summary()
        self.plot_interval_summary()
        return self

    def __repr__(self):
        return f'LFP_QC({self.lfp.filename[:-4]})'
