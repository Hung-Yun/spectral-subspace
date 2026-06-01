#%%
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import io

from spectral import get_power_cov, get_psd, get_spectrogram
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

    def PCA(self):
        from sklearn.decomposition import PCA

        if self.spec_power is None:
            raise ValueError('Compute a channel spectrogram before running PCA.')
        self.pca = PCA().fit(self.spec_power.T)
        return self.pca
    
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

def phase_randomize_trace(trace, rng):
    """Preserve trace PSD while randomizing temporal phase structure."""
    centered = np.asarray(trace, dtype=float) - np.mean(trace)
    trace_fft = np.fft.rfft(centered)
    randomized_fft = trace_fft.copy()

    # DC and, for even-length traces, Nyquist must remain real-valued.
    randomizable = slice(1, -1 if centered.size % 2 == 0 else None)
    randomized_fft[randomizable] = np.abs(trace_fft[randomizable]) * np.exp(
        1j * rng.uniform(-np.pi, np.pi, randomized_fft[randomizable].size)
    )
    return np.fft.irfft(randomized_fft, n=centered.size) + np.mean(trace)


DATA_PATH = os.path.join(
    'data',
    'neural',
    'EMU-0130_subj-YFA_task-WheelOfFortune_run-01_NSP-2_ds_lfp.mat',
)

mat = io.loadmat(DATA_PATH, squeeze_me=True, struct_as_record=False)
data = {key: value for key, value in mat.items() if not key.startswith('__')}
recording = ProcessedLFP(data)

#%% PSDs for all channels
recording.plot_psd(window_s=1.0, overlap_frac=0.5, max_hz=100)

#%% Spectrogram for one selected channel

SPECTROGRAM_KWARGS = dict(
    freqs_hz=np.arange(1, 101, dtype=float),
    fwhm=0.5,
    wavelet_window_s=1.0,
)
spec_time, spec_freqs_hz, spec_power = recording.compute_spectrogram(
    channel=0,
    start_s=0,
    duration_s=350,
    **SPECTROGRAM_KWARGS,
)

display_step = max(int(round(recording.fs / 100)), 1)
plt.figure(figsize=(6, 3), dpi=300)
plt.pcolormesh(
    spec_time[::display_step],
    spec_freqs_hz,
    np.log(spec_power[:, ::display_step] + 1e-12),
    cmap='viridis',
    shading='auto',
    vmin=-10,
    vmax=10,
)
plt.colorbar(label='Log power')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title(f'Spectrogram: {recording.channel_names[recording.spec_channel_idx]}')
finish_plot()


#%% PCA of the selected channel spectrogram trace-level phase-randomized PCA null
from sklearn.decomposition import PCA

recording.PCA()

# PCA centers each frequency feature before fitting.
cumulative_variance = np.cumsum(recording.pca.explained_variance_ratio_)
z_power, _ = get_power_cov(recording.spec_power)
z_pca = PCA().fit(z_power.T)
z_cumulative_variance = np.cumsum(z_pca.explained_variance_ratio_)

N_NULL_SURROGATES = 100
NULL_SEED = 0
rng = np.random.default_rng(NULL_SEED)
null_cumulative_variance = np.empty((N_NULL_SURROGATES, recording.spec_power.shape[0]))
z_null_cumulative_variance = np.empty_like(null_cumulative_variance)

for null_idx in range(N_NULL_SURROGATES):
    if null_idx % 10 == 0:
        print(f'Running phase-randomized surrogate {null_idx + 1}/{N_NULL_SURROGATES}')
    surrogate_trace = phase_randomize_trace(recording.spec_trace, rng)
    _, _, surrogate_power = get_spectrogram(
        surrogate_trace,
        fs=recording.fs,
        **SPECTROGRAM_KWARGS,
    )
    surrogate_pca = PCA().fit(surrogate_power.T)
    null_cumulative_variance[null_idx] = np.cumsum(surrogate_pca.explained_variance_ratio_)
    surrogate_z_power, _ = get_power_cov(surrogate_power)
    surrogate_z_pca = PCA().fit(surrogate_z_power.T)
    z_null_cumulative_variance[null_idx] = np.cumsum(surrogate_z_pca.explained_variance_ratio_)

null_median = np.median(null_cumulative_variance, axis=0)
null_lower, null_upper = np.percentile(null_cumulative_variance, [2.5, 97.5], axis=0)
pc1_p_value = (np.sum(null_cumulative_variance[:, 0] >= cumulative_variance[0]) + 1) / (N_NULL_SURROGATES + 1)
print(f'Observed PC1 variance: {cumulative_variance[0]:.3f}; phase-null p = {pc1_p_value:.3f}')

N_PCS_TO_PLOT = recording.spec_power.shape[0]
component_numbers = np.arange(1, N_PCS_TO_PLOT + 1)
plt.figure(figsize=(3,3), dpi=300)
plt.fill_between(
    component_numbers,
    null_lower[:N_PCS_TO_PLOT],
    null_upper[:N_PCS_TO_PLOT],
    color='0.85',
    label='Phase-null 95% interval',
)
plt.plot(component_numbers, cumulative_variance[:N_PCS_TO_PLOT], lw=1, color='r', label='Original')
plt.plot(component_numbers, null_median[:N_PCS_TO_PLOT], lw=1, color='k', ls='--', label='Phase-null median')
plt.xlabel('Number of PCA components')
plt.ylabel('Cumulative variance explained')
plt.title('PCA of spectrogram power')
plt.legend(frameon=False, loc='best', fontsize=7)
finish_plot()

#%% Z-scored log-power PCA with the same phase-randomized null
z_null_median = np.median(z_null_cumulative_variance, axis=0)
z_null_lower, z_null_upper = np.percentile(z_null_cumulative_variance, [2.5, 97.5], axis=0)
z_pc1_p_value = (np.sum(z_null_cumulative_variance[:, 0] >= z_cumulative_variance[0]) + 1) / (N_NULL_SURROGATES + 1)

raw_n_pcs_95 = np.searchsorted(cumulative_variance, 0.95) + 1
z_n_pcs_95 = np.searchsorted(z_cumulative_variance, 0.95) + 1
raw_null_n_pcs_95 = np.array([np.searchsorted(curve, 0.95) + 1 for curve in null_cumulative_variance])
z_null_n_pcs_95 = np.array([np.searchsorted(curve, 0.95) + 1 for curve in z_null_cumulative_variance])
print(
    'PCs for 95% variance: '
    f'raw={raw_n_pcs_95} (null median={np.median(raw_null_n_pcs_95):.0f}); '
    f'z-scored log-power={z_n_pcs_95} (null median={np.median(z_null_n_pcs_95):.0f})'
)
print(f'Z-scored log-power PC1 variance: {z_cumulative_variance[0]:.3f}; phase-null p = {z_pc1_p_value:.3f}')

plt.figure(figsize=(3,3), dpi=300)
plt.fill_between(
    component_numbers,
    z_null_lower[:N_PCS_TO_PLOT],
    z_null_upper[:N_PCS_TO_PLOT],
    color='0.85',
    label='Phase-null 95% interval',
)
plt.plot(component_numbers, z_cumulative_variance[:N_PCS_TO_PLOT], lw=1, color='r', label='Original')
plt.plot(component_numbers, z_null_median[:N_PCS_TO_PLOT], lw=1, color='k', ls='--', label='Phase-null median')
plt.xlabel('Number of PCA components')
plt.ylabel('Cumulative variance explained')
plt.title('PCA of z-scored log-power')
plt.legend(frameon=False, loc='best', fontsize=7)
finish_plot()

#%% Plot the first couple of PCs

fig, ax = plt.subplots(1,5, figsize=(15,3), dpi=300)

for i in range(5):
    ax[i].plot(z_pca.components_[i], color='k')
    ax[i].set_title(f'PC {i + 1} ({z_pca.explained_variance_ratio_[i] * 100:.1f}% var)')
    ax[i].set_xlabel('Frequency bin')
    ax[i].set_ylabel('Component weight')
plt.tight_layout()
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
