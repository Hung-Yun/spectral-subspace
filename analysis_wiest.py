# %% Imports and paths
from pathlib import Path
from warnings import warn
import gc

import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
from scipy import io, signal
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from spectral import get_spectrogram, get_psd
from specparam import SpectralModel
from decomposition import FAResults

from utils import *
from wiest2023_meta import PAIR_SPECS
REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = REPO_ROOT / "data" / "wiest"
OFF_DIR = DATA_ROOT / "aperiodic_stn_meds_off"
ON_PARENT = DATA_ROOT / "aperiodic_stn_meds_on"

on_candidates = [ON_PARENT]
if ON_PARENT.exists():
    on_candidates.extend(path for path in ON_PARENT.iterdir() if path.is_dir())
ON_DIR = next((path for path in on_candidates if any(path.glob("*_ON.mat"))), ON_PARENT)

DETREND_CHUNK_SIZE = 1_000_000
NOTCH_QUALITY_FACTOR = 30.0
PSD_WINDOW_S = 1.0
SPECPARAM_FIT_RANGE = (40.0, 90.0)
MAINS_NOISE_RANGE = (47.0, 54.0)
SPECPARAM_SETTINGS = {
    "aperiodic_mode": "fixed",
    "peak_width_limits": (2, 12),
    "max_n_peaks": np.inf,
    "min_peak_height": 0,
    "peak_threshold": 2,
    "verbose": False,
}

frequency_bands = {
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (13, 35),
    "low gamma": (35, 70),
    "high gamma": (70, 100),
}

fig_set(font_size=8, linewidth=0.8)
SAVEFIG = True
SHOWFIG = True
#%% READ SESSION PAIRS

same_location = PAIR_SPECS["physical_loc_off"].eq(PAIR_SPECS["physical_loc_on"])
specific_location = ~PAIR_SPECS["physical_loc_off"].eq("Virtual")
keep_same_location = same_location & specific_location

pair_specs = PAIR_SPECS.loc[keep_same_location].reset_index(drop=True).copy()
excluded_pair_specs = PAIR_SPECS.loc[~keep_same_location].reset_index(drop=True).copy()
sessions = pair_specs['subject_hemi'].values

def decode_hdf5_chars(dataset):
    """Decode one MATLAB v7.3 character array."""
    return "".join(chr(int(value)) for value in dataset[()].reshape(-1))

def load_data(subject_hemi, on, verbose=False):
    """Return one complete 1-D channel and its Python slice boundaries.

    Parameters
    ----------
    subject_hemi : str
        A retained subject-and-hemisphere identifier from ``pair_specs``, such
        as ``G10_leSTN``.
    on : bool
        ``True`` loads the ON-medication file; ``False`` loads OFF medication.

    Returns
    -------
    trace : np.ndarray, shape (n_samples,)
        The complete author-selected channel, before detrending.
    (start_ix, stop_ix) : tuple[int, int]
        Python zero-indexed, stop-exclusive boundaries reproducing the
        authors' inclusive MATLAB slice when used as ``trace[start_ix:stop_ix]``.
    fs : float
        Sampling rate in Hz read from this specific MAT file.
    """
    if not isinstance(on, (bool, np.bool_)):
        raise TypeError("on must be True for ON medication or False for OFF.")

    matches = pair_specs.loc[pair_specs["subject_hemi"].eq(subject_hemi)]
    if matches.empty:
        available = ", ".join(pair_specs["subject_hemi"])
        raise ValueError(f"Unknown or excluded subject_hemi {subject_hemi!r}. Available same-location subject-hemi identifiers: {available}")
    if len(matches) != 1:
        raise ValueError(f"Expected one row for {subject_hemi!r}, found {len(matches)}.")

    spec = matches.iloc[0]
    condition = "on" if on else "off"
    folder = ON_DIR if on else OFF_DIR
    path = folder / spec[f"file_{condition}"]
    channel_ix = int(spec[f"channel_ix_{condition}"])
    expected_physical_loc = spec[f"physical_loc_{condition}"]
    start_s = float(spec[f"start_s_{condition}"])
    stop_s = float(spec[f"stop_s_{condition}"])

    if not path.exists():
        raise FileNotFoundError(path)

    if h5py.is_hdf5(path):
        # MATLAB v7.3 is HDF5. Read only the requested channel, but read that
        # channel for the complete recording. HDF5 stores WvData transposed
        # relative to MATLAB: samples x channels in these uploaded files.
        with h5py.File(path, "r") as mat:
            fs = float(mat["SmrData/Fs"][0, 0])
            wave_data = mat["SmrData/WvData"]
            title_refs = mat["SmrData/WvTits"][()].reshape(-1)
            physical_loc = decode_hdf5_chars(mat[title_refs[channel_ix]])

            if wave_data.shape[1] == len(title_refs):
                trace = np.asarray(wave_data[:, channel_ix], dtype=float)
            elif wave_data.shape[0] == len(title_refs):
                trace = np.asarray(wave_data[channel_ix, :], dtype=float)
            else:
                raise ValueError(f"Cannot identify the channel axis in {path.name}: WvData shape={wave_data.shape}, titles={len(title_refs)}.")
        mat_format = "v7.3/HDF5"
    else:
        # MATLAB Level-5 files are not HDF5. scipy loads the SmrData variable
        # into memory first; we then copy out only the requested 1-D channel.
        loaded = io.loadmat(path, struct_as_record=False, squeeze_me=True)
        smr_data = loaded["SmrData"]
        fs = float(smr_data.Fs)
        wave_data = np.asarray(smr_data.WvData)
        titles = np.asarray(smr_data.WvTits, dtype=object).reshape(-1)
        physical_loc = str(titles[channel_ix])

        if wave_data.ndim == 1:
            if len(titles) != 1 or channel_ix != 0:
                raise ValueError(f"Unexpected one-dimensional WvData in {path.name}.")
            trace = np.array(wave_data, dtype=float, copy=True)
        elif wave_data.shape[0] == len(titles):
            trace = np.array(wave_data[channel_ix, :], dtype=float, copy=True)
        elif wave_data.shape[1] == len(titles):
            trace = np.array(wave_data[:, channel_ix], dtype=float, copy=True)
        else:
            raise ValueError(f"Cannot identify the channel axis in {path.name}: WvData shape={wave_data.shape}, titles={len(titles)}.")

        del loaded, smr_data, wave_data
        gc.collect()
        mat_format = "v5"

    trace = np.asarray(trace, dtype=float).reshape(-1)
    if physical_loc != expected_physical_loc:
        raise ValueError(f"Metadata mismatch in {path.name}: channel index {channel_ix} is {physical_loc!r}, expected {expected_physical_loc!r}.")

    # MATLAB selected start_s*fs:stop_s*fs with one-based, inclusive indices.
    # The equivalent Python slice starts one sample earlier and excludes stop_ix.
    fs = int(fs)
    start_ix = int(round(start_s * fs)) - 1
    stop_ix = int(round(stop_s * fs))
    if not 0 <= start_ix < stop_ix <= trace.size:
        raise IndexError(f"Requested slice [{start_ix}:{stop_ix}] is outside a trace with {trace.size} samples in {path.name}.")

    if verbose:
        print(f"subject_hemi     : {subject_hemi}")
        print(f"condition        : {'ON' if on else 'OFF'} medication")
        print(f"file             : {path.name}")
        print(f"MAT format       : {mat_format}")
        print(f"physical location: {physical_loc}")
        print(f"channel index    : {channel_ix} (Python zero-indexed)")
        print(f"sampling rate    : {fs:g} Hz")
        print(f"complete trace   : {trace.shape} ({trace.size / fs:.2f} s)")
        print(f"snippet slice    : [{start_ix}:{stop_ix}] ({(stop_ix-start_ix)/fs:.6f} s)")

    return trace, (start_ix, stop_ix), fs

def detrend(trace):
    """Remove the least-squares straight line from a complete 1-D trace.

    File-format handling has already happened in ``load_data``. Therefore this
    function works identically for traces originating from v5 or v7.3 files.
    It accumulates the regression in chunks to avoid allocating a second
    full-recording sample-index array.
    """
    trace = np.asarray(trace)
    if trace.ndim != 1:
        raise ValueError(f"detrend expects a 1-D input; received shape {trace.shape}.")
    if trace.size < 2:
        raise ValueError("detrend requires at least two samples.")
    if not np.issubdtype(trace.dtype, np.number):
        raise TypeError(f"detrend expects numeric data; received {trace.dtype}.")
    if not np.isfinite(trace).all():
        raise ValueError("detrend does not accept NaN or infinite values.")

    n_samples = trace.size
    center = (n_samples - 1) / 2
    value_sum = 0.0
    centered_product_sum = 0.0

    for chunk_start in range(0, n_samples, DETREND_CHUNK_SIZE):
        chunk_stop = min(chunk_start + DETREND_CHUNK_SIZE, n_samples)
        values = np.asarray(trace[chunk_start:chunk_stop], dtype=float)
        sample_ix = np.arange(chunk_start, chunk_stop, dtype=float) - center
        value_sum += values.sum(dtype=float)
        centered_product_sum += np.dot(sample_ix, values)

    intercept = value_sum / n_samples
    centered_square_sum = n_samples * (n_samples**2 - 1) / 12
    slope = centered_product_sum / centered_square_sum

    detrended = np.array(trace, dtype=float, copy=True)
    for chunk_start in range(0, n_samples, DETREND_CHUNK_SIZE):
        chunk_stop = min(chunk_start + DETREND_CHUNK_SIZE, n_samples)
        sample_ix = np.arange(chunk_start, chunk_stop, dtype=float) - center
        detrended[chunk_start:chunk_stop] -= intercept + slope * sample_ix

    return detrended

def notch(trace, fs, line_freq=50.0):
    """Remove one line-noise frequency from a 1-D trace with zero phase shift.

    A quality factor of 30 gives a narrow stopband around 50 Hz. ``filtfilt``
    applies the second-order IIR notch forward and backward, so the output has
    no phase delay. Apply this to the complete channel before taking a snippet
    whenever memory permits; that keeps filter-edge artifacts away from the
    selected analysis interval.
    """
    trace = np.asarray(trace)
    if trace.ndim != 1:
        raise ValueError(f"notch expects a 1-D input; received shape {trace.shape}.")
    if not np.issubdtype(trace.dtype, np.number):
        raise TypeError(f"notch expects numeric data; received {trace.dtype}.")
    if not np.isfinite(trace).all():
        raise ValueError("notch does not accept NaN or infinite values.")

    fs = float(fs)
    line_freq = float(line_freq)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be positive and finite; received {fs}.")
    if not 0 < line_freq < fs / 2:
        raise ValueError(f"line_freq must be between 0 and the Nyquist frequency ({fs / 2:g} Hz); received {line_freq:g} Hz.")
    if trace.size <= 9:
        raise ValueError("notch requires more than 9 samples for zero-phase filtering.")

    numerator, denominator = signal.iirnotch(w0=line_freq, Q=NOTCH_QUALITY_FACTOR, fs=fs)
    return signal.filtfilt(numerator, denominator, np.asarray(trace, dtype=float))

def filter_boxcar(data, window_size):
    kernel = np.ones(window_size) / window_size
    # 'same' mode ensures the output array matches the input size
    return np.vstack([np.convolve(row, kernel, mode='same') for row in data])

def filter_gaussian(data, sigma, radius=None):
    """
    Applies a 1D Gaussian filter across axis=1 (rows) of a 2D matrix 
    using a pure NumPy list comprehension.
    """
    if radius is None:
        radius = int(3 * sigma) # 1. Define the kernel width (standard practice is 3 to 4 sigma on each side)
    
    x = np.arange(-radius, radius + 1) # 2. Create the 1D Gaussian coordinate grid [-radius, ..., 0, ..., radius]
    kernel = np.exp(-0.5 * (x / sigma) ** 2) # 3. Calculate the Gaussian curve equation
    kernel /= np.sum(kernel) # 4. Normalize the kernel so its elements sum up to 1.0 (prevents darkening/brightening)
    return np.vstack([np.convolve(row, kernel, mode='same') for row in data]) # 5. Apply row-by-row via list comprehension exactly like your boxcar code

def rank_n_reconstruction(fa: FAResults, start_rank: int, end_rank: int):
    """
    Zero-indexing for start and end ranks.
        Rank-1 reconstruction: start_rank = 0, end_rank = 1
        Rank-2 reconstruction: start_rank = 0, end_rank = 2
        Second-rank reconstruction: start_rank = 1, end_rank = 2

    Return reconstructed power (n_freq, n_sample)
    """

    rank_ix = slice(start_rank, end_rank)
    
    rank_component = fa.transformed_latents[:, rank_ix] @ fa.subspace[rank_ix, :]
    rank_recon = fa.fa.mean_ + rank_component # fa.fa.mean_ is almost 0

    # Returning to original scale (inverse norm_log transform)
    feature_mean = log_specs[freqs_mask].mean(axis=1)
    feature_std = log_specs[freqs_mask].std(axis=1)
    rank_log = rank_recon * feature_std[None, :] + feature_mean[None, :]
    rank_power = 10 ** rank_log

    return rank_power.T

def plot_recon_spectrogram_difference():
    # plot reconstructed spectrogram vs original spectrogram
    plot_spectrogram(spec_time=spectrogram_time, spec_freq=on.spec_freqs[freqs_mask], spec_power=recon_power)
    plot_spectrogram(spec_time=spectrogram_time, spec_freq=on.spec_freqs[freqs_mask], spec_power=spectrogram[freqs_mask])

def plot_recon_psd_difference(include_original: bool):

    if include_original:
        # plot PSD of reconstructed and original PSD
        fig, ax = plt.subplots(1,2,figsize=(4,2), dpi=300)
        ax[0].plot(on.spec_freqs[freqs_mask], spectrogram[freqs_mask][:, spect_is_on.astype(bool)].mean(1), label='ON meds')
        ax[0].plot(on.spec_freqs[freqs_mask], spectrogram[freqs_mask][:, ~spect_is_on.astype(bool)].mean(1), label='OFF meds')
        ax[1].plot(on.spec_freqs[freqs_mask], recon_power[:, spect_is_on.astype(bool)].mean(1), label='ON meds')
        ax[1].plot(on.spec_freqs[freqs_mask], recon_power[:, ~spect_is_on.astype(bool)].mean(1), label='OFF meds')
        ax[0].set_title('Original spectrogram')
        ax[1].set_title('Rank-1 recon spectrogram')
        ax[0].set_xlabel('Frequency (Hz)')
        ax[1]
        plt.legend(frameon=False)
        finish_plot()
    else:
        plt.figure(figsize=(2,2), dpi=300)
        plt.plot(on.spec_freqs[freqs_mask], recon_power[:, spect_is_on.astype(bool)].mean(1), label='ON meds')
        plt.plot(on.spec_freqs[freqs_mask], recon_power[:, ~spect_is_on.astype(bool)].mean(1), label='OFF meds')
        plt.title('Original spectrogram')
        plt.title('Rank-1 recon PSD')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD (V^2/Hz)')
        plt.legend(frameon=False, fontsize=6)
        finish_plot()

class WiestRecording:
    """A single subject-and-hemisphere session with ON or OFF medication.

    This class is a convenience wrapper around ``load_data``.  It loads both
    ON and OFF channels, applies the 50 Hz notch filter, and detrends 
    channel.
    """

    def __init__(self, subject_hemi, on_meds, apply_notch=True, verbose=False):
        self.subject_hemi = subject_hemi
        self.on_meds = on_meds
        self.trace, (self.start_ix, self.stop_ix), self.fs = load_data(subject_hemi=subject_hemi, on=self.on_meds, verbose=verbose)
        if apply_notch:
            self.trace = notch(self.trace, fs=self.fs)
        else:
            warn('No Notch Filter Applied')
            self.trace = self.trace
        self.snippet = self.trace[self.start_ix:self.stop_ix]
        self.snippet = detrend(self.snippet)
        self.specparam_fit_range = None
        self.interp_freq_range = None
        self.crop_psd_f = None
        self.crop_psd = None
        self.specparams = None

        if verbose:
            print("\nAfter detrending and slicing")
            print(f"trace shape      : {self.trace.shape}")
            print(f"snippet shape    : {self.snippet.shape}")
            print(f"sampling rate    : {self.fs:g} Hz")
            print(f"trace mean       : {self.trace.mean():.6g}")
            print(f"snippet mean / SD: {self.snippet.mean():.6g} / {self.snippet.std():.6g}")

    def __repr__(self):
        return (
            f"WiestRecording(subject_hemi={self.subject_hemi!r}, "
            f"on_meds={self.on_meds})"
        )
    
    @property
    def snippet_time(self):
        """Return a time array for the snippet in seconds."""
        return np.arange(self.snippet.size, dtype=float) / self.fs
    
    def spectral_analysis(
            self, 
            window_s=1.0, 
            overlap_frac=0.5,
            wavelet_window_s=1.0,
            fwhm=0.3,
            freqs_hz=np.arange(1, 101, 1)
        ):

        """Return the PSD and spectrogram of the snippet."""
        self.psd_f, self.psd = get_psd(trace=self.snippet, fs=self.fs, window_s=window_s, overlap_frac=overlap_frac, window='hann', axis=0)

        self.spec_time, self.spec_freqs, self.spec_power = get_spectrogram(trace=self.snippet, fs=self.fs, freqs_hz=freqs_hz, fwhm=fwhm, wavelet_window_s=wavelet_window_s)

    def crop_freq(self, freq_range):
        self.specparam_fit_range = freq_range
        if len(freq_range) != 2 or freq_range[0] >= freq_range[1]:
            raise ValueError("freq_range must be an increasing (low, high) pair.")
        keep = (self.psd_f >= freq_range[0]) & (self.psd_f <= freq_range[1])
        if keep.sum() < 3:
            raise ValueError(f"Only {keep.sum()} bins fall in frequency range {freq_range}.")
        self.crop_psd_f = self.psd_f[keep].copy()
        self.crop_psd = self.psd[keep].copy()

    def interp_freq(self, freq_range):
        if self.crop_psd_f is None or self.crop_psd is None:
            raise ValueError('Frequencies not cropped properly. Check crop_freq().')
        if len(freq_range) != 2 or freq_range[0] >= freq_range[1]:
            raise ValueError("freq_range must be an increasing (low, high) pair.")
        self.interp_freq_range = freq_range

        freqs = np.asarray(self.crop_psd_f, dtype=float)
        cleaned = np.array(self.crop_psd, dtype=float, copy=True)
            
        low, high = self.interp_freq_range
        contaminated = (freqs >= low) & (freqs <= high)
        left_candidates = np.flatnonzero(freqs < low)
        right_candidates = np.flatnonzero(freqs > high)
        if not np.any(contaminated):
            raise ValueError(f"No cropped PSD bins fall within interpolation range {freq_range}.")
        if not len(left_candidates) or not len(right_candidates):
            raise ValueError(f"Interpolation range {freq_range} must have clean bins on both sides.")
        left = left_candidates[-1]
        right = right_candidates[0]
        weights = (freqs[contaminated] - freqs[left]) / (freqs[right] - freqs[left])
        interpolation_shape = (weights.size,) + (1,) * (cleaned.ndim - 1)
        weights = weights.reshape(interpolation_shape)
        cleaned[contaminated] = (1 - weights) * cleaned[left] + weights * cleaned[right]
        self.crop_psd = cleaned

    def fit_specparam(self, mode):
        model = SpectralModel(**SPECPARAM_SETTINGS)
        if mode == 'full':
            model.fit(self.psd_f, self.psd)
        elif mode == 'crop':
            model.fit(self.crop_psd_f, self.crop_psd)
        else:
            raise ValueError('Wrong mode!')
        self.specparams = model
        self.specparams.fit_mode = mode

def plot_verify_psd_interpolation(data):
    plt.figure(figsize=(3,3), dpi=300)
    data.crop_freq(SPECPARAM_FIT_RANGE) # First crop frequency
    plt.plot(data.crop_psd_f,data.crop_psd, label='notch-filtered')
    data.interp_freq(MAINS_NOISE_RANGE) # And then interpolate around the notch-filtered area
    plt.plot(data.crop_psd_f,data.crop_psd,c='r', ls='--', label='interpolated')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (V^2/Hz)')
    plt.legend(frameon=False)
    finish_plot()

def plot_psd(on, off):
    plt.figure(figsize=(3,3), dpi=300)
    plt.plot(off.psd_f, off.psd, label="OFF", color="blue")
    plt.plot(on.psd_f, on.psd, label="ON", color="red")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (V^2/Hz)')
    plt.xlim(1, 100)
    plt.legend(frameon=False)
    finish_plot()

def plot_specparam_fit(on, off):
    freqs = on.specparams.data.freqs
    clr = ['b','r']
    plt.figure(figsize=(3,3), dpi=300)
    for i, datum in enumerate([off, on]):
        meds = 'ON' if datum.on_meds else 'OFF'
        offset, exponent = datum.specparams.get_params('aperiodic')
        fit_curve = 10**offset * freqs**(-exponent)
        plt.plot(datum.crop_psd_f,datum.crop_psd,c=clr[i], ls='--', label=f'{meds} data')
        plt.plot(freqs, fit_curve, c=clr[i], label=f'{meds} fit: {exponent:.2f}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (V^2/Hz)')
    plt.title(session)
    plt.legend(frameon=False)
    finish_plot()

def plot_spectrogram(spec_time, spec_freq, spec_power, stride: int = 100, dir_name=None):
    plt.figure(figsize=(8,3), dpi=300)
    plt.pcolormesh(spec_time[::stride], spec_freq, 10 * np.log10(spec_power[:,::stride]), shading='auto', cmap='viridis', vmin=-40, vmax=0)
    plt.axvline(0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('OFF to ON')
    plt.colorbar(label='Power (dB)')
    plt.xlim(spec_time[0], spec_time[-1])
    finish_plot(
        filename='spectrogram',
        save_dir=dir_name,
        savefig=SAVEFIG,
        show=SHOWFIG,
        exts=('pdf',)
    )

def plot_power_transformation(raw, log, norm_log):
    # Inspect how log transformation and standardization change the distribution.
    fig, ax = plt.subplots(1,3,figsize=(6,2), dpi=300)
    ax[0].hist(raw, bins=50)
    ax[1].hist(log, bins=50)
    ax[2].hist(norm_log, bins=50)
    ax[0].set_title('Raw')
    ax[1].set_title('Log')
    ax[2].set_title('Norm_log')
    ax[0].set_ylabel('Counts')
    finish_plot()

def plot_subspace_dim(fa: FAResults, dir_name=None):
    fig, ax = plt.subplots(1,5,figsize=(12,3), dpi=300, sharey=True)
    for i in range(5):
        ax[i].scatter(fa.freq_hz, fa.subspace[i], color='k',s=6)
        ax[i].plot(fa.freq_hz, fa.subspace[i], color='k',lw=1, ls='--')
        ax[i].set_title(f'Dim {i+1}')
        ax[i].set_xlabel('Frequency (Hz)')
        ax[i].axhline(0, ls='--', lw=0.8, c='k')
    ax[0].set_ylabel('FA weight')
    finish_plot(
        filename='subspace_dim',
        save_dir=dir_name,
        savefig=SAVEFIG,
        show=SHOWFIG,
        exts=('pdf',)
    )

def band_similarity_dataframe(modes, freqs, bands):
    """
    Compare every FA frequency mode with each canonical frequency band
    Return absolute cosine similarity for every mode-by-band pairing.
    """
    modes = np.asarray(modes, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    if modes.ndim != 2 or modes.shape[1] != freqs.size:
        raise ValueError("modes must have shape (n_modes, n_frequencies).")

    mode_norms = np.linalg.norm(modes, axis=1, keepdims=True)
    if np.any(mode_norms == 0):
        raise ValueError("Every subspace dimension must have a nonzero norm.")
    normalized_modes = modes / mode_norms

    rows = []
    for band_name, (low, high) in bands.items():
        band_vector = ((freqs >= low) & (freqs <= high)).astype(float)
        if not band_vector.any():
            raise ValueError(f"No frequencies fall within the {band_name} band ({low}, {high}).")
        band_vector /= np.linalg.norm(band_vector)
        similarities = np.abs(normalized_modes @ band_vector)
        for subspace_dim, similarity in enumerate(similarities, start=1):
            rows.append({"band name": band_name, 
                         "cosine similarity": float(similarity), 
                         "subspace dim": subspace_dim,
                         'normed cosine similarity': float(similarity) / (high-low)
                         })

    return pd.DataFrame(rows, columns=["band name", "cosine similarity", "subspace dim", 'normed cosine similarity'])

def plot_band_similarity(fa):

    band_similarity_df = band_similarity_dataframe(fa.subspace[:5], fa.freq_hz, frequency_bands)

    fig, ax = plt.subplots(1,5,figsize=(12,3), dpi=300)
    for i in range(5):
        ax[i].bar(range(5),band_similarity_df[band_similarity_df['subspace dim']==i+1]['normed cosine similarity'] , color='k')
        ax[i].set_title(f'Dim {i+1}')
        ax[i].set_xticks(range(5))
        ax[i].set_xticklabels(frequency_bands.keys(), rotation=45)
    finish_plot()

def plot_transformed_latents(time, fa, dir_name=None):
    fa.fa_transform()

    z = fa.transformed_latents.T
    # z_ = filter_boxcar(z, 1024*5)
    z_ = filter_gaussian(z, 4096)
    
    fig, ax = plt.subplots(5,1,figsize=(6,4), dpi=300, sharex=True)
    for i in range(5):
        ax[i].plot(time, z_[i], c='k')
        ax[i].axvline(0, ls='--', lw=0.8, c='r')
        ax[i].axhline(0, ls='--', lw=0.8, c='k')
    finish_plot(
        filename='transformed_latents',
        save_dir=dir_name,
        savefig=SAVEFIG,
        show=SHOWFIG,
        exts=('pdf',)
    )

## Exploratory LDA with a chronological train-test split
def average_log_power_blocks(log_power, fs, block_s=1.0, edge_s=1.0):
    """Average frequency-by-time log power into non-overlapping time blocks."""
    log_power = np.asarray(log_power, dtype=float)
    block_samples = int(round(block_s * fs))
    edge_samples = int(round(edge_s * fs))
    if log_power.ndim != 2 or block_samples < 1:
        raise ValueError("log_power must be frequency-by-time and block_s must be positive.")

    stop = log_power.shape[1] - edge_samples
    if stop <= edge_samples:
        raise ValueError("edge_s removes the complete recording.")
    trimmed = log_power[:, edge_samples:stop]
    n_blocks = trimmed.shape[1] // block_samples
    if n_blocks < 4:
        raise ValueError("Too few complete time blocks remain for classification.")
    trimmed = trimmed[:, :n_blocks * block_samples]
    return trimmed.reshape(trimmed.shape[0], n_blocks, block_samples).mean(axis=2).T

def chronological_holdout(blocks, test_fraction=0.25, gap_blocks=2):
    """Train on early blocks and test on late blocks, leaving a temporal gap."""
    n_test = max(1, int(np.floor(len(blocks) * test_fraction)))
    test_start = len(blocks) - n_test
    train_stop = test_start - gap_blocks
    if train_stop < 2:
        raise ValueError("Not enough training blocks after applying the temporal gap.")
    return blocks[:train_stop], blocks[test_start:]

def lda_ing():
    """
    Temporary holding of the lda function
    """
    # Work from log power, and let the pipeline learn scaling from training data only.
    # Remove 47-54 Hz so the classifier cannot use the notch/line-noise region.
    lda_freq_mask = (off.spec_freqs < MAINS_NOISE_RANGE[0]) | (off.spec_freqs > MAINS_NOISE_RANGE[1])
    lda_freq_mask = lda_freq_mask & freqs_mask

    n_off_time = off.spec_power.shape[1]
    off_log_power = log_specs[lda_freq_mask, :n_off_time]
    on_log_power = log_specs[lda_freq_mask, n_off_time:]

    off_blocks = average_log_power_blocks(off_log_power, off.fs, block_s=0.1, edge_s=1.0)
    on_blocks = average_log_power_blocks(on_log_power, on.fs, block_s=0.1, edge_s=1.0)
    off_train, off_test = chronological_holdout(off_blocks, test_fraction=0.33, gap_blocks=2)
    on_train, on_test = chronological_holdout(on_blocks, test_fraction=0.33, gap_blocks=2)

    X_train = np.vstack([off_train, on_train])
    y_train = np.concatenate([np.zeros(len(off_train), dtype=int), np.ones(len(on_train), dtype=int)])
    X_test = np.vstack([off_test, on_test])
    y_test = np.concatenate([np.zeros(len(off_test), dtype=int), np.ones(len(on_test), dtype=int)])

    lda_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ])
    lda_clf.fit(X_train, y_train)
    y_pred = lda_clf.predict(X_test)
    y_probability = lda_clf.predict_proba(X_test)[:, 1]

    lda_test_results = pd.DataFrame([{
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_probability),
        "n_train_blocks": len(y_train),
        "n_test_blocks": len(y_test),
    }])
    print("\nChronological holdout LDA")
    print(lda_test_results.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("Confusion matrix (rows=true, columns=predicted)")
    print(confusion_matrix(y_test, y_pred))

    fig, ax = plt.subplots(1,2, figsize=(5, 2.5), dpi=300)
    test_scores = lda_clf.predict_proba(X_test)[:, 1]
    ax[0].hist(test_scores[y_test == 0], bins=10, histtype="step", linewidth=1.5, color="k", label="OFF")
    ax[0].hist(test_scores[y_test == 1], bins=10, histtype="step", linewidth=1.5, color="r", label="ON")
    ax[0].set(xlabel="Held-out LDA decision score", ylabel="Count", title=f"Later time blocks, ROC AUC={roc_auc_score(y_test, test_scores):.3f}")
    ax[0].legend(frameon=False)

    lda_coefficients = lda_clf.named_steps["lda"].coef_.ravel()

    ax[1].plot(off.spec_freqs[lda_freq_mask], lda_coefficients, color="r", lw=1, ls='--')
    ax[1].scatter(off.spec_freqs[lda_freq_mask], lda_coefficients, color="r", s=6)
    ax[1].axhline(0, color="0.5", linestyle="--")
    ax[1].set(xlabel="Frequency (Hz)", ylabel="Standardized LDA coefficient", title="Training-set direction")
    finish_plot(
        filename='lda_coeff_classification',
        save_dir=f'plots/wiest/{session}',
        savefig=SAVEFIG,
        show=SHOWFIG,
        exts=('pdf',)
    )

    coeff = lda_coefficients / np.linalg.norm(lda_coefficients)
    print([float(np.abs(x@coeff)) * 100 for x in fa.subspace])

#%% Read data
i=2
# for i in range(len(sessions)):
session = sessions[i]
off = WiestRecording(subject_hemi=session, on_meds=False, apply_notch=True, verbose=True)
on = WiestRecording(subject_hemi=session, on_meds=True, apply_notch=True, verbose=True)

off.spectral_analysis(window_s=PSD_WINDOW_S)
on.spectral_analysis(window_s=PSD_WINDOW_S)

off.crop_freq(SPECPARAM_FIT_RANGE) # First crop frequency
off.interp_freq(MAINS_NOISE_RANGE) # And then interpolate around the notch-filtered area
on.crop_freq(SPECPARAM_FIT_RANGE) # First crop frequency
on.interp_freq(MAINS_NOISE_RANGE) # And then interpolate around the notch-filtered area

off.fit_specparam('crop') # Fit the crop mode instead of the full mode
on.fit_specparam('crop')

section('Process spectrogram')
spectrogram = np.concatenate([off.spec_power, on.spec_power],axis=1,)
off_spectrogram_time = (np.arange(off.spec_power.shape[1], dtype=float) - off.spec_power.shape[1] ) / off.fs
on_spectrogram_time = (np.arange(on.spec_power.shape[1], dtype=float)  / on.fs)
spectrogram_time = np.concatenate([off_spectrogram_time, on_spectrogram_time])
spect_is_on = np.array([0] * off.spec_power.shape[1] + [1] * on.spec_power.shape[1])

medication_onset_ix = off.spec_power.shape[1]
assert spectrogram_time[medication_onset_ix] == 0
assert np.all(np.diff(spectrogram_time) > 0)

# Log-transform
log_specs = np.log10(np.maximum(spectrogram, np.finfo(float).tiny))
log_specs_std = log_specs.std(axis=1, keepdims=True)
log_specs_std[log_specs_std == 0] = 1
norm_specs = (log_specs - log_specs.mean(axis=1, keepdims=True)) / log_specs_std

plot_spectrogram(spectrogram_time, off.spec_freqs, spectrogram, dir_name=f'plots/wiest/{session}')
# plot_power_transformation(spectrogram[0], log_specs[0], norm_specs[0])

section('Fit FA')

masks = (40,90)
freqs_mask = (on.spec_freqs >= masks[0]) & (on.spec_freqs <= masks[1])
noise_mask = (off.spec_freqs < MAINS_NOISE_RANGE[0]) | (off.spec_freqs > MAINS_NOISE_RANGE[1])
freqs_mask = freqs_mask & noise_mask
fa = FAResults()
fa.fa_fit(norm_specs[freqs_mask].T, shared_var_thresh=0.95,
        #   cv_components=np.arange(norm_specs[freqs_mask].shape[0]), 
        #   max_iter=int(1e4)
)
fa.freq_hz = on.spec_freqs[freqs_mask]

announce(f'FA complete, d={fa.d_shared}.')
fa.fa_transform()
z = fa.transformed_latents.T

plot_subspace_dim(fa, dir_name=f'plots/wiest/{session}')
plot_transformed_latents(spectrogram_time, fa, dir_name=f'plots/wiest/{session}')

z_ = filter_gaussian(z, 1024)
off.latent_mean = z_[:, spectrogram_time < 0].mean(axis=1)
on.latent_mean = z_[:, spectrogram_time >= 0].mean(axis=1)
fig, ax = plt.subplots(1,5,figsize=(12,2.5), dpi=300, sharey=True)
for i in range(5):
    ax[i].hist(z[i, spectrogram_time < 0], color='k', histtype='step', lw=1.5, density=True, bins=500)
    ax[i].hist(z[i, spectrogram_time >= 0], color='r', histtype='step', lw=1.5, density=True, bins=500)
    ax[i].set_xlabel(f'Latent activity, dim {i+1}')
    ax[i].set_ylabel('Density')
finish_plot(
    filename='latent_activity_histograms',
    save_dir=f'plots/wiest/{session}',
    savefig=SAVEFIG,
    show=SHOWFIG,
    exts=('pdf',)
)


# Separate ON and OFF, do FA for each.
# on
log_specs = np.log10(np.maximum(on.spec_power, np.finfo(float).tiny))
log_specs_std = log_specs.std(axis=1, keepdims=True)
log_specs_std[log_specs_std == 0] = 1
norm_specs = (log_specs - log_specs.mean(axis=1, keepdims=True)) / log_specs_std

fa_on = FAResults()
fa_on.fa_fit(norm_specs[freqs_mask].T, shared_var_thresh=0.95)
fa_on.freq_hz = on.spec_freqs[freqs_mask]

# off
log_specs = np.log10(np.maximum(off.spec_power, np.finfo(float).tiny))
log_specs_std = log_specs.std(axis=1, keepdims=True)
log_specs_std[log_specs_std == 0] = 1
norm_specs = (log_specs - log_specs.mean(axis=1, keepdims=True)) / log_specs_std

fa_off = FAResults()
fa_off.fa_fit(norm_specs[freqs_mask].T, shared_var_thresh=0.95)
fa_off.freq_hz = off.spec_freqs[freqs_mask]

# plot first PC comparison

plt.figure(figsize=(3,3), dpi=300)
plt.scatter(fa_on.freq_hz, fa_on.subspace[0], label='ON', color='r',s=6)
plt.plot(fa_on.freq_hz, fa_on.subspace[0], color='r',lw=1, ls='-')

plt.scatter(fa_off.freq_hz, fa_off.subspace[0], label='OFF', color='g',s=6)
plt.plot(fa_off.freq_hz, fa_off.subspace[0], color='g',lw=1, ls='-')

plt.scatter(fa_off.freq_hz, fa.subspace[0], label='Combined', color='k',s=6)
plt.plot(fa_off.freq_hz, fa.subspace[0], color='k',lw=0.5, ls='--')

plt.xlabel('Frequency (Hz)')
plt.ylabel('FA weights')

plt.legend(frameon=False)
finish_plot(
    filename='first_PC_comparison',
    save_dir=f'plots/wiest/{session}',
    savefig=True,
    show=SHOWFIG,
    exts=('pdf',)
)

#%% some supplementary figure

plt.figure(figsize=(3,3), dpi=300)
plt.plot(on.psd_f, on.psd, label="ON", color="red")
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (V^2/Hz)')
plt.xlim(1, 100)
plt.ylim(4e-3,5)
plt.xscale('log')
plt.yscale('log')
plt.legend(frameon=False)
finish_plot()

#%% Repeat Figure 2C (left) in Wiest paper, note their N=30 ours 16

df = []
for j in range(len(sessions)):
    session = sessions[j]
    print(session)
    off = WiestRecording(subject_hemi=session, on_meds=False, apply_notch=False, verbose=False)
    on = WiestRecording(subject_hemi=session, on_meds=True, apply_notch=False, verbose=False)

    for datum in [off, on]:
        datum.spectral_analysis(window_s=4.0)
        datum.crop_freq(SPECPARAM_FIT_RANGE) # First crop frequency
        datum.interp_freq(MAINS_NOISE_RANGE) # And then interpolate around the notch-filtered area
        datum.fit_specparam('crop') # Fit the crop mode instead of the full mode
        offset, exponent = datum.specparams.get_params('aperiodic')

        df.append({'session': session, 'offset':offset, 'exponent':exponent, 'on_meds': datum.on_meds})
df = pd.DataFrame(df)

fig, ax = plt.subplots(1,1,figsize=(3,3), dpi=300)
sns.boxplot(data=df, x='on_meds', y='exponent', ax=ax, linecolor='k', color='white', fliersize=0)
sns.stripplot(data=df, x='on_meds', y='exponent', ax=ax, edgecolor='k', linewidth=1, facecolor='none')
plt.title('Before/after Levodopa, exponent change')
plt.xticks([0,1],['Off meds','On meds'])
plt.xlabel('')
finish_plot()

pg.ttest(df[df['on_meds']]['exponent'], df[~df['on_meds']]['exponent'], paired=True, alternative='greater')

#%% First rank reconstruction

recon_power = rank_n_reconstruction(fa, start_rank=0, end_rank=1)
plot_recon_psd_difference(include_original=False)

## TODO: and then fit psd


# %%
for i in range(len(sessions)):
    session = sessions[i]
    off = WiestRecording(subject_hemi=session, on_meds=False, apply_notch=True, verbose=True)
    on = WiestRecording(subject_hemi=session, on_meds=True, apply_notch=True, verbose=True)

    off.spectral_analysis(window_s=PSD_WINDOW_S)
    on.spectral_analysis(window_s=PSD_WINDOW_S)

    off.crop_freq(SPECPARAM_FIT_RANGE) # First crop frequency
    off.interp_freq(MAINS_NOISE_RANGE) # And then interpolate around the notch-filtered area
    on.crop_freq(SPECPARAM_FIT_RANGE) # First crop frequency
    on.interp_freq(MAINS_NOISE_RANGE) # And then interpolate around the notch-filtered area

    spec_time = (np.arange(on.spec_power.shape[1], dtype=float)  / on.fs)
    spec_freq = on.spec_freqs
    spec_power = on.spec_power
    stride = 100

    dir_name = f'plots/wiest/{session}'

    plt.figure(figsize=(3,2), dpi=300)
    plt.pcolormesh(spec_time[::stride], spec_freq, 10 * np.log10(spec_power[:,::stride]), shading='auto', cmap='viridis', vmin=-40, vmax=0)
    # plt.axvline(0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Power (dB)')
    plt.xlim(spec_time[0], spec_time[-1])
    finish_plot(
        filename=f'{on.subject_hemi}_ON_spectrogram',
        save_dir=dir_name,
        savefig=SAVEFIG,
        show=SHOWFIG,
        exts=('pdf',)
    )

    spec_time = (np.arange(off.spec_power.shape[1], dtype=float)  / on.fs)
    spec_freq = off.spec_freqs
    spec_power = off.spec_power
    stride = 100

    plt.figure(figsize=(3,2), dpi=300)
    plt.pcolormesh(spec_time[::stride], spec_freq, 10 * np.log10(spec_power[:,::stride]), shading='auto', cmap='viridis', vmin=-40, vmax=0)
    # plt.axvline(0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Power (dB)')
    plt.xlim(spec_time[0], spec_time[-1])
    finish_plot(
        filename=f'{off.subject_hemi}_OFF_spectrogram',
        save_dir=dir_name,
        savefig=SAVEFIG,
        show=SHOWFIG,
        exts=('pdf',)
    )


# %%
plt.figure(figsize=(2.5,2), dpi=300)
plt.plot(spec_freq, spec_power.mean(1))
plt.ylabel('PSD (V^2/Hz)')
plt.xlabel('Frequency (Hz)')
plt.xscale('log')
plt.yscale('log')
finish_plot()
# %%

i=2
session = sessions[i]
off = WiestRecording(subject_hemi=session, on_meds=False, apply_notch=True, verbose=True)
on = WiestRecording(subject_hemi=session, on_meds=True, apply_notch=True, verbose=True)

off.spectral_analysis(window_s=PSD_WINDOW_S)
on.spectral_analysis(window_s=PSD_WINDOW_S)

log_specs = np.log10(np.maximum(on.spec_power, np.finfo(float).tiny))
log_specs_std = log_specs.std(axis=1, keepdims=True)
log_specs_std[log_specs_std == 0] = 1
norm_specs = (log_specs - log_specs.mean(axis=1, keepdims=True)) / log_specs_std

masks = (40,90)
freqs_mask = (on.spec_freqs >= masks[0]) & (on.spec_freqs <= masks[1])
noise_mask = (off.spec_freqs < MAINS_NOISE_RANGE[0]) | (off.spec_freqs > MAINS_NOISE_RANGE[1])
freqs_mask = freqs_mask & noise_mask
fa = FAResults()
fa.fa_fit(norm_specs[freqs_mask].T, shared_var_thresh=0.95,
        #   cv_components=np.arange(norm_specs[freqs_mask].shape[0]), 
        #   max_iter=int(1e4)
)
fa.freq_hz = on.spec_freqs[freqs_mask]

plt.figure(figsize=(3,3), dpi=300)
exp_var = np.cumsum(fa.explained_variance_ratio_)
plt.plot(exp_var, c='k', lw=1, ls='--')
plt.scatter(np.arange(len(exp_var)), exp_var, s=5, color='k')
plt.ylabel('Cumulative explained variance')
plt.xlabel('Number of components')
finish_plot(
        filename=f'cum_exp_variance_ON',
        save_dir=dir_name,
        savefig=SAVEFIG,
        show=SHOWFIG,
        exts=('pdf',)
    )

# %%

plt.figure(figsize=(3,3), dpi=300)
plt.plot(fa.freq_hz, fa.shared_var_per_unit, c='k', lw=1, ls='--')
plt.scatter(fa.freq_hz, fa.shared_var_per_unit, s=5, color='k')
plt.ylabel('Shared variance per frequency')
plt.xlabel('Frequency (Hz)')
finish_plot(
        filename=f'shared_var_per_freq_ON',
        save_dir=dir_name,
        savefig=SAVEFIG,
        show=SHOWFIG,
        exts=('pdf',)
    )

# %%

plt.figure(figsize=(2,2), dpi=300)
plt.plot(fa_off.freq_hz, fa_off.subspace[3],c='k', lw=1, ls='--')
plt.scatter(fa_off.freq_hz, fa_off.subspace[3], s=5, color='k')
plt.ylabel('FA weights')
plt.xlabel('Frequency (Hz)')
finish_plot()

# %%
