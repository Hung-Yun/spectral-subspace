import numpy as np
from scipy import signal

"""

Spectral analysis functions for simulated and real neural time series data.
Functions include:
- get_psd: Compute power spectral density using Welch's method.
- get_spectrogram: Compute time-frequency power using Morlet wavelet convolution.
- get_power_cov: Compute frequency covariance from power time series.
- get_autocorr: Compute autocorrelation function up to a specified lag.

"""


def get_psd(trace, fs, window_s, overlap_frac, window='hann', axis=0):
    trace = np.asarray(trace, dtype=float)
    nperseg = min(int(window_s * fs), trace.shape[axis])
    if nperseg < 1:
        raise ValueError('PSD window is too short for the provided trace.')

    noverlap = min(int(nperseg * overlap_frac), max(nperseg - 1, 0))
    freqs, psd = signal.welch(
        trace,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        axis=axis,
    )
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


def get_power_cov(power_by_freq_time, z_scored=True, eps=1e-12):
    log_power = np.log(power_by_freq_time + eps)
    power_mean = log_power.mean(axis=1, keepdims=True)
    power_std = log_power.std(axis=1, keepdims=True)
    variable_rows = power_std.squeeze() > 0
    power_std[power_std == 0] = 1

    if z_scored:
        z_power = (log_power - power_mean) / power_std
        freq_corr = np.zeros((z_power.shape[0], z_power.shape[0]), dtype=float)
        if np.any(variable_rows):
            if np.sum(variable_rows) == 1:
                freq_corr[np.ix_(variable_rows, variable_rows)] = 1.0
            else:
                freq_corr[np.ix_(variable_rows, variable_rows)] = np.corrcoef(z_power[variable_rows])
        return z_power, freq_corr

    freq_corr = np.zeros((log_power.shape[0], log_power.shape[0]), dtype=float)
    if np.any(variable_rows):
        if np.sum(variable_rows) == 1:
            freq_corr[np.ix_(variable_rows, variable_rows)] = 1.0
        else:
            freq_corr[np.ix_(variable_rows, variable_rows)] = np.corrcoef(log_power[variable_rows])
    return None, freq_corr


def get_autocorr(trace, max_lag_samples):
    trace = np.asarray(trace, dtype=float)
    trace = trace - trace.mean()
    if np.allclose(trace, 0):
        return np.zeros(max_lag_samples + 1, dtype=float)

    full_corr = signal.correlate(trace, trace, mode='full')
    acf = full_corr[trace.shape[0] - 1:trace.shape[0] + max_lag_samples]
    return acf / acf[0]
