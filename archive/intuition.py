#%% imports

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal

#%% FIRLS filter kernel design and inspection

def firls_kernel(
        samprate=30000, 
        filtorder=18001, 
        lower_bnd=12, 
        upper_bnd=30, 
        lower_trans=0.1, 
        upper_trans=0.1, 
        filter_shape=[0, 0, 1, 1, 0, 0]
):

    # In the firls method, we designed the filter in terms of the frequencies at which the gain changes,
    # and the desired gain at those frequencies. The transition bands are defined as a fraction of the
    # lower and upper bounds, so the actual frequencies at which the gain changes are:

    filter_freqs = [
        0,
        lower_bnd * (1 - lower_trans),
        lower_bnd,
        upper_bnd,
        upper_bnd * (1 + upper_trans),
        samprate / 2,
    ]

    # Generating the filter kernel and its frequency response based on the above specifications.
    filterkern = signal.firls(filtorder, filter_freqs, filter_shape, fs=samprate)

    # The frequencies corresponding to the FFT of the filter kernel, and the power of the filter at those frequencies.
    # Usually hz can go as high as samprate, but the filter response is symmetric around samprate/2,
    # so we only need to look at the frequencies up to samprate/2 (Nyquist frequency).
    hz = np.linspace(0, samprate / 2, int(np.floor(len(filterkern) / 2) + 1))
    filterpow = np.abs(np.fft.fft(filterkern)) ** 2

    plt.figure(figsize=(3, 3), dpi=300)
    plt.plot(filterkern)
    plt.xlabel('Time points')
    plt.ylabel('Weight')
    sns.despine(trim=False)
    plt.tight_layout()
    plt.show()

    # Ideally we want the black line (the actual filter response) to match the 
    # red points (the desired filter response at the specified frequencies).
    plt.figure(figsize=(3, 3), dpi=300)
    plt.plot(hz, filterpow[:len(hz)], 'k-')
    plt.plot(filter_freqs, filter_shape, 'ro-')
    plt.xlim([0, upper_bnd + 40])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Filter gain')
    sns.despine(trim=False)
    plt.tight_layout()
    plt.show()


firls_kernel(lower_bnd=12, upper_bnd=30) # Traditional beta band
firls_kernel(lower_bnd=65, upper_bnd=68) # Check out different band specs
firls_kernel(filtorder=3001 , lower_bnd=65, upper_bnd=68) # Check out different filter orders. This one is much less sharp, but also much faster to compute.

#%% Wavelet kernels

def gaussian_wavelet(center_freq_hz=20, wavelet_fwhm=0.5, wavelet_fs=1000):
    """
    wavelet_fwhm controls how wide that Gaussian is in seconds.
    So if wavelet_fwhm = 0.3, that means the Gaussian’s width at half of its peak height is about 0.3 s.
    """
    wavelet_time = np.arange(-1, 1, 1 / wavelet_fs)
    gaussian = np.exp(-(4 * np.log(2) * wavelet_time**2) / wavelet_fwhm**2)
    wavelet = np.exp(1j * 2 * np.pi * center_freq_hz * wavelet_time) * gaussian
    wavelet_fft = np.fft.fft(wavelet)
    wavelet_hz = np.linspace(0, wavelet_fs / 2, int(np.floor(len(wavelet) / 2) + 1))
    wavelet_power = np.abs(wavelet_fft[:len(wavelet_hz)]) ** 2

    plt.figure(figsize=(3, 3), dpi=300)
    plt.plot(wavelet_time, np.real(wavelet), label='Real part')
    plt.plot(wavelet_time, np.imag(wavelet), label='Imag part')
    plt.plot(wavelet_time, np.abs(wavelet), ls='--', c='k', label='Magnitude')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend(frameon=False, loc='upper right')
    sns.despine(trim=False)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(3, 3), dpi=300)
    plt.plot(wavelet_hz, wavelet_power, 'k-')
    plt.axvline(center_freq_hz, color='tab:red', ls='--', lw=1)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    sns.despine(trim=False)
    plt.tight_layout()
    plt.show()

gaussian_wavelet(wavelet_fwhm=0.5) # A wavelet with a FWHM of 0.5 s is pretty wide in the time domain, and narrow in the frequency domain.
# gaussian_wavelet(wavelet_fwhm=0.1) # A wavelet with a FWHM of 0.1 s is much narrower in the time domain, but much wider in the frequency domain. This illustrates the time-frequency tradeoff of wavelets: narrower wave

#%% The wavelets to find spectrograms

wavelet_window_s=0.5
fwhm = 0.4
fs = 30000
freqs_hz = np.linspace(0, 100, 100)

wavetime = np.arange(-wavelet_window_s, wavelet_window_s, 1 / fs)
gaussian = np.exp(-(4 * np.log(2) * wavetime**2) / fwhm**2)
wavelets = np.zeros((len(freqs_hz), len(wavetime)), dtype=complex)
for freq_idx, freq_hz in enumerate(freqs_hz):
    wavelets[freq_idx] = np.exp(1j * 2 * np.pi * freq_hz * wavetime) * gaussian

#%%
ix = 60

plt.figure(figsize=(3, 3), dpi=300)
plt.plot(wavetime, np.real(wavelets[ix]), label='Real')
plt.plot(wavetime, np.imag(wavelets[ix]), label='Imag')
plt.plot(wavetime, np.abs(wavelets[ix]), ls='--', c='k', label='Magnitude')
plt.title(f'Wavelet for {freqs_hz[ix]:.1f} Hz\nFWHM {fwhm}s', fontsize=8)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend(frameon=False, loc='lower left')
sns.despine(trim=False)
plt.tight_layout()
plt.show()

plt.figure(figsize=(3, 3), dpi=300)
wavelet_fft = np.fft.fft(wavelets[ix])
wavelet_hz = np.linspace(0, fs / 2, int(np.floor(len(wavelets[ix]) / 2) + 1))
wavelet_power = np.abs(wavelet_fft[:len(wavelet_hz)]) ** 2
plt.plot(wavelet_hz, wavelet_power, 'k-')
plt.xlim([0, 100])
plt.axvline(freqs_hz[ix], color='r', ls='--')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title(f'Power spectrum for {freqs_hz[ix] :.1f} Hz\nFWHM {fwhm}s', fontsize=8)
sns.despine(trim=False)
plt.tight_layout()
plt.show()
#%%
plt.figure(figsize=(4,3), dpi=300)
plt.pcolormesh(wavetime, freqs_hz, np.real(wavelets), cmap='coolwarm')
plt.title(f'Real part of wavelets\nWindow width {wavelet_window_s}s, FWHM {fwhm}s\ngranularity {len(freqs_hz)} freqs', fontsize=8)
plt.xticks(np.arange(-wavelet_window_s, wavelet_window_s + 0.1, 0.2), fontsize=8)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.colorbar()
sns.despine(trim=False)
plt.tight_layout()
plt.show()
# %%

plt.figure(figsize=(3,3), dpi=300)
for i in range(len(wavelets)):
    plt.plot(wavetime, np.abs(wavelets[i]))
plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()
# %%
