#%%
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import seaborn as sns

from preprocess import LFP_processor, get_ns5_path, get_output_mat_path


SUBJECT = 'YFT'
EMU_ID = 32
REGIONS = ['HPC']
RAW_WINDOW_S = 5
DS_WINDOW_S = 5

# Sanity-check raw and saved downsampled LFP for one session.

subject=SUBJECT
emu_id=EMU_ID
regions=REGIONS
raw_window_s=RAW_WINDOW_S
ds_window_s=DS_WINDOW_S
ns5_path = get_ns5_path(subject, emu_id)
ds_mat_path = get_output_mat_path(ns5_path)

lfp = LFP_processor(ns5_path, regions=regions)
if len(lfp.chosen_channel_ids) == 0:
    raise ValueError(f'No channels found for regions={regions} in {ns5_path}')

channel_id = lfp.chosen_channel_ids[0]
channel_name = lfp.chosen_chan.iloc[0]['channel_name']

raw_end = min(int(raw_window_s * lfp.fs), lfp.n_samples)
raw_trace = lfp.recording.get_traces(
    channel_ids=[channel_id],
    segment_index=lfp.segment_index,
    start_frame=0,
    end_frame=raw_end,
).squeeze()
raw_time = np.arange(raw_trace.shape[0]) / lfp.fs

ds_mat = scipy.io.loadmat(ds_mat_path, squeeze_me=True)
ds_lfp = np.asarray(ds_mat['lfp_ds'])
if ds_lfp.ndim == 1:
    ds_trace = ds_lfp
else:
    ds_trace = ds_lfp[:min(int(ds_window_s * float(ds_mat['fs'])), ds_lfp.shape[0]), 0]
ds_fs = float(ds_mat['fs'])
ds_time = np.arange(ds_trace.shape[0]) / ds_fs
#%%
fig, ax = plt.subplots(1,1, figsize=(6,3), sharex=False, dpi=300)
ax.plot(raw_time, raw_trace, lw=0.8)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')

ax.plot(ds_time, ds_trace, lw=0.8, color='tab:orange')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
fig.tight_layout()
sns.despine(trim=False)


#%%
# This file is intentionally parked as a reference-only scratchpad for now.
# The code below is legacy LFP analysis logic copied from older work.
# Nothing in this file should execute until we rebuild it into a clean,
# self-contained simulation script.


#%%
###### LEGACY REFERENCE CODE BELOW
###### KEEP FOR LOGIC / VISUAL IDEAS ONLY
###### NOT CURRENTLY USED FOR THE NEW SIMULATION
###### -----------------------------------------

### plot raw data
### plot filtered data

# Filtsample = signal.filtfilt(filterkern, 1, LFP[Ch])

# plt.figure(figsize=(12, 6))
# plt.plot(
#     np.linspace(-4.17, 3, 21500),
#     LFP[Ch, Point[0] - 12500:Point[0] + 9000] * 1e3,
#     c="b",
#     alpha=0.2,
# )
# plt.plot(
#     np.linspace(-4.17, 3, 21500),
#     Filtsample[Point[0] - 12500:Point[0] + 9000] * 1e3,
#     c="b",
# )
# plt.axvline(0, c="k", ls="--")
# plt.axvline(-1, c="k", ls="--")
# plt.ylabel("Voltage (mV)")
# plt.xlabel("Time (Second)")
# plt.title("Filtered data around a stimulation-aligned event", size=16)
# plt.xticks(
#     [-4, -3, -2, -1, 0, 1, 2, 3],
#     [-4, -3, -2, "-1\nBegin stim\nMay vary", "0\nEnd of stim", 1, 2, 3],
# )
# plt.show()


### plot instantaneous power

# Power = Filtsample[Point[1]:Point[1] + 9000] ** 2
# Smooth = np.zeros(Power.shape)

# win = 100
# for i in range(win + 1, len(Power) - win - 1):
#     Smooth[i] = np.mean(Power[i - win:i + win])

# plt.figure(figsize=(8, 4))
# plt.plot(np.linspace(0, 3, 9000), Power * 1e6, c="b", alpha=0.2)
# plt.plot(np.linspace(0, 3, 9000), Smooth * 1e6, c="r")
# plt.ylabel("Power (uV^2)")
# plt.xlabel("Time (Second)")
# plt.title("Smoothed beta-band power after stimulation", size=16)
# plt.xticks([0, 1, 2, 3], ["0\nEnd of stim", 1, 2, 3])
# plt.show()


### plot downsampled power
### Downsample after low-pass filtering to avoid aliasing

# Downsample_factor = 30
# New_SR = SR_LFP / Downsample_factor

# fkern = signal.firwin(int(14 * New_SR / 2), New_SR / 2, fs=SR_LFP, pass_zero=True)
# fsignal = signal.filtfilt(fkern, 1, Filtsample[Point[1]:Point[1] + 9000])
# signal_dsG = fsignal[:-1:Downsample_factor]

# plt.figure(figsize=(8, 4))
# plt.plot(
#     np.linspace(0, 3, 9000),
#     Filtsample[Point[1]:Point[1] + 9000] * 1e3,
#     "r--",
#     label="Original",
# )
# plt.plot(np.linspace(0, 3, 300), signal_dsG * 1e3, "b--", label="Downsampled")
# plt.xticks([0, 1, 2, 3], ["0\nEnd of stim", 1, 2, 3])
# plt.title("Downsampled filtered signal", size=16)
# plt.ylabel("Voltage (mV)")
# plt.xlabel("Time (Second)")
# plt.legend(loc="lower left")
# plt.show()


#%%
###### FILTER DESIGN REFERENCE
###### -----------------------

### Define filter parameters for beta wave

# lower_bnd = 12  # Hz
# upper_bnd = 30  # Hz
# lower_trans = 0.1
# upper_trans = 0.1
# samprate = 30000  # Hz
# filtorder = 3001

# filter_shape = [0, 0, 1, 1, 0, 0]
# filter_freqs = [
#     0,
#     lower_bnd * (1 - lower_trans),
#     lower_bnd,
#     upper_bnd,
#     upper_bnd + upper_bnd * upper_trans,
#     samprate / 2,
# ]

# filterkern = signal.firls(filtorder, filter_freqs, filter_shape, fs=samprate)
# hz = np.linspace(0, samprate / 2, int(np.floor(len(filterkern) / 2) + 1))
# filterpow = np.abs(scipy.fftpack.fft(filterkern)) ** 2

# plt.figure(figsize=(12, 4))
# plt.subplot(121)
# plt.plot(filterkern)
# plt.xlabel("Time points")
# plt.title("Filter kernel (firls)")

# plt.subplot(122)
# plt.plot(hz, filterpow[:len(hz)], "ks-")
# plt.plot(filter_freqs, filter_shape, "ro-")
# plt.xlim([0, upper_bnd + 20])
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Filter gain")
# plt.title("Frequency response")
# plt.show()


#%%
###### PSD REFERENCE
###### -------------

# c = ["#FF0000", "#FF7F00", "#D5B515", "#00FF00", "#0000FF", "#2E2B5F", "#8B00FF", "k"]
# plt.figure(figsize=(8, 4), frameon=False)
# for i in range(len(channel_subset)):
#     f, psd = signal.welch(
#         LFP_sub[i],
#         SR_LFP,
#         nperseg=1024,
#         window="hanning",
#         noverlap=512,
#         nfft=8192,
#     )
#     plt.plot(f, psd * 1e9, c=c[i], lw=2, label=f"Channel {(i + 1) * 10}")

# plt.xlabel("Frequency (Hz)")
# plt.ylabel("PSD ($nV^2$/Hz)")
# plt.xlim([0, 80])
# plt.title("PSD for sample channels")
# plt.legend(loc="upper right")
# plt.savefig("PSD.png")
# plt.show()


#%%
###### TIME-FREQUENCY REFERENCE
###### ------------------------

# nfrex = 100
# frex = np.linspace(0, 50, nfrex)
# fwhm = 0.3
# # Adjust this to see the waveform in the left figure.
# # Re-check this whenever the frequency range changes.

# wavetime = np.arange(-1, 1, 1 / SR_LFP)
# wavelets = np.zeros((nfrex, len(wavetime)), dtype=complex)

# for wi in range(nfrex):
#     gaussian = np.exp(-(4 * np.log(2) * wavetime**2) / fwhm**2)
#     wavelets[wi, :] = np.exp(1j * 2 * np.pi * frex[wi] * wavetime) * gaussian

# plt.figure(figsize=(6, 5))
# plt.plot(wavetime, np.real(wavelets[10, :]), label="Real part")
# plt.plot(wavetime, np.imag(wavelets[10, :]), label="Imag part")
# plt.xlabel("Time")
# plt.xlim([-0.5, 0.5])
# plt.title("Adjust fwhm to see different waveform.")
# plt.legend()
# plt.show()


### Compute the power for each frequency at each time point

# LFP_tf = LFP_sub[:, int(500 * SR_LFP):int(504 * SR_LFP)]
# data_len = LFP_tf.shape[1]
# tf = np.zeros((len(channel_subset), nfrex, data_len))

# for i in range(len(channel_subset)):
#     nconv = data_len + len(wavetime) - 1
#     halfk = int(np.floor(len(wavetime) / 2))

#     dataX = scipy.fftpack.fft(LFP_tf[i], nconv)

#     for fi in range(nfrex):
#         waveX = scipy.fftpack.fft(wavelets[fi, :], nconv)
#         waveX = waveX / np.max(waveX)
#         convres = scipy.fftpack.ifft(waveX * dataX)
#         convres = convres[halfk - 1:-halfk]
#         tf[i, fi, :] = np.abs(convres) ** 2


### Spectrogram

# plt.figure(figsize=(12, 7.5), frameon=False)
# for i in range(8):
#     plt.subplot(2, 4, i + 1)
#     plt.pcolormesh(range(data_len), frex, tf[i] * 1e9, vmax=1, vmin=0, cmap="hot")
#     plt.colorbar(orientation="horizontal", aspect=75, pad=0.2)
#     plt.title(f"Channel {(i + 1) * 10}")
#     plt.xticks(np.linspace(0, int(4 * SR_LFP), 5), [0, 1, 2, 3, 4])
#     plt.xlabel("Time (s)")
#     if i in [1, 2, 3, 5, 6, 7]:
#         plt.yticks([])
#     else:
#         plt.yticks(np.arange(0, 51, 10))
#         plt.ylabel("Frequency (Hz)")
# plt.subplots_adjust(wspace=0.15, hspace=0.05)
# plt.savefig("Spectrogram.png")
