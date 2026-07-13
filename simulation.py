import numpy as np


def gaussian(freqs_hz, center_hz, sd_hz):
    return np.exp(-0.5 * ((freqs_hz - center_hz) / sd_hz) ** 2)


def target_psd(freqs_hz, exponent=-1.5, offset=0.0, peak_center_hz=10,
               peak_sd_hz=2, peak_height=1.0):
    f = np.maximum(freqs_hz, 0.5)
    log_power = offset + exponent * np.log10(f)
    log_power += peak_height * gaussian(freqs_hz, peak_center_hz, peak_sd_hz)
    return 10 ** log_power


def signal_from_psd(psd, n_samples, rng):
    amps = np.sqrt(psd)
    phases = rng.uniform(0, 2 * np.pi, len(psd))
    phases[0] = 0
    phases[-1] = 0

    sig = np.fft.irfft(amps * np.exp(1j * phases), n=n_samples)
    sig -= sig.mean()
    sig /= sig.std()
    return sig