import numpy as np


#%% Functions for simulating oscillatory signals with time-varying envelopes

def _build_smooth_noise(time, random_state, scale, window_s):
    if scale == 0:
        return np.zeros(time.shape[0])

    dt = time[1] - time[0]
    n_samples = time.shape[0]
    noise = random_state.standard_normal(n_samples)
    smooth_window_samples = max(int(round(window_s / dt)), 1)
    kernel = np.ones(smooth_window_samples, dtype=float)
    kernel /= kernel.sum()
    pad = smooth_window_samples // 2
    noise_pad = np.pad(noise, pad_width=pad, mode='wrap')
    smooth_noise = np.convolve(noise_pad, kernel, mode='valid')
    smooth_noise = smooth_noise[:n_samples]
    smooth_noise -= smooth_noise.mean()

    noise_std = smooth_noise.std()
    if noise_std > 0:
        smooth_noise /= noise_std

    return scale * smooth_noise


def build_oscillation_envelopes(sim_time, rng, base_amplitudes, envelope_mode,
                                envelope_scales, smooth_window_s):
    base_amplitudes = np.asarray(base_amplitudes, dtype=float)
    envelope_scales = np.asarray(envelope_scales, dtype=float)

    if base_amplitudes.shape != envelope_scales.shape:
        raise ValueError('base_amplitudes and envelope_scales must have the same shape.')

    envelopes = np.repeat(base_amplitudes[:, None], sim_time.shape[0], axis=1)
    if envelope_mode not in {'constant', 'ind', 'shared'}:
        raise ValueError("envelope_mode must be 'constant', 'ind', or 'shared'.")

    if envelope_mode != 'constant':
        if envelope_mode == 'ind':
            for freq_idx, scale in enumerate(envelope_scales):
                envelopes[freq_idx] += _build_smooth_noise(sim_time, rng, scale, smooth_window_s)
        else:
            shared_noise = _build_smooth_noise(sim_time, rng, scale=1, window_s=smooth_window_s)
            envelopes += envelope_scales[:, None] * shared_noise[None, :]

    return np.clip(envelopes, a_min=0, a_max=None)


def generate_oscillation_trace(sim_time, rng, freqs_hz, base_amplitudes, phases_rad, envelope_mode,
                               envelope_scales, smooth_window_s, additive_noise_sd=0):
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    base_amplitudes = np.asarray(base_amplitudes, dtype=float)
    phases_rad = np.asarray(phases_rad, dtype=float)
    envelope_scales = np.asarray(envelope_scales, dtype=float)

    if len({arr.shape for arr in (freqs_hz, base_amplitudes, phases_rad, envelope_scales)}) != 1:
        raise ValueError('All per-frequency parameter arrays must have the same shape.')

    envelopes = build_oscillation_envelopes(
        sim_time=sim_time,
        rng=rng,
        base_amplitudes=base_amplitudes,
        envelope_mode=envelope_mode,
        envelope_scales=envelope_scales,
        smooth_window_s=smooth_window_s,
    )
    carriers = np.sin(2 * np.pi * freqs_hz[:, None] * sim_time[None, :] + phases_rad[:, None])
    sim_trace = np.sum(envelopes * carriers, axis=0)
    if additive_noise_sd > 0:
        sim_trace += additive_noise_sd * rng.standard_normal(sim_time.shape[0])
    return sim_trace, envelopes


#%% Functions for simulating PSD-shaped noise traces

def generate_pinknoise_trace(sim_time, rng, alpha, target_sd, mean, fmin_hz):
    """Generate one stationary 1/f^alpha trace by shaping Fourier coefficients.

    The construction follows the standard frequency-domain recipe for a
    pink-noise-like background:

    1. Build the one-sided rFFT frequency grid from the sample spacing.
    2. Define the target background power spectrum on that grid as
       P(f) ~ 1 / max(f, fmin_hz)^alpha for positive frequencies.
       The max(..., fmin_hz) floor prevents the singularity at f = 0.
    3. Convert target power into Fourier magnitudes using
       |X(f)| ~ sqrt(P(f)) ~ 1 / max(f, fmin_hz)^(alpha / 2).
    4. Assign each positive-frequency bin a random phase drawn uniformly from
       [0, 2*pi). DC is held at zero, and the Nyquist bin stays real-valued
       when the trace length is even.
    5. Apply the inverse real FFT to obtain a real-valued time series.
    6. Demean and rescale the trace to the requested standard deviation, then
       add the requested mean offset.

    Parameters
    ----------
    sim_time : ndarray
        Simulation time vector in seconds. Its spacing determines the FFT
        frequency grid and therefore the lowest nonzero resolvable frequency.
    rng : numpy.random.Generator
        Random number generator used for phase sampling.
    alpha : float
        Power-law exponent in the target PSD. alpha = 1 gives pink noise.
    target_sd : float
        Desired standard deviation of the output trace after normalization.
    mean : float
        Constant offset added after scaling.
    fmin_hz : float or None
        Optional low-frequency floor for the 1/f^alpha law. If None, use the
        first nonzero FFT bin.

    Returns
    -------
    trace : ndarray
        Real-valued pink-noise trace sampled on sim_time.
    fft_freqs_hz : ndarray
        One-sided FFT frequency grid corresponding to the generated spectrum.
    target_power : ndarray
        Idealized one-sided target PSD shape, normalized to 1 at the first
        positive-frequency bin.
    effective_fmin_hz : float
        The low-frequency floor actually used during spectral shaping.
    """
    sim_time = np.asarray(sim_time, dtype=float)
    if sim_time.ndim != 1 or sim_time.size < 2:
        raise ValueError('sim_time must be a one-dimensional array with at least 2 samples.')
    if alpha < 0:
        raise ValueError('pink_alpha must be non-negative.')
    if target_sd < 0:
        raise ValueError('pink_sd must be non-negative.')

    dt = float(sim_time[1] - sim_time[0])
    if dt <= 0:
        raise ValueError('sim_time must be strictly increasing.')

    # Build the one-sided FFT frequency grid used by rFFT / irFFT.
    # The first nonzero bin sets the natural low-frequency resolution of the trace.
    n_samples = sim_time.size
    fft_freqs_hz = np.fft.rfftfreq(n_samples, d=dt)
    if fft_freqs_hz.size < 2:
        raise ValueError('Need at least one positive FFT bin to generate pink noise.')

    min_positive_hz = float(fft_freqs_hz[1])
    if fmin_hz is None:
        effective_fmin_hz = min_positive_hz
    else:
        effective_fmin_hz = float(fmin_hz)

    max_fft_hz = float(fft_freqs_hz[-1])
    if effective_fmin_hz <= 0:
        raise ValueError('pink_fmin_hz must be positive.')
    if effective_fmin_hz > max_fft_hz:
        raise ValueError(
            f'pink_fmin_hz ({effective_fmin_hz:g} Hz) exceeds the maximum FFT bin ({max_fft_hz:g} Hz).'
        )

    # Shape the Fourier magnitudes so the implied power spectrum follows 1/f^alpha.
    # Because power is squared magnitude, the magnitude scales as 1/f^(alpha/2).
    amplitudes = np.zeros_like(fft_freqs_hz)
    positive = fft_freqs_hz > 0
    amplitudes[positive] = 1.0 / np.power(
        np.maximum(fft_freqs_hz[positive], effective_fmin_hz),
        alpha / 2.0,
    )

    # Populate the one-sided complex spectrum with random phases.
    # DC stays at zero, the Nyquist bin must stay real when n_samples is even,
    # and all other positive-frequency bins get independent random phases.
    spectrum = np.zeros_like(fft_freqs_hz, dtype=complex)
    if n_samples % 2 == 0:
        interior = slice(1, -1)
        spectrum[-1] = amplitudes[-1] * rng.choice((-1.0, 1.0))
    else:
        interior = slice(1, None)

    n_interior = spectrum[interior].size
    if n_interior > 0:
        phases = rng.uniform(0.0, 2.0 * np.pi, size=n_interior)
        spectrum[interior] = amplitudes[interior] * np.exp(1j * phases)

    # Transform back to the time domain and normalize to the requested mean / SD.
    trace = np.fft.irfft(spectrum, n=n_samples)
    trace -= trace.mean()
    if target_sd == 0:
        trace.fill(0.0)
    else:
        trace_std = trace.std()
        if trace_std == 0:
            raise RuntimeError('Generated pink-noise trace has zero variance before scaling.')
        trace *= target_sd / trace_std
    trace += mean

    # Save the idealized target PSD shape for downstream inspection.
    # It is normalized to 1 at the first positive-frequency bin.
    target_power = np.zeros_like(fft_freqs_hz)
    target_power[positive] = 1.0 / np.power(
        np.maximum(fft_freqs_hz[positive], effective_fmin_hz),
        alpha,
    )
    if target_power[positive].size > 0:
        target_power[positive] /= target_power[positive][0]

    return trace, fft_freqs_hz, target_power, effective_fmin_hz
