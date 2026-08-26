"""Notebook-style replication of Wiest et al. (2023), Figure 2.

Execute this file one ``# %%`` cell at a time.  There is deliberately no command
line interface, configuration class, or output-writing pipeline.  One paired
hemisphere is analyzed immediately.  Set ``RUN_FULL_COHORT = True`` in the
cohort cell when ready to process the strict same-location subset.

Paper method reproduced here
----------------------------
* Author-selected resting STN-LFP channel and approximately 60 s interval
* Linear detrending
* 1-90 Hz power from 50-cycle complex Morlet wavelets
* Interpolation of 50 Hz mains bins and 1-90 Hz mean-power normalization
* FOOOF 1.0.0 fixed-mode exponent from 40-90 Hz
* FOOOF knee-mode largest periodic beta peak from 8-35 Hz
* Paired sign-flip comparisons and Figure 2-style correlations on that subset
"""

# %% Imports and paths
from pathlib import Path
import gc

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import io, signal, stats

try:
    from fooof import FOOOF
    import fooof
except ImportError as error:
    raise ImportError(
        "Figure 2 replication requires FOOOF 1.0.0. Run `uv sync`, then "
        "restart the notebook kernel."
    ) from error


REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = REPO_ROOT / "data" / "wiest"
OFF_DIR = DATA_ROOT / "aperiodic_stn_meds_off"
ON_PARENT = DATA_ROOT / "aperiodic_stn_meds_on"

on_candidates = [ON_PARENT]
if ON_PARENT.exists():
    on_candidates.extend(path for path in ON_PARENT.iterdir() if path.is_dir())
ON_DIR = next(
    (path for path in on_candidates if any(path.glob("*_ON.mat"))),
    ON_PARENT,
)


def section(title):
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92)


# %% Explicit replacement for the authors' positional channel/time arrays
### EXPLAIN THE HARD CODING
# The released Med_OFFs.m and Med_ONs.m files contain three separate positional
# objects: an alphabetically ordered file list, a MATLAB 1-indexed channel
# vector, and a start/stop-time matrix.  There is no participant metadata table
# joining them.  PAIR_SPECS makes that implicit positional join explicit:
#
#   1. Match row N of each channel/time array to file N in the author's order.
#   2. Subtract one from each MATLAB channel number to store a Python 0-indexed
#      row of SmrData.WvData as channel_ix_off/channel_ix_on.
#   3. Read that row's SmrData.WvTits entry and store its exact uploaded label
#      as physical_loc_off/physical_loc_on (for example, L02 or R13).
#   4. Keep the authors' time boundaries unchanged.  Their MATLAB slice
#      start*fs:stop*fs includes both endpoints; load_selected_segment preserves
#      that behavior.
#
# This is deliberately hard-coded provenance, not a new data-driven channel
# selection.  PAIR_SPECS retains all 30 published selections.  pair_specs below
# is the analysis DataFrame: it keeps only matching, explicitly identified
# physical labels.  Generic "Virtual" labels are excluded because equality of
# that string cannot prove that the underlying contact pair is the same.
PAIR_SPECS = pd.DataFrame(
    [
        ("G10_leSTN", "G10_leSTN_OFF.mat", 8, "Le02", 760, 820, "G10_leSTN_ON.mat", 8, "Le02", 1, 61),
        ("G10_riSTN", "G10_riSTN_OFF.mat", 10, "R02", 2360, 2420, "G10_riSTN_ON.mat", 9, "R02", 1, 61),
        ("G23_leSTN", "G23_leSTN_OFF.mat", 9, "L13", 1, 61, "G23_leSTN_ON.mat", 9, "L13", 70, 130),
        ("G23_riSTN", "G23_riSTN_OFF.mat", 10, "R02", 1550, 1610, "G23_riSTN_ON.mat", 10, "R02", 1460, 1520),
        ("G24_leSTN", "G24_leSTN_OFF.mat", 8, "L02", 1, 61, "G24_leSTN_ON.mat", 8, "L02", 1, 61),
        ("G24_riSTN", "G24_riSTN_OFF.mat", 10, "R02", 1, 61, "G24_riSTN_ON.mat", 10, "R02", 1, 61),
        ("G25_leSTN", "G25_leSTN_OFF.mat", 0, "L24", 1, 61, "G25_leSTN_ON.mat", 0, "L24", 1, 61),
        ("G25_riSTN", "G25_riSTN_OFF.mat", 0, "R24", 1, 61, "G25_riSTN_ON.mat", 0, "R24", 1, 61),
        ("G27_leSTN", "G27_leSTN_OFF.mat", 0, "L13", 1, 61, "G27_leSTN_ON.mat", 29, "Lv13", 277, 337),
        ("G27_riSTN", "G27_riSTN_OFF.mat", 0, "R24", 1, 61, "G27_riSTN_ON.mat", 29, "Rv13", 1, 61),
        ("G28_leSTN", "G28_leSTN_OFF.mat", 6, "L02", 1, 61, "G28_leSTN_ON.mat", 13, "L02", 1, 61),
        ("G28_riSTN", "G28_riSTN_OFF.mat", 9, "R13", 1, 61, "G28_riSTN_ON.mat", 13, "R24", 230, 290),
        ("G30_leSTN", "G30_leSTN_OFF.mat", 8, "L02", 1, 61, "G30_leSTN_ON.mat", 17, "L24", 1, 61),
        ("G30_riSTN", "G30_riSTN_OFF.mat", 6, "R02", 1, 61, "G30_riSTN_ON.mat", 17, "R13", 1, 61),
        ("G31_leSTN", "G31_leSTN_OFF.mat", 6, "L02", 1, 61, "G31_leSTN_ON.mat", 14, "L13", 1, 59),
        ("G31_riSTN", "G31_riSTN_OFF.mat", 9, "R13", 1, 61, "G31_riSTN_ON.mat", 14, "R13", 1, 61),
        ("G32_leSTN", "G32_leSTN_OFF.mat", 5, "L13", 1, 61, "G32_leSTN_ON.mat", 22, "L13", 1, 61),
        ("G32_riSTN", "G32_riSTN_OFF.mat", 6, "R02", 1, 61, "G32_riSTN_ON.mat", 22, "R13", 1, 61),
        ("G33_leSTN", "G33_leSTN_OFF.mat", 19, "Virtual", 1, 61, "G33_leSTN_ON.mat", 24, "Virtual", 1, 61),
        ("G33_riSTN", "G33_riSTN_OFF.mat", 21, "Virtual", 1, 61, "G33_riSTN_ON.mat", 26, "Virtual", 1, 61),
        ("G34_leSTN", "G34_leSTN_OFF.mat", 23, "Virtual", 200, 260, "G34_leSTN_ON.mat", 23, "Virtual", 200, 260),
        ("G34_riSTN", "G34_riSTN_OFF.mat", 25, "Virtual", 200, 260, "G34_riSTN_ON.mat", 25, "Virtual", 200, 260),
        ("K11_riSTN", "K11_riSTN_OFF.mat", 2, "R02", 1, 61, "K11_riSTN_ON.mat", 7, "R13", 1, 61),
        ("K6_leSTN", "K6_leSTN_OFF.mat", 15, "Virtual", 1, 61, "K6_leSTN_ON.mat", 15, "Virtual", 1, 61),
        ("K6_riSTN", "K6_riSTN_OFF.mat", 18, "Virtual", 1, 61, "K6_riSTN_ON.mat", 18, "Virtual", 1, 61),
        ("K7_leSTN", "K7_leSTN_OFF.mat", 11, "L13", 1, 61, "K7_leSTN_ON.mat", 7, "L13", 1, 61),
        ("K8_leSTN", "K8_leSTN_OFF.mat", 6, "L02", 70, 130, "K8_leSTN_ON.mat", 13, "L02", 1, 61),
        ("K8_riSTN", "K8_riSTN_OFF.mat", 8, "R02", 70, 130, "K8_riSTN_ON.mat", 13, "R02", 1, 61),
        ("XG37_leSTN", "XG37_leSTN_OFF.mat", 8, "L24", 1, 61, "XG37_ERNA_leSTN_ON.mat", 19, "L24", 1, 61),
        ("XG39_riSTN", "XG39_riSTN_OFF.mat", 7, "R13", 1, 61, "XG39_ERNA_riSTN_ON.mat", 17, "R13", 1, 61),
    ],
    columns=(
        "hemisphere",
        "file_off",
        "channel_ix_off",
        "physical_loc_off",
        "start_s_off",
        "stop_s_off",
        "file_on",
        "channel_ix_on",
        "physical_loc_on",
        "start_s_on",
        "stop_s_on",
    ),
)
PAIR_SPECS.index.name = "author_pair_ix"
PAIR_SPECS = PAIR_SPECS.reset_index()

same_location = PAIR_SPECS["physical_loc_off"].eq(
    PAIR_SPECS["physical_loc_on"]
)
specific_location = ~PAIR_SPECS["physical_loc_off"].eq("Virtual")
keep_same_location = same_location & specific_location

pair_specs = PAIR_SPECS.loc[keep_same_location].reset_index(drop=True).copy()
excluded_pair_specs = PAIR_SPECS.loc[~keep_same_location].reset_index(drop=True).copy()

section("1. AUTHOR SELECTIONS AND STRICT SAME-LOCATION FILTER")
print(f"Published hemisphere pairs:       {len(PAIR_SPECS)}")
print(f"Retained same-location pairs:     {len(pair_specs)}")
print(f"Excluded mismatch/unknown pairs:  {len(excluded_pair_specs)}")
print("\nRetained pairs:")
print(
    pair_specs[
        [
            "hemisphere",
            "channel_ix_off",
            "physical_loc_off",
            "channel_ix_on",
            "physical_loc_on",
        ]
    ].to_string(index=False)
)
# %% MAT readers: slice v7.3 files; load legacy v5 files
def decode_hdf5_chars(dataset):
    return "".join(chr(int(value)) for value in dataset[()].reshape(-1))


def linear_trend_coefficients(values):
    """Return intercept-at-center and slope without an ill-conditioned fit."""
    values = np.asarray(values, dtype=float)
    center = (values.size - 1) / 2
    centered_samples = np.arange(values.size, dtype=float) - center
    intercept = values.mean()
    slope = np.dot(centered_samples, values) / np.dot(
        centered_samples, centered_samples
    )
    return intercept, slope


def hdf5_linear_trend_coefficients(dataset, channel_ix, chunk_size=1_000_000):
    """Fit the whole HDF5 channel in chunks, as MATLAB detrend did."""
    n_samples = dataset.shape[0]
    center = (n_samples - 1) / 2
    value_sum = 0.0
    centered_product_sum = 0.0

    for chunk_start in range(0, n_samples, chunk_size):
        chunk_stop = min(chunk_start + chunk_size, n_samples)
        values = np.asarray(
            dataset[chunk_start:chunk_stop, channel_ix], dtype=float
        )
        samples = np.arange(chunk_start, chunk_stop, dtype=float) - center
        value_sum += values.sum(dtype=float)
        centered_product_sum += np.dot(samples, values)

    intercept = value_sum / n_samples
    centered_square_sum = n_samples * (n_samples**2 - 1) / 12
    slope = centered_product_sum / centered_square_sum
    return intercept, slope


def load_selected_segment(
    path, channel_ix, expected_physical_loc, start_s, stop_s
):
    """Load exactly the channel/window selected by the released MATLAB code."""
    channel_ix = int(channel_ix)
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as mat:
            fs = float(mat["SmrData/Fs"][0, 0])
            wave_data = mat["SmrData/WvData"]  # HDF storage: samples x channels
            source_samples = wave_data.shape[0]
            titles = mat["SmrData/WvTits"][()].reshape(-1)
            physical_loc = decode_hdf5_chars(mat[titles[channel_ix]])

            # MATLAB used start*fs:stop*fs, including both endpoints.
            first_sample = int(round(start_s * fs)) - 1
            stop_sample = int(round(stop_s * fs))
            trace = np.asarray(
                wave_data[first_sample:stop_sample, channel_ix], dtype=float
            )
            intercept, slope = hdf5_linear_trend_coefficients(
                wave_data, channel_ix
            )
        mat_format = "v7.3/HDF5"
    else:
        loaded = io.loadmat(path, struct_as_record=False, squeeze_me=True)
        smr_data = loaded["SmrData"]
        fs = float(smr_data.Fs)
        wave_data = np.asarray(smr_data.WvData)
        titles = np.asarray(smr_data.WvTits, dtype=object).reshape(-1)
        physical_loc = str(titles[channel_ix])

        first_sample = int(round(start_s * fs)) - 1
        stop_sample = int(round(stop_s * fs))
        if wave_data.ndim == 1:
            if channel_ix != 0:
                raise IndexError(
                    f"{path.name} has one channel, requested index {channel_ix}."
                )
            whole_channel = np.asarray(wave_data, dtype=float)
        else:
            whole_channel = np.asarray(wave_data[channel_ix], dtype=float)
        source_samples = whole_channel.size
        intercept, slope = linear_trend_coefficients(whole_channel)
        trace = whole_channel[first_sample:stop_sample].copy()
        del loaded, smr_data, wave_data
        gc.collect()
        mat_format = "v5"

    if physical_loc != expected_physical_loc:
        raise ValueError(
            f"Metadata mismatch for {path.name}: row {channel_ix} is "
            f"{physical_loc!r}, expected {expected_physical_loc!r}."
        )

    # The released scripts detrend the entire channel before selecting the window.
    selected_samples = np.arange(first_sample, stop_sample, dtype=float)
    source_center = (source_samples - 1) / 2
    trace -= intercept + slope * (selected_samples - source_center)
    return {
        "path": path,
        "trace": trace,
        "fs": fs,
        "source_samples": source_samples,
        "channel_ix": channel_ix,
        "physical_loc": physical_loc,
        "start_s": start_s,
        "stop_s": stop_s,
        "mat_format": mat_format,
    }


# %% Load one paired hemisphere immediately
PAIR_INDEX = 0  # Change this integer (0 to len(pair_specs)-1) to inspect another pair.
selected_spec = pair_specs.iloc[PAIR_INDEX]

off_recording = load_selected_segment(
    OFF_DIR / selected_spec["file_off"],
    selected_spec["channel_ix_off"],
    selected_spec["physical_loc_off"],
    selected_spec["start_s_off"],
    selected_spec["stop_s_off"],
)
on_recording = load_selected_segment(
    ON_DIR / selected_spec["file_on"],
    selected_spec["channel_ix_on"],
    selected_spec["physical_loc_on"],
    selected_spec["start_s_on"],
    selected_spec["stop_s_on"],
)


section("2. ONE PAIRED HEMISPHERE")

print(f"Hemisphere: {selected_spec['hemisphere']}")
for condition, recording in (("OFF", off_recording), ("ON", on_recording)):
    trace = recording["trace"]
    print(f"\n{condition}")
    print(f"  file:          {recording['path'].name}")
    print(f"  MAT format:    {recording['mat_format']}")
    print(
        f"  selected:      row index {recording['channel_ix']} "
        f"({recording['physical_loc']})"
    )
    print(f"  source window: {recording['start_s']}-{recording['stop_s']} s")
    print(f"  source length: {recording['source_samples'] / recording['fs']:.2f} s")
    print(f"  sampling rate: {recording['fs']:g} Hz")
    print(f"  samples:       {trace.size:,}")
    print(f"  duration:      {trace.size / recording['fs']:.6f} s")
    print(f"  mean / SD:     {trace.mean():.4f} / {trace.std():.4f} stored units")
    print(f"  finite:        {np.isfinite(trace).all()}")

print("\nThese are condition-paired resting snippets, not sample-aligned sessions.")
print("Their means need not be zero: the authors detrended each complete source")
print("channel before taking the selected interval, and this notebook does the same.")


# %% Plot five seconds of the detrended LFPs
plot_duration_s = 5

fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True, constrained_layout=True)
for axis, condition, recording, color in (
    (axes[0], "OFF levodopa", off_recording, "tab:purple"),
    (axes[1], "ON levodopa", on_recording, "0.45"),
):
    n_plot = min(int(plot_duration_s * recording["fs"]), recording["trace"].size)
    time_s = np.arange(n_plot) / recording["fs"]
    axis.plot(time_s, recording["trace"][:n_plot], color=color, linewidth=0.7)
    axis.set_ylabel("Stored units")
    axis.set_title(
        f"{selected_spec['hemisphere']} {condition}: "
        f"{recording['physical_loc']}"
    )
axes[-1].set_xlabel("Time within selected snippet (s)")


# %% Paper-matched 50-cycle Morlet spectrum
FREQUENCIES_HZ = np.arange(1.0, 91.0)
MORLET_WIDTH = 50
MORLET_GWIDTH = 3


def morlet_power_spectrum(trace, fs, frequencies_hz, label=""):
    """Average power using the FieldTrip-style Morlet definition in the paper."""
    trace = np.asarray(trace, dtype=float)
    power = np.full(len(frequencies_hz), np.nan)

    for frequency_index, frequency_hz in enumerate(frequencies_hz):
        spectral_sd_hz = frequency_hz / MORLET_WIDTH
        temporal_sd_s = 1 / (2 * np.pi * spectral_sd_hz)
        half_width_s = MORLET_GWIDTH * temporal_sd_s
        wavelet_time_s = np.arange(-half_width_s, half_width_s + 0.5 / fs, 1 / fs)
        amplitude = 1 / np.sqrt(temporal_sd_s * np.sqrt(np.pi))
        wavelet = (
            amplitude
            * np.exp(-(wavelet_time_s**2) / (2 * temporal_sd_s**2))
            * np.exp(2j * np.pi * frequency_hz * wavelet_time_s)
        )

        convolution = signal.fftconvolve(trace, wavelet, mode="same")
        convolution *= np.sqrt(2 / fs)

        # FieldTrip returns NaN where a complete wavelet does not fit.
        edge_samples = len(wavelet_time_s) // 2
        if 2 * edge_samples >= convolution.size:
            continue
        valid = convolution[edge_samples:-edge_samples]
        power[frequency_index] = np.mean(np.abs(valid) ** 2)

        if label and frequency_hz % 10 == 0:
            print(f"  {label}: finished {frequency_hz:g} Hz")

    return power


def interpolate_mains_and_normalize(power, frequencies_hz):
    """Follow released medication code: interpolate 47-54 Hz, then normalize."""
    spectrum = np.asarray(power, dtype=float).copy()
    mains_mask = (frequencies_hz >= 47) & (frequencies_hz <= 54)
    clean_mask = ~mains_mask & np.isfinite(spectrum)
    spectrum[mains_mask] = np.interp(
        frequencies_hz[mains_mask], frequencies_hz[clean_mask], spectrum[clean_mask]
    )
    spectrum /= np.mean(spectrum)
    return spectrum


section("3. ONE-PAIR MORLET SPECTRA")
print("Computing 1-90 Hz spectra. Progress is printed every 10 Hz.")

off_power_raw = morlet_power_spectrum(
    off_recording["trace"], off_recording["fs"], FREQUENCIES_HZ, "OFF"
)
on_power_raw = morlet_power_spectrum(
    on_recording["trace"], on_recording["fs"], FREQUENCIES_HZ, "ON"
)
off_spectrum = interpolate_mains_and_normalize(off_power_raw, FREQUENCIES_HZ)
on_spectrum = interpolate_mains_and_normalize(on_power_raw, FREQUENCIES_HZ)

print(f"off_spectrum : shape={off_spectrum.shape}, normalized 1-90 Hz power")
print(f"on_spectrum  : shape={on_spectrum.shape}, normalized 1-90 Hz power")


# %% FOOOF 1.0.0 fits: fixed exponent and knee-mode periodic beta
def fit_figure2_spectrum(spectrum, frequencies_hz):
    exponent_model = FOOOF(
        peak_width_limits=[2, 12],
        max_n_peaks=np.inf,
        min_peak_height=0,
        peak_threshold=2,
        aperiodic_mode="fixed",
        verbose=False,
    )
    exponent_model.fit(frequencies_hz, spectrum, [40, 90])

    beta_model = FOOOF(
        peak_width_limits=[2, 12],
        max_n_peaks=np.inf,
        min_peak_height=0,
        peak_threshold=2,
        aperiodic_mode="knee",
        verbose=False,
    )
    beta_model.fit(frequencies_hz, spectrum, [5, 90])
    beta_peaks = np.asarray(beta_model.peak_params_).reshape(-1, 3)
    beta_peaks = beta_peaks[
        (beta_peaks[:, 0] > 8) & (beta_peaks[:, 0] < 35)
    ]
    if beta_peaks.size:
        largest_beta_peak = beta_peaks[np.argmax(beta_peaks[:, 1])]
        beta_frequency_hz = float(largest_beta_peak[0])
        periodic_beta_power = float(largest_beta_peak[1])
    else:
        beta_frequency_hz = np.nan
        periodic_beta_power = np.nan

    return {
        "exponent_model": exponent_model,
        "exponent": float(exponent_model.aperiodic_params_[1]),
        "r_squared": float(exponent_model.r_squared_),
        "beta_model": beta_model,
        "beta_frequency_hz": beta_frequency_hz,
        "periodic_beta_power": periodic_beta_power,
    }


off_fit = fit_figure2_spectrum(off_spectrum, FREQUENCIES_HZ)
on_fit = fit_figure2_spectrum(on_spectrum, FREQUENCIES_HZ)


section("4. ONE-PAIR FOOOF RESULTS")
print(f"FOOOF version:       {fooof.__version__} (paper used 1.0.0)")
print(f"OFF exponent:        {off_fit['exponent']:.4f}")
print(f"ON exponent:         {on_fit['exponent']:.4f}")
print(f"ON - OFF exponent:   {on_fit['exponent'] - off_fit['exponent']:.4f}")
print(f"OFF fit R-squared:   {off_fit['r_squared']:.4f}")
print(f"ON fit R-squared:    {on_fit['r_squared']:.4f}")
print(f"OFF periodic beta:   {off_fit['periodic_beta_power']:.4f} at {off_fit['beta_frequency_hz']:.2f} Hz")
print(f"ON periodic beta:    {on_fit['periodic_beta_power']:.4f} at {on_fit['beta_frequency_hz']:.2f} Hz")

fig, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
axis.loglog(FREQUENCIES_HZ, off_spectrum, color="tab:purple", label="OFF levodopa")
axis.loglog(FREQUENCIES_HZ, on_spectrum, color="0.5", label="ON levodopa")
axis.loglog(
    off_fit["exponent_model"].freqs,
    10 ** off_fit["exponent_model"]._ap_fit,
    color="tab:purple",
    linestyle="--",
    label=f"OFF 40-90 Hz fit, exponent={off_fit['exponent']:.3f}",
)
axis.loglog(
    on_fit["exponent_model"].freqs,
    10 ** on_fit["exponent_model"]._ap_fit,
    color="black",
    linestyle="--",
    label=f"ON 40-90 Hz fit, exponent={on_fit['exponent']:.3f}",
)
axis.axvspan(47, 54, color="tab:red", alpha=0.08, label="interpolated mains bins")
axis.set(xlabel="Frequency (Hz)", ylabel="Normalized power", xlim=(1, 90))
axis.set_title(f"Wiest Figure 2 method: {selected_spec['hemisphere']}")
axis.legend(fontsize=8)


# %% Full Figure 2 cohort -- deliberately opt in because this is slow
RUN_FULL_COHORT = True

cohort_rows = []
cohort_off_spectra = None
cohort_on_spectra = None

if RUN_FULL_COHORT:
    section("5. STRICT SAME-LOCATION FIGURE 2 ANALYSIS")
    print(
        f"This computes {2 * len(pair_specs)} Morlet spectra and can take "
        "several minutes."
    )

    for pair_number, (_, spec) in enumerate(pair_specs.iterrows(), start=1):
        print(f"\nPair {pair_number:02d}/{len(pair_specs)}: {spec['hemisphere']}")
        off = load_selected_segment(
            OFF_DIR / spec["file_off"],
            spec["channel_ix_off"],
            spec["physical_loc_off"],
            spec["start_s_off"],
            spec["stop_s_off"],
        )
        on = load_selected_segment(
            ON_DIR / spec["file_on"],
            spec["channel_ix_on"],
            spec["physical_loc_on"],
            spec["start_s_on"],
            spec["stop_s_on"],
        )

        off_raw = morlet_power_spectrum(off["trace"], off["fs"], FREQUENCIES_HZ)
        on_raw = morlet_power_spectrum(on["trace"], on["fs"], FREQUENCIES_HZ)
        off_spec = interpolate_mains_and_normalize(off_raw, FREQUENCIES_HZ)
        on_spec = interpolate_mains_and_normalize(on_raw, FREQUENCIES_HZ)
        off_result = fit_figure2_spectrum(off_spec, FREQUENCIES_HZ)
        on_result = fit_figure2_spectrum(on_spec, FREQUENCIES_HZ)

        cohort_rows.append(
            {
                "hemisphere": spec["hemisphere"],
                "off_exponent": off_result["exponent"],
                "on_exponent": on_result["exponent"],
                "off_r_squared": off_result["r_squared"],
                "on_r_squared": on_result["r_squared"],
                "off_periodic_beta": off_result["periodic_beta_power"],
                "on_periodic_beta": on_result["periodic_beta_power"],
                "off_spectrum": off_spec,
                "on_spectrum": on_spec,
            }
        )
        print(
            f"  exponent OFF/ON = {off_result['exponent']:.3f} / "
            f"{on_result['exponent']:.3f}"
        )
        del off, on, off_raw, on_raw, off_result, on_result
        gc.collect()

    cohort_off_spectra = np.stack([row["off_spectrum"] for row in cohort_rows])
    cohort_on_spectra = np.stack([row["on_spectrum"] for row in cohort_rows])
else:
    section("5. STRICT SAME-LOCATION FIGURE 2 ANALYSIS")
    print("Skipped. Set RUN_FULL_COHORT = True and rerun this cell when ready.")
    print(
        f"The one-pair path is complete; the retained cohort has "
        f"{len(pair_specs)} pairs."
    )


# %% Figure 2B: cohort spectra and fits
if RUN_FULL_COHORT:
    off_mean = cohort_off_spectra.mean(axis=0)
    on_mean = cohort_on_spectra.mean(axis=0)
    off_sem = stats.sem(cohort_off_spectra, axis=0)
    on_sem = stats.sem(cohort_on_spectra, axis=0)
    off_group_fit = fit_figure2_spectrum(off_mean, FREQUENCIES_HZ)
    on_group_fit = fit_figure2_spectrum(on_mean, FREQUENCIES_HZ)

    fig, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    axis.loglog(FREQUENCIES_HZ, off_mean, color="tab:purple", label="OFF levodopa")
    axis.fill_between(
        FREQUENCIES_HZ, off_mean - off_sem, off_mean + off_sem,
        color="tab:purple", alpha=0.2,
    )
    axis.loglog(FREQUENCIES_HZ, on_mean, color="0.45", label="ON levodopa")
    axis.fill_between(
        FREQUENCIES_HZ, on_mean - on_sem, on_mean + on_sem,
        color="0.5", alpha=0.2,
    )
    axis.loglog(
        off_group_fit["exponent_model"].freqs,
        10 ** off_group_fit["exponent_model"]._ap_fit,
        "--", color="tab:purple",
    )
    axis.loglog(
        on_group_fit["exponent_model"].freqs,
        10 ** on_group_fit["exponent_model"]._ap_fit,
        "--", color="black",
    )
    axis.set(xlabel="Frequency (Hz)", ylabel="Normalized power", xlim=(1, 90))
    axis.set_title(
        f"Figure 2B-style analysis: {len(pair_specs)} same-location hemispheres"
    )
    axis.legend()


# %% Figure 2C: paired exponent and periodic-beta comparisons
def paired_sign_flip_test(first, second, n_permutations=50_000, seed=82467):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    differences = first - second
    differences = differences[np.isfinite(differences)]
    observed_t = stats.ttest_1samp(differences, 0).statistic
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(n_permutations, differences.size))
    permuted = signs * differences
    permuted_t = permuted.mean(axis=1) / (
        permuted.std(axis=1, ddof=1) / np.sqrt(differences.size)
    )
    p_value = (np.sum(np.abs(permuted_t) >= abs(observed_t)) + 1) / (
        n_permutations + 1
    )
    return float(observed_t), float(p_value)


if RUN_FULL_COHORT:
    off_exponents = np.array([row["off_exponent"] for row in cohort_rows])
    on_exponents = np.array([row["on_exponent"] for row in cohort_rows])
    off_beta = np.array([row["off_periodic_beta"] for row in cohort_rows])
    on_beta = np.array([row["on_periodic_beta"] for row in cohort_rows])

    exponent_t, exponent_p = paired_sign_flip_test(off_exponents, on_exponents)
    beta_t, beta_p = paired_sign_flip_test(off_beta, on_beta)
    exponent_d = np.nanmean(on_exponents - off_exponents) / np.nanstd(
        on_exponents - off_exponents, ddof=1
    )

    section("6. FIGURE 2C PAIRED RESULTS")
    print(f"Exponent OFF mean: {off_exponents.mean():.4f}")
    print(f"Exponent ON mean:  {on_exponents.mean():.4f}")
    print(f"Exponent test:     t={exponent_t:.3f}, p={exponent_p:.5f}, paired d={exponent_d:.3f}")
    print(f"Periodic beta:     t={beta_t:.3f}, p={beta_p:.5f}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 5), constrained_layout=True)
    for axis, off_values, on_values, ylabel, p_value in (
        (axes[0], off_exponents, on_exponents, "Aperiodic exponent", exponent_p),
        (axes[1], off_beta, on_beta, "Periodic beta power", beta_p),
    ):
        finite = np.isfinite(off_values) & np.isfinite(on_values)
        axis.plot(
            np.tile([0, 1], (finite.sum(), 1)).T,
            np.column_stack([off_values[finite], on_values[finite]]).T,
            color="0.75", linewidth=0.7,
        )
        axis.scatter(np.zeros(finite.sum()), off_values[finite], color="tab:purple")
        axis.scatter(np.ones(finite.sum()), on_values[finite], color="0.45")
        axis.set(xticks=[0, 1], xticklabels=["OFF", "ON"], ylabel=ylabel)
        axis.set_title(f"paired permutation p={p_value:.4g}")


# %% Figure 2D-F: medication changes and clinical scores
UPDRS_OFF_MINUS_ON = np.array(
    [6, 5, 7, 9, 3, 3, np.nan, np.nan, 1, 3, 11, 3, 6, 7, 8,
     6, 4, 5, 14, 8, 6, 5, 8, 13, 7, 10.5, 18, 12.5, 6, 1],
    dtype=float,
)
updrs_same_location = UPDRS_OFF_MINUS_ON[
    pair_specs["author_pair_ix"].to_numpy()
]


def spearman_plot(axis, x, y, xlabel, ylabel):
    finite = np.isfinite(x) & np.isfinite(y)
    rho, p_value = stats.spearmanr(x[finite], y[finite])
    axis.scatter(x[finite], y[finite], facecolors="none", edgecolors="tab:red")
    if finite.sum() >= 2:
        slope, intercept = np.polyfit(x[finite], y[finite], 1)
        x_line = np.linspace(x[finite].min(), x[finite].max(), 100)
        axis.plot(x_line, slope * x_line + intercept, color="0.65")
    axis.set(xlabel=xlabel, ylabel=ylabel)
    axis.set_title(f"Spearman rho={rho:.3f}, p={p_value:.3f}, n={finite.sum()}")
    return rho, p_value


if RUN_FULL_COHORT:
    exponent_change = off_exponents - on_exponents
    beta_change = off_beta - on_beta

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    figure2d = spearman_plot(
        axes[0], beta_change, exponent_change,
        "Periodic beta OFF - ON", "Exponent OFF - ON",
    )
    figure2e = spearman_plot(
        axes[1], updrs_same_location, beta_change,
        "Contralateral UPDRS OFF - ON", "Periodic beta OFF - ON",
    )
    figure2f = spearman_plot(
        axes[2], updrs_same_location, exponent_change,
        "Contralateral UPDRS OFF - ON", "Exponent OFF - ON",
    )


# %% Variables ready for frequency-mode analysis
section("7. READY-TO-USE VARIABLES")

print(f"off_recording : selected OFF trace, {off_recording['trace'].shape}")
print(f"on_recording  : selected ON trace,  {on_recording['trace'].shape}")
print(f"off_spectrum  : normalized Morlet spectrum, {off_spectrum.shape}")
print(f"on_spectrum   : normalized Morlet spectrum, {on_spectrum.shape}")
print(f"off_fit       : exponent={off_fit['exponent']:.4f}, periodic beta={off_fit['periodic_beta_power']:.4f}")
print(f"on_fit        : exponent={on_fit['exponent']:.4f}, periodic beta={on_fit['periodic_beta_power']:.4f}")
print(
    f"cohort_rows   : {len(cohort_rows)} rows "
    f"({len(pair_specs)} after RUN_FULL_COHORT=True)"
)
print("\nNext: compare the paper's scalar exponent with frequency-mode coordinates")
print("using the same paired spectra, without changing the channel/window selection.")
