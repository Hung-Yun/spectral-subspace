"""Reproduce the locally supported analyses in Kragel et al. (2021).

Paper
-----
Kragel et al., "Rapid coordination of effective learning by the human
hippocampus," Science Advances 7:eabf7144.

Run this file one ``# %%`` cell at a time. The fast cells inventory the release,
read EyeLink events, and reproduce the patient revisitation measure in Figure
2A. The opt-in cells convert the very large ASCII iEEG files and reproduce the
central Figure 3C-E hippocampal-theta analysis with BOSC.

Scope
-----
* Figure 2A: local patient revisitation measure and an example gaze sequence.
* Figure 3B/C: example theta bouts and Pepisode spectra.
* Figure 3D/E: low/high theta prevalence aligned to revisitation onset.
* Figure 1 and Figure 2B need the full study-test stimulus/event join.
* Figure 2C includes three external healthy-control datasets.
* Figure 3A/F and Figure 4 require external atlas volumes; Figure 4 also needs
  the much heavier FieldTrip connectivity pipeline.

The wavelet, aperiodic fit, BOSC thresholds, duration criterion, theta ranges,
and fixation window follow the released MATLAB code. The one intentional
approximation is interictal-discharge detection: the proprietary detector in
the MATLAB workflow is replaced by a documented robust high-frequency
amplitude detector. Set ``IED_MODE = "none"`` to quantify its influence.
"""

# %% Imports and paths
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import gc
import re

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage, optimize, signal, stats

try:
    from eyelinkio import read_edf
except ImportError as error:
    raise ImportError(
        "Reading the native EyeLink EDF files requires eyelinkio==0.3.0. "
        "Run `uv sync`, then restart the notebook kernel."
    ) from error

from utils import fig_set, finish_plot


REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = REPO_ROOT / "data" / "kragel"
EEG_DIR = DATA_ROOT / "eeg"
EYE_DIR = DATA_ROOT / "eye"
BEHAV_DIR = DATA_ROOT / "behav"
LOCALIZATION_DIR = DATA_ROOT / "localization"
DERIVED_DIR = DATA_ROOT / "derived"

ARCHIVE_ROOT = REPO_ROOT / "archive" / "kragel"
AUTHOR_CODE_ROOT = ARCHIVE_ROOT / "code"
PAPER_PATH = ARCHIVE_ROOT / "paper.pdf"

SUBJECTS = tuple(f"S{number}" for number in range(1, 7))
EXPECTED_EEG_BLOCKS = {"S1": 8, "S2": 8, "S3": 3, "S4": 3, "S5": 4, "S6": 5}
HC_CONTACTS = {
    "S1": {"B1", "B2", "B3", "C1", "D1", "AL'1"},
    "S2": {"B1", "B2", "C1", "C2", "D1", "D2"},
    "S3": {"B1", "B2", "C1", "C2", "D1", "D2"},
    "S4": {"C1", "C2", "D1", "D2"},
    "S5": {"B1", "B2", "D1", "D2"},
    "S6": {"B1", "B2", "B3", "C1", "C2"},
}

FREQUENCIES = np.logspace(np.log10(1), np.log10(40), 50)
WAVELET_CYCLES = 6
BOSC_POWER_QUANTILE = 0.95
BOSC_MIN_CYCLES = 3
FIXATION_HALF_WINDOW_S = 0.750
REVISIT_DISTANCE_PX = 120.0
SCENE_DELAY_S = 0.800
SCENE_DURATION_S = 3.000
IED_MODE = "robust_hf"  # "robust_hf" or "none"

fig_set(font_size=8, linewidth=0.8)


def section(title):
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92)


# %% Verify the local release without loading the recordings
def inventory():
    rows = []
    for subject in SUBJECTS:
        eeg = sorted((EEG_DIR / subject).glob("*.m00"))
        eye = sorted((EYE_DIR / subject).glob("*.edf"))
        behav = sorted((BEHAV_DIR / subject).glob("*.txt"))
        loc = sorted((LOCALIZATION_DIR / subject).glob("*.csv"))
        rows.append(
            {
                "subject": subject,
                "eeg_blocks": len(eeg),
                "expected_eeg": EXPECTED_EEG_BLOCKS[subject],
                "eye_edf": len(eye),
                "behavior_txt": len(behav),
                "localization_csv": len(loc),
                "eeg_gib": sum(path.stat().st_size for path in eeg) / 2**30,
            }
        )
    table = pd.DataFrame(rows)
    table.loc[table["subject"].eq("S6"), "note"] = (
        "block 4 present but excluded from paper neural analysis"
    )
    return table


inventory_table = inventory()
section("LOCAL KRAGEL RELEASE")
print(inventory_table.to_string(index=False))
print(f"\nRaw EEG total: {inventory_table['eeg_gib'].sum():.2f} GiB")
print("Expected totals: 31 EEG blocks, 64 EDFs, 62 behavior files, 6 CSVs")


# %% Native .m00 header and bipolar-channel handling
@dataclass(frozen=True)
class M00Header:
    n_samples: int
    n_channels: int
    fs: float
    raw_labels: tuple[str, ...]
    labels: tuple[str, ...]


def parse_m00_header(path):
    """Read only the two header lines of a Nihon Kohden ASCII export."""
    path = Path(path)
    with path.open("rb") as stream:
        first = stream.readline().decode("latin-1").strip()
        second = stream.readline().decode("latin-1").strip()

    def header_number(name, cast=float):
        match = re.search(rf"{re.escape(name)}=([^ ]+)", first)
        if match is None:
            raise ValueError(f"Missing {name!r} in {path}")
        return cast(match.group(1))

    n_samples = header_number("TimePoints", int)
    n_channels = header_number("Channels", int)
    fs = 1000.0 / header_number("SamplingInterval[ms]")

    # S5 writes REF 1 instead of REF1. The isolated token is not a channel.
    raw_labels = []
    for token in second.split():
        if token == "1" and raw_labels and raw_labels[-1].upper().endswith("-REF"):
            raw_labels[-1] += token
        else:
            raw_labels.append(token)
    if len(raw_labels) != n_channels:
        raise ValueError(
            f"{path.name}: header says {n_channels} channels but parsed "
            f"{len(raw_labels)} labels"
        )

    labels = tuple(label.replace("*", "").replace(" ", "") for label in raw_labels)
    return M00Header(n_samples, n_channels, fs, tuple(raw_labels), labels)


def contact_name(label):
    """Return the contact portion of CONTACT-REFERENCE labels."""
    clean = label.replace("*", "").replace(" ", "")
    if clean.upper().startswith("DC"):
        return None
    return clean.split("-", maxsplit=1)[0]


def adjacent_bipolar_pairs(header):
    """Return adjacent-contact pairs in the same manner as ecog_reref.m."""
    locations = {}
    parsed = {}
    for index, label in enumerate(header.labels):
        contact = contact_name(label)
        if contact is None:
            continue
        match = re.fullmatch(r"(.+?)(\d+)", contact)
        if match is None:
            continue
        group, number = match.group(1), int(match.group(2))
        locations.setdefault((group, number), index)
        parsed[index] = (group, number, contact)

    pairs = []
    for left_index, (group, number, left_contact) in parsed.items():
        right_index = locations.get((group, number + 1))
        if right_index is None:
            continue
        right_contact = contact_name(header.labels[right_index])
        pairs.append((f"{left_contact}-{right_contact}", left_index, right_index))
    return pairs


def hippocampal_pairs(subject, header):
    contacts = HC_CONTACTS[subject]
    return [
        pair
        for pair in adjacent_bipolar_pairs(header)
        if contact_name(pair[0]) in contacts
        or pair[0].split("-")[-1] in contacts
    ]


def block_number(path):
    match = re.search(r"block(\d+)", Path(path).stem)
    if match is None:
        raise ValueError(f"No block number in {path}")
    return int(match.group(1))


def raw_block_path(subject, block):
    return EEG_DIR / subject / f"{subject}_sl_block{block}.m00"


def hc_cache_path(subject, block):
    return DERIVED_DIR / "hc_bipolar" / subject / f"{subject}_block{block}_hc.h5"


def bosc_cache_path(subject, block):
    return DERIVED_DIR / "bosc" / subject / f"{subject}_block{block}_bosc.h5"


def convert_hc_block(subject, block, chunk_rows=50_000, overwrite=False):
    """Stream one raw block into a compact HC-bipolar HDF5 cache.

    Only the raw contacts needed for author-defined hippocampal bipolar pairs
    and the synchronization channel are parsed. This avoids materializing a
    1-2 GiB all-channel array in memory.
    """
    source = raw_block_path(subject, block)
    target = hc_cache_path(subject, block)
    if target.exists() and not overwrite:
        return target
    if subject == "S6" and block == 4:
        raise ValueError("S6 block 4 was excluded from the neural analysis.")

    header = parse_m00_header(source)
    pairs = hippocampal_pairs(subject, header)
    if not pairs:
        raise ValueError(f"No hippocampal pairs identified in {source.name}")
    sync_candidates = [
        index for index, label in enumerate(header.labels)
        if label.upper().startswith("DC")
    ]
    if len(sync_candidates) != 1:
        raise ValueError(f"Expected one sync channel, found {sync_candidates}")
    sync_index = sync_candidates[0]

    usecols = sorted({index for _, left, right in pairs for index in (left, right)} | {sync_index})
    local_index = {raw_index: index for index, raw_index in enumerate(usecols)}
    target.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(target, "w") as out:
        data = out.create_dataset(
            "eeg",
            shape=(len(pairs) + 1, header.n_samples),
            dtype="f4",
            chunks=(1, min(header.n_samples, 65_536)),
            compression="gzip",
            compression_opts=4,
        )
        string_type = h5py.string_dtype("utf-8")
        out.create_dataset(
            "pairs", data=np.asarray([pair[0] for pair in pairs] + ["SYNC"], dtype=object),
            dtype=string_type,
        )
        out.attrs["fs"] = header.fs
        out.attrs["source"] = str(source.relative_to(REPO_ROOT))
        out.attrs["paper_excluded"] = False

        cursor = 0
        reader = pd.read_csv(
            source,
            sep=r"\s+",
            skiprows=2,
            header=None,
            usecols=usecols,
            dtype=np.float32,
            chunksize=chunk_rows,
        )
        for frame in reader:
            values = frame.to_numpy(dtype=np.float32, copy=False)
            stop = cursor + len(frame)
            for pair_index, (_, left, right) in enumerate(pairs):
                data[pair_index, cursor:stop] = (
                    values[:, local_index[left]] - values[:, local_index[right]]
                )
            data[-1, cursor:stop] = values[:, local_index[sync_index]]
            cursor = stop
        if cursor != header.n_samples:
            raise ValueError(
                f"Converted {cursor} samples from {source.name}; expected {header.n_samples}"
            )
    return target


# Set true only when ready to create about one compact HDF5 per raw block.
RUN_ASCII_CONVERSION = False
if RUN_ASCII_CONVERSION:
    for subject in SUBJECTS:
        for source in sorted((EEG_DIR / subject).glob("*.m00")):
            block = block_number(source)
            if subject == "S6" and block == 4:
                continue
            print(convert_hc_block(subject, block))


# %% EyeLink encoding fixations and the Figure 2A revisitation definition
def _message_text(value):
    if isinstance(value, bytes):
        return value.decode("latin-1", errors="replace").strip().lower()
    return str(value).strip().lower()


def classify_revisits(fixations, threshold_px=REVISIT_DISTANCE_PX):
    """Add the labels produced by spotlight_enc_cvp.m's nearest-point rule."""
    result = fixations.copy()
    result["location_id"] = -1
    result["is_revisit"] = False
    result["initial_of_revisited"] = False
    next_location = 0

    for (_, _), indices in result.groupby(["block", "trial"], sort=False).groups.items():
        trial_indices = list(indices)
        points = result.loc[trial_indices, ["x", "y"]].to_numpy(float)
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        np.fill_diagonal(distances, np.nan)
        locations = np.full(len(points), -1, dtype=int)
        novel = np.ones(len(points), dtype=bool)
        for fixation_number in range(len(points)):
            within = distances[fixation_number] < threshold_px
            if np.any(within):
                nearest_distance = np.nanmin(np.where(within, distances[fixation_number], np.nan))
                nearest = int(np.flatnonzero(within & np.isclose(distances[fixation_number], nearest_distance))[0])
                if fixation_number > nearest:
                    locations[fixation_number] = locations[nearest]
                    novel[fixation_number] = False
                    continue
            locations[fixation_number] = next_location
            next_location += 1

        revisited_locations = set(locations[~novel])
        result.loc[trial_indices, "location_id"] = locations
        result.loc[trial_indices, "is_revisit"] = ~novel
        result.loc[trial_indices, "initial_of_revisited"] = [
            is_novel and location in revisited_locations
            for is_novel, location in zip(novel, locations)
        ]
    return result


def read_encoding_fixations(subject, block):
    """Read scene-period fixations and align their times to each trial message."""
    path = EYE_DIR / subject / f"{subject}_s{block}.edf"
    edf = read_edf(path)
    messages = edf["discrete"]["messages"]
    starts = np.asarray(
        [float(row["stime"]) for row in messages if _message_text(row["msg"]) == "encode"]
    )
    # S4_s1 is the one malformed EDF in the release: its first trial message
    # is absent, although the trial recording and later 23 messages are intact.
    # The file time origin is the first recording onset, as it is in every
    # other study EDF, so restore that onset at zero.
    if len(starts) == 23 and starts[0] > 2:
        starts = np.r_[0.0, starts]
    if len(starts) != 24:
        raise ValueError(f"{path.name}: found {len(starts)} encode messages, expected 24")

    rows = []
    for fixation in edf["discrete"]["fixations"]:
        start = float(fixation["stime"])
        stop = float(fixation["etime"])
        trial = int(np.searchsorted(starts, start, side="right") - 1)
        if trial < 0:
            continue
        scene_start = starts[trial] + SCENE_DELAY_S
        scene_stop = scene_start + SCENE_DURATION_S
        # The MATLAB reader retained events whose start OR end was in the scene.
        if stop < scene_start or start >= scene_stop:
            continue
        rows.append(
            {
                "subject": subject,
                "block": block,
                "trial": trial + 1,
                "onset_s": start - starts[trial],
                "duration_s": stop - start,
                "x": float(fixation["axp"]),
                "y": float(fixation["ayp"]),
                "precue": start - starts[trial] < SCENE_DELAY_S,
            }
        )
    return classify_revisits(pd.DataFrame(rows))


def subject_encoding_fixations(subject):
    paths = sorted((EYE_DIR / subject).glob(f"{subject}_s*.edf"))
    return pd.concat(
        [read_encoding_fixations(subject, int(re.search(r"_s(\d+)", path.stem).group(1))) for path in paths],
        ignore_index=True,
    )


def figure2a_patient_revisitation():
    """Reproduce the local-patient revisited-location proportion in Figure 2A."""
    subject_rows = []
    all_fixations = []
    for subject in SUBJECTS:
        fixations = subject_encoding_fixations(subject)
        all_fixations.append(fixations)
        trial_rows = []
        for _, trial in fixations.groupby(["block", "trial"]):
            counts = trial.groupby("location_id").size()
            trial_rows.append(np.mean(counts > 1))
        subject_rows.append(
            {
                "subject": subject,
                "revisited_location_fraction": np.nanmean(trial_rows),
                "revisit_fixation_fraction": fixations["is_revisit"].mean(),
                "n_fixations": len(fixations),
            }
        )
    summary = pd.DataFrame(subject_rows)
    fixations = pd.concat(all_fixations, ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    example = fixations.query("subject == 'S1' and block == 1 and trial == 1")
    axes[0].plot(example["x"], example["y"], color="0.35", linewidth=1)
    axes[0].scatter(example["x"], example["y"], c="0.7", edgecolor="k", s=30, zorder=2)
    revisit = example["is_revisit"]
    axes[0].scatter(example.loc[revisit, "x"], example.loc[revisit, "y"], c="#55a868", s=38, zorder=3)
    for number, (_, row) in enumerate(example.iterrows(), start=1):
        axes[0].text(row["x"], row["y"], str(number), ha="center", va="center", fontsize=6)
    axes[0].invert_yaxis()
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_title("Example study fixation sequence")
    axes[0].set_xlabel("Screen x (pixels)")
    axes[0].set_ylabel("Screen y (pixels)")

    values = summary["revisited_location_fraction"].to_numpy()
    axes[1].scatter(np.ones_like(values), values, color="0.55", zorder=2)
    axes[1].errorbar(
        1,
        values.mean(),
        yerr=stats.sem(values),
        color="k",
        marker="o",
        capsize=3,
        zorder=3,
    )
    axes[1].set(xlim=(0.7, 1.3), xticks=[1], xticklabels=["Patients"], ylabel="Revisited locations / locations")
    axes[1].set_title("Figure 2A measure")
    fig.tight_layout()
    return summary, fixations, fig


# This is fast enough to run interactively, but it is opt-in to keep imports quiet.
RUN_FIGURE_2A = False
if RUN_FIGURE_2A:
    figure2_summary, encoding_fixations, figure2a = figure2a_patient_revisitation()
    print(figure2_summary.to_string(index=False))


# %% Sync pulses, artifact approximation, and paper-matched BOSC
def trial_onsets(sync_trace, fs):
    """Recover the 24 study and 48 test pre-cue onsets from the sync trace."""
    peaks, _ = signal.find_peaks(sync_trace, prominence=2)
    # S5's DC trace encodes each pulse with a stepped 4-6-2 shape. MATLAB's
    # findpeaks counts the first two maxima, whereas SciPy assigns only the
    # central maximum a prominence >=2. Recover MATLAB's 102 peaks explicitly.
    if len(peaks) == 51:
        stepped_peaks, _ = signal.find_peaks(sync_trace, prominence=1)
        stepped_peaks = stepped_peaks[np.asarray(sync_trace)[stepped_peaks] >= 4]
        if len(stepped_peaks) == 102:
            peaks = stepped_peaks
    if len(peaks) < 101:
        raise ValueError(f"Found {len(peaks)} sync peaks; expected at least 101")
    onsets = peaks - 1  # released get_onsets.m takes one sample before each peak
    encoding = onsets[2:49:2][:24]
    recognition = onsets[53:101][:48]
    if len(encoding) != 24:
        raise ValueError(f"Found {len(encoding)} encoding sync onsets, expected 24")
    return encoding, recognition


def remove_line_noise(trace, fs):
    cleaned = np.asarray(trace, dtype=float).copy()
    for frequency in (60.0, 120.0, 180.0):
        if frequency >= fs / 2:
            continue
        b, a = signal.iirnotch(frequency, Q=30.0, fs=fs)
        cleaned = signal.filtfilt(b, a, cleaned)
    return cleaned


def robust_hf_ied_mask(trace, fs, z_threshold=8.0):
    """Approximate the MATLAB IED detector and expand detections by +/-1 s."""
    upper = min(150.0, fs / 2 - 1.0)
    sos = signal.butter(4, (25.0, upper), btype="bandpass", fs=fs, output="sos")
    high_frequency = signal.sosfiltfilt(sos, trace)
    envelope = np.abs(signal.hilbert(high_frequency))
    median = np.median(envelope)
    scale = 1.4826 * np.median(np.abs(envelope - median))
    if scale == 0:
        return np.zeros(trace.size, dtype=bool)
    candidates = envelope > median + z_threshold * scale
    return ndimage.maximum_filter1d(candidates.astype(np.uint8), size=int(2 * fs + 1)) > 0


def morlet_power(trace, frequency, fs, cycles=WAVELET_CYCLES):
    """Match BOSC_tf.m's Morlet definition and same-length convolution."""
    st = 1.0 / (2 * np.pi * (frequency / cycles))
    amplitude = 1.0 / np.sqrt(st * np.sqrt(np.pi))
    time = np.arange(-3.6 * st, 3.6 * st + 0.5 / fs, 1.0 / fs)
    wavelet = amplitude * np.exp(-(time**2) / (2 * st**2)) * np.exp(2j * np.pi * frequency * time)
    transformed = signal.fftconvolve(trace, wavelet, mode="same")
    return np.abs(transformed) ** 2


def robust_aperiodic_fit(frequencies, log10_power):
    """Python equivalent of the no-knee robust_ap_fit.m path."""
    x = np.log10(np.asarray(frequencies, dtype=float))
    y = np.asarray(log10_power, dtype=float)

    def model(params):
        return params[0] - params[1] * x

    initial = np.asarray([y[0], max(0.0, y[0])])
    first = optimize.least_squares(lambda params: model(params) - y, initial).x
    flattened = y - model(first)
    flattened[flattened < 0] = 0
    threshold = np.percentile(flattened, 2.5)
    keep = flattened <= threshold
    if keep.sum() < 3:
        keep = np.ones_like(keep, dtype=bool)
    final = optimize.least_squares(
        lambda params: (params[0] - params[1] * x[keep]) - y[keep], first
    ).x
    return final, final[0] - final[1] * x


def duration_threshold(power, threshold, minimum_samples):
    above = np.asarray(power) > threshold
    changes = np.diff(np.r_[False, above, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    detected = np.zeros_like(above)
    for start, stop in zip(starts, stops):
        if stop - start >= minimum_samples:
            detected[start:stop] = True
    return detected


def bosc_channel(trace, fs, frequencies=FREQUENCIES, ied_mode=IED_MODE):
    """Return Pepisode, episode masks, and aperiodic background for one pair."""
    trace = remove_line_noise(trace, fs)
    if ied_mode == "robust_hf":
        ied_mask = robust_hf_ied_mask(trace, fs)
    elif ied_mode == "none":
        ied_mask = np.zeros(trace.size, dtype=bool)
    else:
        raise ValueError("ied_mode must be 'robust_hf' or 'none'")

    edge = int(np.ceil(WAVELET_CYCLES * fs / np.min(frequencies)))
    valid = ~ied_mask
    valid[:edge] = False
    valid[-edge:] = False
    powers = np.empty((len(frequencies), trace.size), dtype=np.float32)
    mean_power = np.empty(len(frequencies))
    for index, frequency in enumerate(frequencies):
        powers[index] = morlet_power(trace, frequency, fs).astype(np.float32)
        mean_power[index] = np.mean(powers[index, valid])

    ap_params, ap_log10 = robust_aperiodic_fit(frequencies, np.log10(mean_power))
    background = 10**ap_log10
    power_thresholds = stats.chi2.ppf(BOSC_POWER_QUANTILE, 2) * background / 2
    episodes = np.empty_like(powers, dtype=bool)
    for index, frequency in enumerate(frequencies):
        minimum_samples = int(np.ceil(BOSC_MIN_CYCLES * fs / frequency))
        episodes[index] = duration_threshold(powers[index], power_thresholds[index], minimum_samples)
    episodes[:, ied_mask] = False
    p_episode = episodes[:, valid].mean(axis=1)
    return {
        "p_episode": p_episode,
        "episodes": episodes,
        "background": background,
        "power_thresholds": power_thresholds,
        "aperiodic_params": ap_params,
        "ied_mask": ied_mask,
    }


def run_bosc_block(subject, block, overwrite=False, ied_mode=IED_MODE):
    """Run BOSC on every cached HC pair in one block and save the result."""
    source = hc_cache_path(subject, block)
    target = bosc_cache_path(subject, block)
    if target.exists() and not overwrite:
        return target
    if not source.exists():
        raise FileNotFoundError(f"Create the compact cache first: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(source, "r") as inp, h5py.File(target, "w") as out:
        fs = float(inp.attrs["fs"])
        labels = [value.decode() if isinstance(value, bytes) else str(value) for value in inp["pairs"][:-1]]
        n_samples = inp["eeg"].shape[1]
        out.attrs["fs"] = fs
        out.attrs["ied_mode"] = ied_mode
        out.create_dataset("frequencies", data=FREQUENCIES)
        out.create_dataset("pairs", data=np.asarray(labels, dtype=object), dtype=h5py.string_dtype("utf-8"))
        out.create_dataset("sync", data=inp["eeg"][-1], compression="gzip")
        p_episode = out.create_dataset("p_episode", shape=(len(labels), len(FREQUENCIES)), dtype="f4")
        episodes = out.create_dataset(
            "episodes",
            shape=(len(labels), len(FREQUENCIES), n_samples),
            dtype="?",
            chunks=(1, 1, min(n_samples, 65_536)),
            compression="gzip",
            compression_opts=4,
        )
        for pair_index, label in enumerate(labels):
            print(f"{subject} block {block}: BOSC {label}")
            result = bosc_channel(np.asarray(inp["eeg"][pair_index], dtype=float), fs, ied_mode=ied_mode)
            p_episode[pair_index] = result["p_episode"]
            episodes[pair_index] = result["episodes"]
            group = out.create_group(f"pair_{pair_index}")
            group.create_dataset("background", data=result["background"])
            group.create_dataset("power_thresholds", data=result["power_thresholds"])
            group.create_dataset("aperiodic_params", data=result["aperiodic_params"])
            del result
            gc.collect()
    return target


RUN_BOSC = False
if RUN_BOSC:
    for subject in SUBJECTS:
        for source in sorted((EEG_DIR / subject).glob("*.m00")):
            block = block_number(source)
            if subject == "S6" and block == 4:
                continue
            print(run_bosc_block(subject, block))


# %% Figure 3C: subject/electrode theta peaks
def theta_peak_table(subject):
    paths = sorted((DERIVED_DIR / "bosc" / subject).glob("*_bosc.h5"))
    if not paths:
        raise FileNotFoundError(f"No BOSC caches for {subject}")
    by_label = {}
    for path in paths:
        with h5py.File(path, "r") as h5:
            labels = [value.decode() if isinstance(value, bytes) else str(value) for value in h5["pairs"]]
            for label, spectrum in zip(labels, h5["p_episode"]):
                by_label.setdefault(label, []).append(np.asarray(spectrum))

    rows = []
    for label, spectra in by_label.items():
        spectrum = np.mean(spectra, axis=0)
        peak_indices, properties = signal.find_peaks(spectrum, prominence=0.02)
        for index, prominence in zip(peak_indices, properties["prominences"]):
            frequency = FREQUENCIES[index]
            if frequency < 4:
                band = "low"
            elif 4 < frequency < 10:
                band = "high"
            else:
                continue
            rows.append(
                {
                    "subject": subject,
                    "pair": label,
                    "band": band,
                    "frequency": frequency,
                    "p_episode": spectrum[index],
                    "prominence": prominence,
                }
            )
    peaks = pd.DataFrame(rows)
    if peaks.empty:
        return peaks
    # The released epoch code retains the largest peak in each band/pair.
    return (
        peaks.sort_values("p_episode", ascending=False)
        .drop_duplicates(["subject", "pair", "band"])
        .sort_values(["pair", "band"])
        .reset_index(drop=True)
    )


def plot_figure3c(subject="S1"):
    paths = sorted((DERIVED_DIR / "bosc" / subject).glob("*_bosc.h5"))
    spectra = []
    for path in paths:
        with h5py.File(path, "r") as h5:
            spectra.extend(np.asarray(h5["p_episode"]))
    spectra = np.asarray(spectra)
    fig, ax = plt.subplots(figsize=(3.1, 2.5))
    ax.plot(FREQUENCIES, np.mean(spectra, axis=0), color="#c44e52")
    ax.fill_between(
        FREQUENCIES,
        np.mean(spectra, axis=0) - stats.sem(spectra, axis=0),
        np.mean(spectra, axis=0) + stats.sem(spectra, axis=0),
        color="#c44e52",
        alpha=0.22,
    )
    ax.set(xlabel="Frequency (Hz)", ylabel="Pepisode", xlim=(1, 40), title=f"Figure 3C - {subject} HC")
    fig.tight_layout()
    return fig


# %% Figure 3B: automatically select example low/high-theta episodes
def plot_figure3b(subject="S1", block=1, pair_index=1):
    raw_path = hc_cache_path(subject, block)
    result_path = bosc_cache_path(subject, block)
    peaks = theta_peak_table(subject)
    with h5py.File(raw_path, "r") as raw, h5py.File(result_path, "r") as result:
        fs = float(raw.attrs["fs"])
        trace = np.asarray(raw["eeg"][pair_index], dtype=float)
        label_value = result["pairs"][pair_index]
        label = label_value.decode() if isinstance(label_value, bytes) else str(label_value)
        pair_peaks = peaks.loc[peaks["pair"].eq(label)].set_index("band")
        fig, axes = plt.subplots(2, 1, figsize=(4.5, 2.8), sharex=False)
        colors = {"low": "#0066cc", "high": "#cc6633"}
        for ax, band in zip(axes, ("low", "high")):
            frequency = float(pair_peaks.loc[band, "frequency"])
            frequency_index = int(np.argmin(np.abs(FREQUENCIES - frequency)))
            detected = np.asarray(result["episodes"][pair_index, frequency_index])
            changes = np.diff(np.r_[False, detected, False].astype(np.int8))
            starts = np.flatnonzero(changes == 1)
            stops = np.flatnonzero(changes == -1)
            if len(starts) == 0:
                ax.text(0.5, 0.5, f"No {band}-theta bout", transform=ax.transAxes, ha="center")
                continue
            # Choose a long, non-edge bout deterministically.
            choice = int(np.argmax(stops - starts))
            start, stop = starts[choice], stops[choice]
            pad = int(round(0.25 * fs))
            window = np.arange(max(0, start - pad), min(trace.size, stop + pad))
            time = (window - start) / fs
            ax.plot(time, trace[window], color="0.15", linewidth=0.7)
            bout = np.arange(start, stop)
            ax.plot((bout - start) / fs, trace[bout], color=colors[band], linewidth=1.2)
            ax.set_ylabel("uV")
            ax.set_title(f"{label}: {frequency:.1f}-Hz {band} theta")
        axes[-1].set_xlabel("Time from detected bout onset (s)")
        fig.tight_layout()
    return fig


# %% Figure 3D/E: fixation-aligned theta prevalence
def extract_subject_fixation_epochs(subject, overwrite=False):
    """Save fixation-aligned low/high episode masks for one participant."""
    target = DERIVED_DIR / "figure3_epochs" / f"{subject}_fixation_epochs.npz"
    if target.exists() and not overwrite:
        return target
    peaks = theta_peak_table(subject)
    if peaks.empty:
        raise ValueError(f"No theta peaks detected for {subject}")

    epoch_rows = []
    metadata_rows = []
    for result_path in sorted((DERIVED_DIR / "bosc" / subject).glob("*_bosc.h5")):
        block = block_number(result_path)
        fixations = read_encoding_fixations(subject, block)
        with h5py.File(result_path, "r") as h5:
            fs = float(h5.attrs["fs"])
            labels = [value.decode() if isinstance(value, bytes) else str(value) for value in h5["pairs"]]
            encoding_onsets, _ = trial_onsets(np.asarray(h5["sync"]), fs)
            half = int(round(FIXATION_HALF_WINDOW_S * fs))
            target_n = 1501  # authors resampled every participant to 1 kHz
            for fixation_index, fixation in fixations.iterrows():
                # Figure 3 contrasts revisitations with locations never revisited;
                # the initial fixation at a location revisited later is excluded.
                if fixation["is_revisit"]:
                    condition = "revisit"
                elif fixation["initial_of_revisited"]:
                    continue
                else:
                    condition = "other"
                center = int(encoding_onsets[int(fixation["trial"]) - 1] + round(fixation["onset_s"] * fs))
                if center - half < 0 or center + half >= h5["episodes"].shape[-1]:
                    continue
                for pair_index, label in enumerate(labels):
                    pair_peaks = peaks.loc[peaks["pair"].eq(label)]
                    for band in ("low", "high"):
                        selected = pair_peaks.loc[pair_peaks["band"].eq(band)]
                        if selected.empty:
                            continue
                        frequency = float(selected.iloc[0]["frequency"])
                        frequency_index = int(np.argmin(np.abs(FREQUENCIES - frequency)))
                        episode = np.asarray(
                            h5["episodes"][pair_index, frequency_index, center - half:center + half + 1],
                            dtype=float,
                        )
                        if episode.size != target_n:
                            episode = signal.resample(episode, target_n)
                            episode = np.clip(episode, 0, 1)
                        epoch_rows.append(episode.astype(np.float32))
                        metadata_rows.append(
                            (block, int(fixation["trial"]), fixation_index, label, band, condition, frequency)
                        )
    if not epoch_rows:
        raise ValueError(f"No fixation epochs created for {subject}")
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = np.asarray(
        metadata_rows,
        dtype=[
            ("block", "i2"), ("trial", "i2"), ("fixation", "i4"),
            ("pair", "U32"), ("band", "U4"), ("condition", "U8"), ("frequency", "f4"),
        ],
    )
    np.savez_compressed(target, epochs=np.asarray(epoch_rows), metadata=metadata)
    return target


def wilcoxon_z_timecourse(epochs):
    """Match run_encoding_stats.m: each time point versus each trial's window mean."""
    epochs = np.asarray(epochs, dtype=float)
    reference = np.nanmean(epochs, axis=1)
    z = np.full(epochs.shape[1], np.nan)
    for time_index in range(epochs.shape[1]):
        valid = np.isfinite(epochs[:, time_index]) & np.isfinite(reference)
        if valid.sum() < 3 or np.allclose(epochs[valid, time_index], reference[valid]):
            continue
        result = stats.wilcoxon(
            epochs[valid, time_index], reference[valid], method="approx", correction=False
        )
        z[time_index] = getattr(result, "zstatistic", np.nan)
    return z


def subject_figure3_traces(subject):
    loaded = np.load(DERIVED_DIR / "figure3_epochs" / f"{subject}_fixation_epochs.npz")
    epochs, metadata = loaded["epochs"], loaded["metadata"]
    traces = {}
    for band, condition in product(("low", "high"), ("revisit", "other")):
        pair_traces = []
        for pair in np.unique(metadata["pair"]):
            selected = (metadata["band"] == band) & (metadata["condition"] == condition) & (metadata["pair"] == pair)
            if selected.sum() >= 3:
                pair_traces.append(wilcoxon_z_timecourse(epochs[selected]))
        traces[(band, condition)] = np.nanmean(pair_traces, axis=0)
    return traces


def exact_sign_flip_p(differences):
    """Two-sided, time-resolved exact paired t permutation p values."""
    differences = np.asarray(differences, dtype=float)
    n_subjects, n_times = differences.shape
    signs = np.asarray(list(product((-1.0, 1.0), repeat=n_subjects)))

    def t_values(values):
        mean = np.nanmean(values, axis=0)
        standard_error = np.nanstd(values, axis=0, ddof=1) / np.sqrt(n_subjects)
        return np.divide(mean, standard_error, out=np.zeros(n_times), where=standard_error > 0)

    observed = np.abs(t_values(differences))
    null = np.empty((len(signs), n_times))
    for index, sign_vector in enumerate(signs):
        null[index] = np.abs(t_values(differences * sign_vector[:, None]))
    return (np.sum(null >= observed, axis=0) + 1) / (len(signs) + 1)


def fdr_bh(p_values, alpha=0.05):
    p_values = np.asarray(p_values, dtype=float)
    valid_indices = np.flatnonzero(np.isfinite(p_values))
    ordered = valid_indices[np.argsort(p_values[valid_indices])]
    thresholds = alpha * np.arange(1, len(ordered) + 1) / len(ordered)
    passed = p_values[ordered] <= thresholds
    rejected = np.zeros(p_values.shape, dtype=bool)
    if np.any(passed):
        cutoff = p_values[ordered[np.flatnonzero(passed)[-1]]]
        rejected[valid_indices] = p_values[valid_indices] <= cutoff
    return rejected


def plot_figure3de():
    subject_traces = {subject: subject_figure3_traces(subject) for subject in SUBJECTS}
    time_ms = np.arange(-750, 751)

    low_revisit = np.asarray([subject_traces[s][("low", "revisit")] for s in SUBJECTS])
    low_other = np.asarray([subject_traces[s][("low", "other")] for s in SUBJECTS])
    high_revisit = np.asarray([subject_traces[s][("high", "revisit")] for s in SUBJECTS])
    high_other = np.asarray([subject_traces[s][("high", "other")] for s in SUBJECTS])
    revisit = np.nanmean(np.stack([low_revisit, high_revisit]), axis=0)
    other = np.nanmean(np.stack([low_other, high_other]), axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharex=True)
    colors = {"revisit": "#55a868", "other": "0.35", "low": "#0066cc", "high": "#cc6633"}

    for values, label in ((revisit, "Revisitation"), (other, "Other")):
        color = colors[label.lower() if label == "Other" else "revisit"]
        mean = np.nanmean(values, axis=0)
        sem = stats.sem(values, axis=0, nan_policy="omit")
        axes[0].plot(time_ms, mean, color=color, label=label)
        axes[0].fill_between(time_ms, mean - sem, mean + sem, color=color, alpha=0.2)
    significant = fdr_bh(exact_sign_flip_p(revisit - other))
    axes[0].scatter(time_ms[significant], np.full(significant.sum(), -1.35), s=2, color="#c44e52")
    axes[0].set(title="Figure 3D", ylabel="Theta prevalence (Z)", ylim=(-1.5, 1.5))
    axes[0].legend(frameon=False)

    for revisit_band, other_band, label in (
        (low_revisit, low_other, "low"), (high_revisit, high_other, "high")
    ):
        delta = revisit_band - other_band
        mean = np.nanmean(delta, axis=0)
        sem = stats.sem(delta, axis=0, nan_policy="omit")
        axes[1].plot(time_ms, mean, color=colors[label], label=label.capitalize() + " theta")
        axes[1].fill_between(time_ms, mean - sem, mean + sem, color=colors[label], alpha=0.2)
    axes[1].set(title="Figure 3E", ylabel="Revisitation - other (Z)", ylim=(-2, 2))
    axes[1].legend(frameon=False)

    for ax in axes:
        ax.axvline(0, color="k", linestyle="--", linewidth=0.8)
        ax.set(xlabel="Time to fixation onset (ms)", xlim=(-750, 750))
    fig.tight_layout()
    return subject_traces, fig


RUN_FIXATION_EPOCHS = False
if RUN_FIXATION_EPOCHS:
    for subject in SUBJECTS:
        print(extract_subject_fixation_epochs(subject))
    figure3_subject_traces, figure3de = plot_figure3de()


# %% Recommended execution order
section("HOW TO RUN THE EXPENSIVE FIGURE 3 REPRODUCTION")
print(
    "1. Set RUN_ASCII_CONVERSION=True and run that cell once.\n"
    "2. Set RUN_BOSC=True and run the BOSC cell (the slowest step).\n"
    "3. Inspect plot_figure3c() and plot_figure3b().\n"
    "4. Set RUN_FIXATION_EPOCHS=True to create Figure 3D/E.\n"
    "5. Re-run with IED_MODE='none' as a sensitivity analysis.\n\n"
    "Caches are written only under data/kragel/derived; raw files are never modified."
)
