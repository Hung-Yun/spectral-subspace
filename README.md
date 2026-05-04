# spectral-subspace

Starter workspace for extracting local field potential (LFP) data from Blackrock
`.ns5` files.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The current active Python did not have `spikeinterface` installed when this
project was scaffolded.

## Inspect an NS5 File

```bash
python scripts/inspect_ns5.py data/EMU-0090_subj-YEY_task-Pacman_time-20240402_124118_NSP-2.ns5 --json outputs/ns5_summary.json
```

This prints the SpikeInterface recording summary, sampling rate, duration, and
the first channel names. The JSON file is useful for confirming which channels
are electrodes and which are auxiliary signals such as audio or photodiode.

## Extract LFP

```bash
python scripts/extract_lfp.py data/EMU-0090_subj-YEY_task-Pacman_time-20240402_124118_NSP-2.ns5 --overwrite
```

By default, the LFP pipeline:

- reads the `.ns5` file with `spikeinterface.extractors.read_blackrock`
- drops channels whose names look non-neural, including `Audio`, `Photodiode`,
  `RoomMic`, `MicLine`, `sync`, `trigger`, `LTC`, camera, pupil, strobe, button,
  and analog input labels
- band-pass filters from 1 to 300 Hz
- resamples to 1000 Hz
- saves a computed SpikeInterface binary recording folder in `outputs/`
- writes `lfp_metadata.json` next to the extracted data

SpikeInterface warns that low-frequency LFP filtering is sensitive to chunk
boundaries. This script opts into 1 Hz filtering and defaults to `30s` chunks,
matching their current recommendation to use 30-60 second chunks and large
filter margins for 1-300 Hz LFP extraction.

Useful options:

```bash
python scripts/extract_lfp.py path/to/file.ns5 \
  --freq-min 0.5 \
  --freq-max 300 \
  --target-fs 1000 \
  --n-jobs 4 \
  --chunk-duration 2s \
  --output outputs/my_lfp \
  --overwrite
```

If channel filtering is too aggressive, use `--include-all-channels` or provide
your own repeated `--exclude-name-part` values.

For a quick smoke test before processing the whole recording:

```bash
python scripts/extract_lfp.py data/EMU-0090_subj-YEY_task-Pacman_time-20240402_124118_NSP-2.ns5 \
  --start-s 0 \
  --duration-s 10 \
  --output outputs/test_lfp_10s \
  --overwrite
```

## Notes

SpikeInterface preprocessing is lazy: the filter and resampling pipeline is only
computed when traces are requested or when the recording is saved. The script
uses `save(format="binary")` so downstream analysis can memory-map the LFP data
quickly without rereading and refiltering the original `.ns5`.
