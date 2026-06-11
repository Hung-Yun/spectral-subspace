import numpy as np
from spectral import get_psd, get_spectrogram
from utils import BANDS, apply_transform, finish_plot
import matplotlib.pyplot as plt
from scipy import io
from pathlib import Path
import pandas as pd
from brpylib import NevFile

def _maybe_scalar(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.item() if value.size == 1 else value
    if isinstance(value, bytes):
        return value.decode()
    return value


class ProcessedLFP:
    def __init__(self, neural_path):
        """
        Processed LFP recording loaded from a downsampled MAT file.
        """
        self.neural_path = neural_path
        mat = io.loadmat(self.neural_path, squeeze_me=True, struct_as_record=False)
        data = {key: value for key, value in mat.items() if not key.startswith('__')}
        self.traces = np.asarray(data['lfp_ds'], dtype=float)
        self.fs = float(data['fs'])
        self.source_fs = float(data['source_fs'])
        self.subject = _maybe_scalar(data.get('subject'))
        self.emu_id = _maybe_scalar(data.get('emu_id'))
        self.task = _maybe_scalar(data.get('task'))
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

    def plot_lfp_traces(self, start_s, dur_s, n_channels=5):
        n_channels = min(int(n_channels), self.traces.shape[1])
        start = int(round(start_s * self.fs))
        stop = min(int(round((start_s + dur_s) * self.fs)), self.traces.shape[0])
        if start < 0 or start >= stop:
            raise ValueError('Requested trace window is outside the recording.')

        time = np.arange(self.traces.shape[0]) / self.fs
        channel_sd = np.nanstd(self.traces, axis=0)
        channel_names = self.channel_names[:n_channels]
        preview_traces = self.traces[start:stop, :n_channels]
        preview_time = time[start:stop]
        offset = 4 * np.nanmedian(channel_sd[:n_channels])

        plt.figure(figsize=(3,3), dpi=300)
        for channel_idx in range(n_channels):
            plt.plot(
                preview_time,
                preview_traces[:, channel_idx] + channel_idx * offset,
                lw=0.6,
                label=channel_names[channel_idx],
            )
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude + offset')
        plt.title('LFP traces (preview)')
        plt.legend(frameon=False, loc='best', fontsize=5)
        finish_plot()


class LFPAnalyzer:
    """
    Per-channel spectral analysis helper built from one ProcessedLFP recording.

    This class is meant for channel-specific analyses such as PSD, spectrogram,
    band-power extraction, and frequency-domain dimensionality reduction over
    one channel's spectrogram matrix.
    """

    def __init__(self, recording, channel, spec_params):
        self.recording = recording
        self.subject = self.recording.subject
        self.task = self.recording.task
        self.channel_idx = self.recording.get_channel_idx(channel)
        self.channel_id = self.recording.channel_ids[self.channel_idx]
        self.channel_name = self.recording.channel_names[self.channel_idx]
        self.channel_region = self._resolve_channel_region()
        self.trace = np.asarray(self.recording.traces[:, self.channel_idx], dtype=float)
        self.fs = float(self.recording.fs)

        self.psd_f = None
        self.psd = None
        self.pca_results = None
        self.fa_results = None

        self.spec_params = spec_params
        self.spec_time = None
        self.spec_freqs_hz = None
        self.spec_power = None
        self.spec_trace = None
        self.band_powers = {band: None for band in BANDS.keys()}
        self.compute_spectrogram(**self.spec_params) 
        for band in BANDS.keys():
            self.get_band_power(band)

    def _resolve_channel_region(self):
        regions = np.atleast_1d(self.recording.regions).astype(str)
        if regions.size == self.recording.traces.shape[1]:
            return regions[self.channel_idx]
        if regions.size == 1:
            return regions[0]
        if regions.size == 0:
            return None
        return '/'.join(regions)

    def summary(self):
        return {
            'channel_idx': int(self.channel_idx),
            'channel_id': str(self.channel_id),
            'channel_name': str(self.channel_name),
            'channel_region': self.channel_region,
            'fs': self.fs,
            'n_samples': int(self.trace.shape[0]),
            'duration_s': float(self.trace.shape[0] / self.fs),
        }

    def compute_psd(self, window_s=1.0, overlap_frac=0.5, **psd_kwargs):
        self.psd_f, self.psd = get_psd(
            self.trace,
            fs=self.fs,
            window_s=window_s,
            overlap_frac=overlap_frac,
            **psd_kwargs,
        )
        return self

    def compute_spectrogram(self, **spectrogram_kwargs):
        self.spec_trace = self.trace
        self.spec_time, self.spec_freqs_hz, self.spec_power = get_spectrogram(
            self.spec_trace,
            fs=self.fs,
            **spectrogram_kwargs,
        )
        return self

    def get_transformed_matrix(self, source='spec_power', transform='raw', eps=1e-12):
        """
        Return a features x observations matrix after optional preprocessing.

        Source selection stays LFP-specific here, while the actual numeric
        transform lives in `utils.apply_transform` for reuse elsewhere.
        """
        if source in ('trace', 'traces'):
            matrix = self.trace[np.newaxis, :]
        elif source == 'spec_power':
            if self.spec_power is None:
                raise ValueError('Spectrogram power matrix not computed yet. Call compute_spectrogram() first.')
            matrix = self.spec_power
        elif source == 'psd':
            if self.psd is None:
                raise ValueError('PSD not computed yet. Call compute_psd() first.')
            matrix = self.psd[np.newaxis, :]
        else:
            raise ValueError("Unsupported source. Use one of: 'trace', 'spec_power', 'psd'.")

        return apply_transform(matrix, transform=transform, axis=1, eps=eps)

    def compute_corr(self, source='spec_power', transform='raw'):
        data = self.get_transformed_matrix(source=source, transform=transform)
        variable_rows = np.nanstd(data, axis=1) > 0
        corr = np.zeros((data.shape[0], data.shape[0]), dtype=float)

        if np.any(variable_rows):
            if np.sum(variable_rows) == 1:
                corr[np.ix_(variable_rows, variable_rows)] = 1.0
            else:
                corr[np.ix_(variable_rows, variable_rows)] = np.corrcoef(data[variable_rows])
        return corr
    
    def get_band_power(self, band):

        if band not in BANDS:
            raise ValueError(f'Band {band!r} not recognized. Valid bands: {list(BANDS.keys())}.')
        if self.spec_power is None:
            raise ValueError('Spectrogram power matrix not computed yet. Call compute_spectrogram() first.')
        
        if self.band_powers[band] is None:
            band_range = BANDS[band]
            band_mask = (self.spec_freqs_hz >= band_range[0]) & (self.spec_freqs_hz <= band_range[1])
            if not np.any(band_mask):
                raise ValueError(f'No spectrogram frequencies found in the {band} band range {band_range}.')
            
            band_power = np.nanmean(self.spec_power[band_mask, :], axis=0)
            self.band_powers[band] = {
                'time': self.spec_time,
                'power': band_power,
            }

        return self.band_powers[band]
    

class Pacman:
    """
    Pacman loader. This is mostly for the trajectories.

    Sessions are stored as a folder containing:
    - one `sessionVars.mat` file with session-level scalar metadata
    - one `taskVariables.mat` file with task configuration structs
    - many numbered trial files, one MATLAB struct per trial
        
    Reading trajectories
        x: behavior.trials[i]['continuous']['npc_position_x']['data'][0]
        y: behavior.trials[i]['continuous']['npc_position_y']['data'][0]

        x -= x[np.where(~np.isnan(x))][0] # they are all front loaded with some nans
        y -= y[np.where(~np.isnan(y))][0]
    """

    def __init__(self, data_path, task_name):
        self.data_path = Path(data_path)
        self.task_name = task_name.lower()

        self.files = None
        self.session_vars = None
        self.task_variables = None
        self.trials = None
        self.readout = self.load_pacman() # reading all the data at once

        # parsing useful information
        self.time_table = self.build_time_table()

    def _load_mat_file(self, path):
        mat = io.loadmat(path, squeeze_me=True, struct_as_record=False)
        return {key: value for key, value in mat.items() if not key.startswith('__')}

    def _mat_to_python(self, value):
        """Recursively unwrap MATLAB structs/cell arrays into plain Python containers."""
        if hasattr(value, '_fieldnames'):
            return {field: self._mat_to_python(getattr(value, field)) for field in value._fieldnames}
        if isinstance(value, np.ndarray):
            if value.dtype == object:
                if value.ndim == 0:
                    return self._mat_to_python(value.item())
                return [self._mat_to_python(item) for item in value.tolist()]
            return np.asarray(value)
        if isinstance(value, list):
            return [self._mat_to_python(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._mat_to_python(item) for item in value)
        if isinstance(value, bytes):
            return value.decode()
        return value

    def _safe_float(self, value):
        value = float(value)
        if np.isnan(value):
            return None
        return value

    def _safe_int(self, value):
        value = float(value)
        if np.isnan(value):
            return None
        return int(value)

    def _extract_trial(self, trial_path):
        """
        Store all trial variables, regardless if they are called or not. For future reference.
        """
        trial_struct = self._load_mat_file(trial_path)['trialData']
        joystick = np.asarray(trial_struct.joystickPosition, dtype=float).T
        npc_position_x = np.asarray(trial_struct.npcPositionX, dtype=float)
        npc_position_y = np.asarray(trial_struct.npcPositionY, dtype=float)
        eye_samples = np.asarray(trial_struct.eyeSamples)

        return {
            'source_file': str(trial_path),
            'trial_num': int(trial_struct.trialNum),
            'block_num': int(trial_struct.blockNum),
            'events': {
                'trial_start_s': self._safe_float(trial_struct.trialStart),
                'iti_start_s': self._safe_float(trial_struct.itiStart),
                'iti_end_s': self._safe_float(trial_struct.itiEnd),
                'wait_start_s': self._safe_float(trial_struct.waitStart),
                'choice_start_s': self._safe_float(trial_struct.choiceStart),
                'choice_to_feedback_start_s': self._safe_float(trial_struct.choice2feedbackStart),
                'feedback_start_s': self._safe_float(trial_struct.feedbackStart),
                'trial_stop_s': self._safe_float(trial_struct.trialStop),
            },
            'outcomes': {
                'choice_made': self._safe_int(trial_struct.choiceMade),
                'rewarded': self._safe_int(trial_struct.rewarded),
                'reward_value': self._safe_float(trial_struct.rewardValue),
                'iti_s': self._safe_float(trial_struct.iti),
                'wait_time_s': self._safe_float(trial_struct.waitTime),
            },
            'continuous': {
                'joystick': {
                    'columns': ('x', 'y', 'time_s'),
                    'data': joystick,
                    'shape_note': '(n_samples, 3)',
                },
                'npc_position_x': {
                    'data': npc_position_x,
                    'shape_note': '(npc_slot, frame)',
                },
                'npc_position_y': {
                    'data': npc_position_y,
                    'shape_note': '(npc_slot, frame)',
                },
                'eye_samples': {
                    'data': eye_samples,
                    'shape_note': tuple(eye_samples.shape),
                },
            },
            'static': {
                'player_color': self._mat_to_python(trial_struct.playerColor),
                'player_size': self._mat_to_python(trial_struct.playerSize),
                'npc_colors': self._mat_to_python(trial_struct.npcColors),
                'npc_type': self._mat_to_python(trial_struct.npcType),
                'npc_size': self._mat_to_python(trial_struct.npcSize),
                'npc_value': self._mat_to_python(trial_struct.npcValue),
                'npc_velocity': self._mat_to_python(trial_struct.npcVelocity),
                'num_npcs': self._mat_to_python(trial_struct.numNpcs),
                'npc_index': self._mat_to_python(trial_struct.npcIndex),
                'starting_positions': self._mat_to_python(trial_struct.startingPositions),
                'player_start_position': np.asarray(trial_struct.playerStartPosition),
            },
        }

    def load_pacman(self):
        session_vars_path = next(self.data_path.glob('*sessionVars.mat'))
        task_variables_path = next(self.data_path.glob('*taskVariables.mat'))
        trial_paths = sorted(
            (
                path for path in self.data_path.glob('*.mat')
                if 'sessionVars' not in path.name and 'taskVariables' not in path.name
            ),
            key=lambda path: int(path.stem.split('_')[-1]),
        )

        session_vars = self._mat_to_python(self._load_mat_file(session_vars_path)['sessionVars'])
        task_variables = self._load_mat_file(task_variables_path)
        task_variables = {key: self._mat_to_python(value) for key, value in task_variables.items()}
        trials = [self._extract_trial(path) for path in trial_paths]

        self.files = {
                'session_vars': str(session_vars_path),
                'task_variables': str(task_variables_path),
                'trial_files': [str(path) for path in trial_paths],
            }
        
        self.session_vars = session_vars
        self.task_variables = task_variables
        self.trials = trials
        return {
            'files': self.files,
            'session_vars': self.session_vars,
            'task_variables': self.task_variables,
            'trials': self.trials,
        }

    def build_time_table(self):
        rows = []
        for trial in self.trials:
            row = {
                'trial_num': trial['trial_num'],
            }
            row.update(trial['events'])
            rows.append(row)

        return pd.DataFrame(rows).sort_values(['trial_num']).reset_index(drop=True)


class Comments:

    def __init__(self, nev_path, fs=30000):

        self._define_tostring()

        self.nev_path = nev_path
        self.fs = fs
        self.nev = NevFile(str(nev_path))
        self.nev_data = self.nev.getdata()

        self.comments = self.nev_data['comments']['Data']
        self.timestamps = self.nev_data['comments']['TimeStamps']
        self.dur_s = (self.timestamps[-1] - self.timestamps[0]) / self.fs

        self._build_nev_comment_df()
        self._build_nev_trial_event_table()

    def _define_tostring(self):
        """
        Some brpylib versions call `numpy.chararray.tostring()`, but newer NumPy
        builds expose `tobytes()` instead. We add the old method name back so
        `NevFile.getdata()` can run unchanged.
        """
        np.chararray.tostring = np.chararray.tobytes 
        return

    def _build_nev_comment_df(self):
        """
        This function builds a DataFrame from the raw comment strings and timestamps in the NEV file.
        """
        cleaned_comment_strings = [str(x).strip() for x in self.comments]
        empty_comment_indices = [i for i, text in enumerate(cleaned_comment_strings) if text == ""]
        nonempty_comment_strings = [text for text in cleaned_comment_strings if text != ""]

        if len(nonempty_comment_strings) != len(self.timestamps):
            raise ValueError(
                "After removing empty comment strings, NEV comment lengths still do not match: "
                f"len(nonempty_comments)={len(nonempty_comment_strings)}, "
                f"len(self.timestamps)={len(self.timestamps)}."
            )

        df = pd.DataFrame({ "comment": nonempty_comment_strings,
                            "timestamp": np.asarray(self.timestamps, dtype=np.int64),})
        df.attrs["empty_comment_indices"] = empty_comment_indices
        df["comment_base"] = df["comment"].str.extract(r"^([A-Za-z0-9]+)")
        df["prev_comment"] = df["comment"].shift(1)
        df["next_comment"] = df["comment"].shift(-1)
        df["prev_timestamp"] = df["timestamp"].shift(1)
        df["next_timestamp"] = df["timestamp"].shift(-1)
        df["same_as_prev"] = df["comment"].eq(df["prev_comment"])
        df["same_as_next"] = df["comment"].eq(df["next_comment"])
        df["dt_from_prev"] = df["timestamp"] - df["prev_timestamp"]
        df["dt_to_next"] = df["next_timestamp"] - df["timestamp"]

        # zero align to first comment timestamp
        df['timestamp'] = df['timestamp'] - df['timestamp'].iloc[0] 

        # downsampled to 1 kHz
        df['timestamp'] //= 30

        # remove duplicates and non-event metadata comments
        unrelated_comments = {'TASKSTART', 'TASKID', 'trialNumber', 'blockNumber'}
        df = df[~df.same_as_next].copy()
        df = df[~df['comment_base'].isin(unrelated_comments)]
        df = df[['comment', 'timestamp']].reset_index(drop=True)
        self.df_comments = df 


    def _build_nev_trial_event_table(self):
        """
        Build one row per trial from the cleaned NEV comment stream.

        Assumptions:
        - `trialStart` marks the start of a new trial row
        - subsequent event comments belong to the most recent trial
        - `reward` and `unrewarded` are stored as a unified outcome label/timestamp
        """
        rows = []
        current = None

        def finalize_current():
            nonlocal current
            if current is not None:
                rows.append(current)
                current = None

        for row in self.df_comments.itertuples(index=False):
            comment_base = row.comment
            timestamp = int(row.timestamp)

            if comment_base == "trialStart":
                finalize_current()
                current = {
                    "trialStart": timestamp,
                    "itiStart": np.nan,
                    "itiEnd": np.nan,
                    "centralCueStart": np.nan,
                    "choiceStart": np.nan,
                    "feedbackStart": np.nan,
                    "trialEnd": np.nan,
                    "outcome_comment": None,
                    "outcome_timestamp": np.nan,
                }
                continue

            if current is None:
                continue

            if comment_base in (
                "itiStart",
                "itiEnd",
                "centralCueStart",
                "choiceStart",
                "feedbackStart",
                "trialEnd",
            ):
                current[comment_base] = timestamp
                if comment_base == "trialEnd":
                    finalize_current()
            elif comment_base in ("reward", "unrewarded"):
                current["outcome_comment"] = comment_base
                current["outcome_timestamp"] = timestamp

        finalize_current()
        df_trials = pd.DataFrame(rows)
        numeric_cols = [col for col in df_trials.columns if col != "outcome_comment"]
        df_trials[numeric_cols] = df_trials[numeric_cols].astype("Int64")
        self.df_trials = df_trials
