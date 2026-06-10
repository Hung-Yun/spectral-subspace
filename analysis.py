#%%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import io, signal

from decomposition import FAResults, PCAResults
from processed import ProcessedLFP
from spectral import get_psd, get_spectrogram
from utils import fig_set, finish_plot

from brpylib import NevFile, NsxFile

DATA_TEMP_DIR = Path("data/temp")

BANDS = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta': (13.0, 30.0),
    'gamma': (30.0, 70.0),
}


plt.figure()
plt.close()
fig_set(font_size=10, linewidth=0.8)

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

    def fit_pca(self, source='spec_power', transform='log_zscore', **pca_kwargs):
        X = self.get_transformed_matrix(source=source, transform=transform).T
        self.pca_results = PCAResults()
        self.pca_results.pca_fit(X=X, **pca_kwargs)
        return self.pca_results

    def fit_fa(self, source='spec_power', transform='log_zscore', **fa_kwargs):
        X = self.get_transformed_matrix(source=source, transform=transform).T
        self.fa_results = FAResults()
        self.fa_results.fa_fit(X=X, **fa_kwargs)
        return self.fa_results
    

class Pacman:
    """
    Pacman loader. This is mostly for the trajectories.

    Sessions are stored as a folder containing:
    - one `sessionVars.mat` file with session-level scalar metadata
    - one `taskVariables.mat` file with task configuration structs
    - many numbered trial files, one MATLAB struct per trial

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

        # remove duplicates
        # df = df[~df.same_as_next][['comment', 'timestamp', 'comment_base']].reset_index(drop=True)
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
            comment_base = row.comment_base
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
        self.df_trials = pd.DataFrame(rows)


#%% TODO: should unify how and where to read data
session_name = 'EMU-0044_subj-YFB_task-Pacman'

def _find_session_path(folder, session_name):
    for filename in os.listdir(folder):
        if filename.startswith(session_name):
            return os.path.join(folder, filename)
    raise FileNotFoundError(f'Could not find a file/folder for session {session_name!r} in {folder!r}.')

neural_path = _find_session_path('data/neural', session_name)
behavior_path = _find_session_path('data/behavior', session_name)

def find_sample_blackrock_pair(data_dir=DATA_TEMP_DIR):
    ns5_files = sorted(data_dir.glob("*.ns5"))
    nev_files = sorted(data_dir.glob("*.nev"))
    if not ns5_files:
        raise FileNotFoundError(f"No .ns5 files found in {data_dir}.")
    if not nev_files:
        raise FileNotFoundError(f"No .nev files found in {data_dir}.")

    for ns5_path in ns5_files:
        paired_nev = data_dir / f"{ns5_path.stem}.nev"
        if paired_nev.exists():
            return ns5_path, paired_nev

    raise FileNotFoundError(
        f"Found .ns5 and .nev files in {data_dir}, but no matching paired stems."
    )
# ns5_path, nev_path = find_sample_blackrock_pair(data_dir=DATA_TEMP_DIR)


#%% Read downsampled LFP data and then find spectrogram in one of the channel
mat = io.loadmat(neural_path, squeeze_me=True, struct_as_record=False)
data = {key: value for key, value in mat.items() if not key.startswith('__')}
recording = ProcessedLFP(data)
spectrogram_kwargs = dict(freqs_hz=np.arange(1, 101, dtype=float),fwhm=0.5,wavelet_window_s=2,)
lfp = LFPAnalyzer(recording, channel=0, spec_params=spectrogram_kwargs)

# TODO: putting these in init so that LFPAnalyzer takes an argument of .
lfp.compute_spectrogram(**spectrogram_kwargs) 
for band in BANDS.keys():
    lfp.get_band_power(band)

#%% Read behavioral data
behavior = Pacman(behavior_path, task_name=recording.task)
# for i in range(3):
#     x = behavior.trials[i]['continuous']['npc_position_x']['data'][0]
#     y = behavior.trials[i]['continuous']['npc_position_y']['data'][0]
#     x -= x[np.where(~np.isnan(x))][0] # they are all front loaded with some nans
#     y -= y[np.where(~np.isnan(y))][0]
    # plt.scatter(x,y, c=np.arange(len(y)), cmap=plt.cm.viridis, s=3) # optional plotting to see the trajectories.


#%%
neural_folder = '/Volumes/stitched/EMU-18112/YFB/EMU-0044_subj-YFB_task-Pacman_time-20240506_115804'
ns5_path = os.path.join(neural_folder, 'EMU-0044_subj-YFB_task-Pacman_time-20240506_115804_NSP-2.ns5')

# So ns5 and nev must be in the same folder
# ended with NSP-2.nev or NSP-2.ns5.
# Should be easy to process this on the server.
ns5 = NsxFile(str(ns5_path))
ns5_data = ns5.getdata()
fs = ns5_data['samp_per_s']
ns5_dur_s = ns5_data['data'][0].shape[1] / fs
#%%
nev_path = os.path.join(neural_folder, 'EMU-0044_subj-YFB_task-Pacman_time-20240506_115804_NSP-2.nev')
nev = Comments(nev_path)

# %% plot band power, will put event lines on top of them

fig, ax = plt.subplots(5,1,figsize=(6,8), dpi=300, sharex=True)
ix = slice(5000, 30000)
for i, band in enumerate(BANDS.keys()):
    band_power = lfp.get_band_power(band)
    band_power['power'] = (band_power['power']- np.nanmean(band_power['power'])) / np.nanstd(band_power['power'])
    ax[i].plot(band_power['time'][ix], band_power['power'][ix], label=band)
    ax[i].legend(frameon=False, loc='best')
plt.xlabel('Time (s)')
plt.ylabel('Band power')
finish_plot()



# %%
