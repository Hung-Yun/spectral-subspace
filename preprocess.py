#%% NOTE: Potentially change filename to downsample.py
#%% Should also create a separate qc.py script for the LFP_QC class

import sys
import os
import argparse
import scipy
import numpy as np
import pandas as pd
import re
from scipy import signal

from utils import _get_repo_datadir, _get_ns5dir

try:
    import spikeinterface as si  # import core only
    import spikeinterface.extractors as se
    import spikeinterface.preprocessing as spre
    import spikeinterface.sorters as ss
    import spikeinterface.postprocessing as spost
    import spikeinterface.comparison as sc
    import spikeinterface.exporters as sexp
    import spikeinterface.curation as scur
    import spikeinterface.widgets as sw
except ModuleNotFoundError:
    si = None
    se = None
    spre = None
    ss = None
    spost = None
    sc = None
    sexp = None
    scur = None
    sw = None

""" TODO: examine the pipeline
OVERALL PREPROCESSING PIPELINE
1. Read NS5/NS3.
2. Extract raw signal, channel info, fs, timestamps, TTL/events.
3. Select channels from target regions.
4. Detect bad channels and bad time intervals.
5. Re-reference.
6. Remove line noise / mark contaminated frequencies.
7. Low-pass anti-alias filter.
8. Downsample to 1 kHz.
9. Save preprocessed LFP matrix: channels × time.
"""


TARGET_LABEL_TO_REGION = {'H': 'HPC','C': 'ACC','A': 'AMY','PH': 'PHG','I': 'INS','OF': 'OFC','OT': 'OTG'}
SENSA_CHANNEL_PATTERN = re.compile(
    r'^m(?P<hemi>[LR])'
    r'(?P<entry>[FPOT]\d+[a-z]?)'
    r'(?P<targets>(?:PH|OF|OT|[A-Z][a-z]?)+)'
    r'(?P<contact>\d+)-(?P<channel>\d+)$'
)
NEURAL_RECORDING_PATTERN = re.compile(
    r'^(?P<emu_label>EMU-(?P<emu_id>\d{1,4}))_subj-(?P<subject>[^_]+)_task-(?P<task>[^_]+)(?:_|$)'
)

REPO_DATADIR = _get_repo_datadir()
SESSIONS_CSV = os.path.join(REPO_DATADIR, 'sessions.csv')
OUTPUT_DIR = os.path.join(REPO_DATADIR, 'neural')
NS5DIR = os.environ.get('SPECTRAL_SUBSPACE_RAWDIR')
DEFAULT_REGIONS = ['HPC']


def get_ns5_path(subject, emu_id, ns5dir=NS5DIR):
    """Construct the expected NS5 file path for a given session."""
    ns5dir = _get_ns5dir(ns5dir)
    subj_folder = os.path.join(ns5dir, subject)
    prefix = f"EMU-{emu_id:04d}"
    session_name = next((f for f in os.listdir(subj_folder) if f.startswith(prefix)), None)
    if session_name is None:
        raise FileNotFoundError(f"No session folder found for EMU-{emu_id:04d} in {subj_folder}")
    
    session_folder = os.path.join(subj_folder, session_name)
    file_name = next((f for f in os.listdir(session_folder) if f.endswith("NSP-2.ns5")), None)
    if file_name is None:
        raise FileNotFoundError(f"No NS5 file found in {session_folder} for EMU-{emu_id:04d}")
    nsp2_ns5_path = os.path.join(session_folder, file_name)

    return nsp2_ns5_path


def load_sessions(ns5dir=NS5DIR):
    """Load the sessions table and attach NS5 metadata."""
    sessions = pd.read_csv(SESSIONS_CSV)
    if 'ns5_path' not in sessions.columns:
        sessions['ns5_path'] = sessions.apply(
            lambda row: get_ns5_path(
                row['patient'],
                row['emu_id'],
                ns5dir=ns5dir,
            ),
            axis=1,
        )
    sessions['size'] = sessions['ns5_path'].apply(lambda p: os.path.getsize(p))
    return sessions

def get_session_row(subject, emu_id, sessions=None):
    """Return the session row for one subject/EMU pair."""
    if sessions is None:
        sessions = load_sessions()

    matches = sessions[
        (sessions['patient'] == subject) &
        (sessions['emu_id'].astype(int) == int(emu_id))
    ]
    if matches.empty:
        raise ValueError(f'No session found for patient={subject}, emu_id={int(emu_id):04d}')
    return matches.iloc[0]

def parse_sensa_channel_name(channel_name):
    """
    Parse SENSA channel names into hemisphere and terminal target region.

    This follows the naming rules in SENSA_mapping_rules.txt:
    side is encoded by L/R, and the distal target label determines region.
    Non-neural analog inputs return NaN values.
    """
    if not isinstance(channel_name, str):
        return pd.Series({'hemi': np.nan, 'region': np.nan})

    match = SENSA_CHANNEL_PATTERN.match(channel_name)
    if match is None:
        return pd.Series({'hemi': np.nan, 'region': np.nan})

    hemisphere = 'left' if match.group('hemi') == 'L' else 'right'
    target_string = match.group('targets')

    # Targets can be chained (proximal -> distal), so keep the last target.
    target_labels = re.findall(r'PH|OF|OT|[A-Z]', target_string)
    target_label = target_labels[-1] if target_labels else None
    region = TARGET_LABEL_TO_REGION.get(target_label, np.nan)

    return pd.Series({'hemi': hemisphere, 'region': region})


def load_blackrock_recording(path_to_file):
    """Load a Blackrock recording using SpikeInterface."""
    if se is None:
        raise ModuleNotFoundError(
            "spikeinterface is required to load Blackrock recordings. "
            "Please activate the environment that has it installed."
        )
    recording = se.read_blackrock(path_to_file)
    return recording

def build_all_chan(recording):
    """Build a DataFrame with channel metadata."""
    channel_ids = recording.get_channel_ids()
    channel_names = recording.get_property("channel_name")
    channel_gains = recording.get_property("gain_to_uV")
    df = pd.DataFrame({
        "channel_id": channel_ids,
        "channel_name": channel_names,
        "gain_to_uV": channel_gains,
    })
    parsed = df['channel_name'].apply(parse_sensa_channel_name)
    df = pd.concat([df, parsed], axis=1)
    return df


def downsample(traces, fs=30000, target_fs=1000, lowpass_hz=400):
    traces_f = signal.sosfiltfilt(signal.butter(6, lowpass_hz, btype='low', fs=fs, output='sos'), traces, axis=0)
    traces_ds = signal.resample_poly(traces_f, up=target_fs, down=fs, axis=0)
    return traces_ds


def get_output_mat_path(ns5_path):
    """Build the default output path for a downsampled LFP .mat file."""
    output_dir = os.path.join(REPO_DATADIR, 'neural')
    stem = os.path.splitext(os.path.basename(ns5_path))[0]
    return os.path.join(output_dir, f'{stem}_ds_lfp.mat')


def parse_neural_recording_filename(path_or_filename):
    """
    Parse subject, EMU ID, and task from a neural recording filename.

    Expected format:
    EMU-0130_subj-YFA_task-WheelOfFortune_run-01_NSP-2.ns5
    """
    stem = os.path.splitext(os.path.basename(path_or_filename))[0]
    match = NEURAL_RECORDING_PATTERN.match(stem)
    if match is None:
        raise ValueError(
            f'Neural recording filename does not match expected pattern: {path_or_filename}'
        )

    parsed = match.groupdict()
    return {
        'subject': parsed['subject'],
        'emu_id': int(parsed['emu_id']),
        'task': parsed['task'],
    }


def save_downsampled_lfp_mat(lfp, output_path, target_fs=1000, lowpass_hz=400):
    """Save downsampled LFP and basic metadata to a MATLAB .mat file."""
    if not hasattr(lfp, 'raw_lfp'):
        lfp.load_raw_lfp()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    lfp_ds = downsample(lfp.raw_lfp, fs=lfp.fs, target_fs=target_fs, lowpass_hz=lowpass_hz)

    scipy.io.savemat(
        output_path,
        {
            'lfp_ds': lfp_ds,
            'fs': float(target_fs),
            'source_fs': float(lfp.fs),
            'subject': lfp.subject,
            'emu_id': lfp.emu_id,
            'task': lfp.task,
            'channel_ids': np.asarray(lfp.chosen_channel_ids),
            'channel_names': lfp.chosen_chan['channel_name'].to_numpy(dtype=object),
            'regions': np.asarray(lfp.regions if lfp.regions is not None else [], dtype=object),
        },
    )
    return output_path


class LFP_processor:

    def __init__(self, path_to_file, regions: list[str] = None):

        # Basic info about the recording
        self.filename = os.path.basename(path_to_file)
        session_metadata = parse_neural_recording_filename(self.filename)
        self.subject = session_metadata['subject']
        self.emu_id = session_metadata['emu_id']
        self.task = session_metadata['task']

        # Load the recording and build the channel table
        self.recording = load_blackrock_recording(path_to_file)
        self.all_chan = build_all_chan(self.recording)

        # Load sampling frequency
        self.fs = self.recording.sampling_frequency
        self.num_segments = self.recording.get_num_segments()
        self.segment_samples = [
            self.recording.get_num_samples(segment_index=i)
            for i in range(self.num_segments)
        ]
        self.segment_index = int(np.argmax(self.segment_samples))
        self.n_samples = self.segment_samples[self.segment_index]
        self.duration_s = self.n_samples / self.fs

        # Set regions of interest
        self.regions = regions
        self.choose_channels(self.regions)

        # Load data
        # self.load_raw_lfp()
        # self.load_raw_analog()

    def choose_channels(self, regions):
        """
        Select channels by parsed brain region labels.

        Example
        -------
        self.choose_channels(['HPC', 'ACC'])
        """
        if regions is None:
            self.chosen_chan = self.all_chan[self.all_chan['region'].notnull()].copy()
            self.chosen_channel_ids = self.chosen_chan.channel_id.tolist()
            return self
        
        if isinstance(regions, str):
            regions = [regions]
        self.chosen_chan = self.all_chan[self.all_chan['region'].isin(regions)].copy()
        self.chosen_channel_ids = self.chosen_chan.channel_id.tolist()
        return self
    
    def load_raw_lfp(self):
        print(' - Reading raw LFP')
        self.raw_lfp = self.recording.get_traces(channel_ids=self.chosen_chan.channel_id.tolist())
        print(' - Done reading raw LFP')
        return self
        

    def load_raw_analog(self, name: list[str] = ['Audio', 'Photodiode']):
        print(' - Reading raw analog')
        self.raw_analog = self.recording.get_traces(channel_ids=
            self.all_chan[self.all_chan['channel_name'].isin(name)].channel_id.tolist()
        )
        return self

    def __repr__(self):
        return 'LFP Processor for ' + self.filename[:-4]

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess one EMU session.')
    parser.add_argument('--subject', type=str, help='Patient code, e.g. YFB.')
    parser.add_argument('--emu-id', type=int, help='EMU id as integer, e.g. 44 for EMU-0044.')
    parser.add_argument(
        '--no-match-sessions',
        action='store_true',
        help='Skip matching against sessions.csv and resolve the NS5 path directly.',
    )
    parser.add_argument(
        '--regions',
        nargs='+',
        default=DEFAULT_REGIONS,
        help='Regions to keep. Default: HPC',
    )
    parser.add_argument(
        '--output-mat',
        type=str,
        help='Optional output path for the downsampled LFP .mat file.',
    )
    parser.add_argument(
        '--target-fs',
        type=int,
        default=1000,
        help='Downsampled sampling rate to save. Default: 1000 Hz.',
    )
    parser.add_argument(
        '--ns5dir',
        type=str,
        help='Optional override for the mounted raw NS5 directory.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ns5dir = _get_ns5dir(args.ns5dir)

    if args.subject is None or args.emu_id is None:
        raise ValueError('Please provide both --subject and --emu-id.')

    if not args.no_match_sessions:
        sessions = load_sessions(ns5dir=ns5dir)
        session = get_session_row(args.subject, args.emu_id, sessions=sessions)
        print(f"Loading {session['patient']} EMU-{int(session['emu_id']):04d}")
        print(f"NS5 path: {session['ns5_path']}")
        print(f"NS5 size (bytes): {int(session['size'])}")

    else:
        session = {
            'patient': args.subject,
            'emu_id': args.emu_id,
            'ns5_path': get_ns5_path(
                args.subject,
                args.emu_id,
                ns5dir=ns5dir,
            ),
        }
        print(f"Loading {session['patient']} EMU-{int(session['emu_id']):04d}")
        print(f"NS5 path: {session['ns5_path']}")

    output_path = args.output_mat or get_output_mat_path(session['ns5_path'])
    if os.path.exists(output_path):
        print(f"Output file already exists. \n\nSkipping processing.")
        return
    else:   
        lfp = LFP_processor(session['ns5_path'], regions=args.regions)
        print(lfp)
        print(f'Chosen channels: {len(lfp.chosen_channel_ids)}')
        print(f'Duration (s): {lfp.duration_s:.2f}')

        print('Reading raw LFP and saving downsampled output')
        save_downsampled_lfp_mat(lfp, output_path, target_fs=args.target_fs)
        print(f'Saved downsampled LFP to {output_path}')

#%%
if __name__ == '__main__':
    main()


#%% OLD SCRIPT Inspect recording segments 

# # Some Blackrock files are split into multiple SpikeInterface segments.
# # Many recording methods need an explicit segment_index in that case.
# num_segments = recording.get_num_segments()
# num_samples = [
#     recording.get_num_samples(segment_index=segment_index)
#     for segment_index in range(num_segments)
# ]
# segment_index = int(np.argmax(num_samples))

# print(f"Using segment_index={segment_index} with {num_samples[segment_index]} samples")
# for i, n_samples in enumerate(num_samples):
#     t1 = recording.get_start_time(segment_index=i)
#     t2 = recording.get_end_time(segment_index=i)
#     print(f"segment {i}: {n_samples} samples, {t1:.4f}s to {t2:.4f}s")

# SEGNUM = 0 if len(num_samples) == 1 else segment_index

# # Pick one neural channel for quick trace sanity checks.
# channel_gains = recording.get_property("gain_to_uV")
# neural_channel_ids = channel_ids[channel_gains == 0.25]
# rng = np.random.default_rng(0)
# ch_id = rng.choice(neural_channel_ids)
# ch_name = channel_names[list(channel_ids).index(ch_id)]
# fs = recording.get_sampling_frequency()
# trace = recording.get_traces(
#     channel_ids=[ch_id],
#     segment_index=SEGNUM,
#     start_frame=0,
#     end_frame=int(fs),  # first second only
# ).flatten()
# print(ch_name, ch_id, trace.shape)


# def load_events(filepath):
#     """
#     loads the events object corresponding to the filepath
#     filepath is the path to the session file (e.g., 'data/sessions/YEW_0033_pursuit.mat')
#     """
#     basedir = os.path.dirname(filepath)
#     eventsdir = os.path.join(basedir, 'events')
#     fnm = os.path.basename(filepath).replace('.mat', '_events.mat')
#     if not os.path.exists(eventsdir):
#         raise Exception(f'Events directory {eventsdir} does not exist. Please check the session file {filepath}.')
#     events_file = os.path.join(eventsdir, fnm)
#     f = scipy.io.loadmat(events_file)
#     return f


# def load_pursuit_trials(filepath):
#     """
#     example:
#        trial_start: 73441
#          iti_start: 73460
#            iti_end: 74100
#         wait_start: 74112
#        chase_start: 75077
#          chase_end: 77697
#     feedback_start: 78050
#          trial_end: 79567
#     """
#     try:
#         f = load_events(filepath)
#     except FileNotFoundError:
#         print(f'WARNING: events file not found for {filepath}. Please check the session file.')
#         return None, []
#     trials = f['events_info'][0,:]
#     # n.b. trial start includes ITI, and trial_end includes feedback period

#     # make dataframe of event times
#     event_order = ['wait_start', 'chase_start', 'chase_end', 'feedback_start', 'trial_end']
#     event_times = {}
#     for event_name in event_order:
#         event_times[event_name] = np.array([x[0,0] for x in trials[event_name]])
#     df = pd.DataFrame(event_times)
#     event_times = list(zip(df['wait_start'].values, df['trial_end'].values))
#     return df, event_times
