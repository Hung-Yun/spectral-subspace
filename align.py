
#%% behavioral data - trajectories

"""
Temporary place here. Will find a place for behavioral data.
"""

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

behavior = Pacman(behavior_path, task_name=recording.task)
for i in range(3):
    x = behavior.trials[i]['continuous']['npc_position_x']['data'][0]
    y = behavior.trials[i]['continuous']['npc_position_y']['data'][0]
    x -= x[np.where(~np.isnan(x))][0] # they are all front loaded with some nans
    y -= y[np.where(~np.isnan(y))][0]
    # plt.scatter(x,y, c=np.arange(len(y)), cmap=plt.cm.viridis, s=3) # optional plotting to see the trajectories.

