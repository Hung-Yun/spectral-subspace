#!/usr/bin/env python
"""
Prepare Pacman neural and behavioral files from Wrangell into this project.

Example
-------
python prepare_wrangell_data.py EMU-0090_subj-YFA_task-Pacman

This will:
1. find the matching neural session folder under
   ~/wrangell/stitched/EMU-18112/<SUBJECT>
2. downsample the NSP-2 NS5 file into data/neural
3. copy the NSP-2 NEV file into data/neural
4. copy the matching behavior folder from
   ~/wrangell/datalake/emu/<SUBJECT>Datafile/BEHAV/Pacman
   into data/behavior/<SESSION_NAME>
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
import re


SESSION_PATTERN = re.compile(
    r'(?P<session_name>EMU-(?P<emu_id>\d{4})_subj-(?P<subject>[^_]+)_task-Pacman)'
)
DATETIME_PATTERN = re.compile(r'(?:time-|__)(?P<datetime>\d{8}_\d{6})')


def expand_path(path):
    return Path(path).expanduser().resolve()


def parse_session(session):
    match = SESSION_PATTERN.search(session)
    if match is None:
        raise ValueError(
            f"Could not parse Pacman session from {session!r}. "
            "Expected something like EMU-0090_subj-YFA_task-Pacman."
        )

    datetime_match = DATETIME_PATTERN.search(session)
    return {
        'session_name': match.group('session_name'),
        'subject': match.group('subject'),
        'emu_id': match.group('emu_id'),
        'datetime': datetime_match.group('datetime') if datetime_match else None,
    }


def find_one(candidates, description):
    candidates = sorted(candidates)
    if len(candidates) == 0:
        raise FileNotFoundError(f'No {description} found.')
    if len(candidates) > 1:
        formatted = '\n'.join(f'  {path}' for path in candidates)
        raise RuntimeError(f'Found multiple {description} candidates:\n{formatted}')
    return candidates[0]


def find_neural_session_dir(session_info, stitched_root):
    subject_root = stitched_root / session_info['subject']
    if not subject_root.exists():
        raise FileNotFoundError(f'Subject neural root does not exist: {subject_root}')

    candidates = [
        path for path in subject_root.iterdir()
        if path.is_dir() and path.name.startswith(session_info['session_name'])
    ]
    if session_info['datetime'] is not None:
        candidates = [path for path in candidates if session_info['datetime'] in path.name]

    return find_one(candidates, f"neural session folder for {session_info['session_name']}")


def find_nsp2_file(neural_session_dir, suffix):
    suffix = suffix.lower()
    candidates = [
        path for path in neural_session_dir.iterdir()
        if path.is_file()
        and path.name.lower().endswith(suffix)
        and 'nsp-2' in path.name.lower()
    ]
    return find_one(candidates, f'NSP-2 {suffix} file in {neural_session_dir}')


def extract_datetime(*names):
    for name in names:
        match = DATETIME_PATTERN.search(str(name))
        if match is not None:
            return match.group('datetime')
    raise ValueError(f'Could not find datetime tag in: {names}')


def find_behavior_dir(session_info, behavior_root, datetime_tag):
    subject = session_info['subject']
    subject_behavior_root = behavior_root / f'{subject}Datafile' / 'BEHAV' / 'Pacman'
    if not subject_behavior_root.exists():
        raise FileNotFoundError(f'Subject behavior root does not exist: {subject_behavior_root}')

    exact = subject_behavior_root / f'Pacman__{subject}__{datetime_tag}'
    if exact.exists():
        return exact

    candidates = [
        path for path in subject_behavior_root.iterdir()
        if path.is_dir() and path.name.startswith(f'Pacman__{subject}__') and datetime_tag in path.name
    ]
    return find_one(candidates, f'behavior folder for {subject} at {datetime_tag}')


def copy_file(src, dst, overwrite=False, dry_run=False):
    if dst.exists() and not overwrite:
        print(f'SKIP existing file: {dst}')
        return dst
    print(f'COPY file: {src} -> {dst}')
    if dry_run:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def copy_folder(src, dst, overwrite=False, dry_run=False):
    if dst.exists():
        if not overwrite:
            print(f'SKIP existing folder: {dst}')
            return dst
        print(f'REPLACE folder: {dst}')
        if not dry_run:
            shutil.rmtree(dst)

    print(f'COPY folder: {src} -> {dst}')
    if dry_run:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    return dst


def run_downsample(ns5_path, output_mat, regions, target_fs, overwrite=False, dry_run=False):
    if output_mat.exists() and not overwrite:
        print(f'SKIP existing downsampled MAT: {output_mat}')
        return output_mat

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / 'downsample.py'),
        '--input-ns5',
        str(ns5_path),
        '--output-mat',
        str(output_mat),
        '--target-fs',
        str(target_fs),
        '--regions',
        *regions,
    ]
    print('RUN:', ' '.join(cmd))
    if dry_run:
        return output_mat
    output_mat.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)
    return output_mat


def prepare_session(session, args):
    session_info = parse_session(session)
    session_name = session_info['session_name']

    stitched_root = expand_path(args.stitched_root)
    behavior_root = expand_path(args.behavior_root)
    dest_root = expand_path(args.dest_root)
    dest_neural = dest_root / 'data' / 'neural'
    dest_behavior = dest_root / 'data' / 'behavior'

    print(f'\n=== {session_name} ===')
    neural_session_dir = find_neural_session_dir(session_info, stitched_root)
    ns5_path = find_nsp2_file(neural_session_dir, '.ns5')
    nev_path = find_nsp2_file(neural_session_dir, '.nev')
    datetime_tag = session_info['datetime'] or extract_datetime(neural_session_dir.name, ns5_path.name)
    behavior_dir = find_behavior_dir(session_info, behavior_root, datetime_tag)

    print(f'Neural folder: {neural_session_dir}')
    print(f'NS5: {ns5_path}')
    print(f'NEV: {nev_path}')
    print(f'Behavior folder: {behavior_dir}')

    output_mat = dest_neural / f'{ns5_path.stem}_ds_lfp.mat'
    output_nev = dest_neural / nev_path.name
    output_behavior = dest_behavior / session_name

    if not args.skip_downsample:
        run_downsample(
            ns5_path=ns5_path,
            output_mat=output_mat,
            regions=args.regions,
            target_fs=args.target_fs,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
    if not args.skip_nev:
        copy_file(nev_path, output_nev, overwrite=args.overwrite, dry_run=args.dry_run)
    if not args.skip_behavior:
        copy_folder(behavior_dir, output_behavior, overwrite=args.overwrite, dry_run=args.dry_run)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Batch prepare Pacman NS5/NEV/behavior data from Wrangell into spectral-subspace.'
    )
    parser.add_argument(
        'sessions',
        nargs='+',
        help='Session names, e.g. EMU-0090_subj-YFA_task-Pacman. Full time-tagged names also work.',
    )
    parser.add_argument(
        '--stitched-root',
        default='~/wrangell/stitched/EMU-18112',
        help='Root containing per-subject neural folders. Default: ~/wrangell/stitched/EMU-18112',
    )
    parser.add_argument(
        '--behavior-root',
        default='~/wrangell/datalake/emu',
        help='Root containing <SUBJECT>Datafile/BEHAV/Pacman. Default: ~/wrangell/datalake/emu',
    )
    parser.add_argument(
        '--dest-root',
        default='~/wrangell/hungyun-elias/spectral-subspace',
        help='Destination repo root. Default: ~/wrangell/hungyun-elias/spectral-subspace',
    )
    parser.add_argument('--regions', nargs='+', default=['HPC'], help='Regions passed to downsample.py. Default: HPC')
    parser.add_argument('--target-fs', type=int, default=1000, help='Downsampled sampling rate. Default: 1000')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing copied/downsampled outputs.')
    parser.add_argument('--dry-run', action='store_true', help='Print planned actions without copying or downsampling.')
    parser.add_argument('--skip-downsample', action='store_true', help='Do not downsample NS5 files.')
    parser.add_argument('--skip-nev', action='store_true', help='Do not copy NEV files.')
    parser.add_argument('--skip-behavior', action='store_true', help='Do not copy behavior folders.')
    return parser.parse_args()


def main():
    args = parse_args()
    for session in args.sessions:
        prepare_session(session, args)


if __name__ == '__main__':
    main()
