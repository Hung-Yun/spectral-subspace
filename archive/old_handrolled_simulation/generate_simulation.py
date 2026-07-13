import argparse
import json
import os

import numpy as np
from scipy import io

from archive.old_handrolled_simulation.simulate_core import generate_oscillation_trace, generate_pinknoise_trace


def parse_args():
    """Parse CLI arguments for simulation generation.

    This parser is intentionally flat for now because the simulation families
    still share most arguments. Once the method-specific options grow further,
    refactor this into argparse subparsers so each method can define its own
    arguments cleanly:

    - shared args: output path, overwrite flag, sampling rate, duration, seeds
    - oscillation args: envelope mode, frequencies, amplitudes, phases, etc.
    - pink-noise args: PSD-shaping parameters and any method-specific options

    At that point, prefer a structure like:
    `generate_simulation.py oscillation ...`
    `generate_simulation.py pinknoise ...`
    """
    parser = argparse.ArgumentParser(
        description='Generate simulated traces and save them in a ProcessedLFP-compatible MAT file.'
    )
    parser.add_argument(
        '--method',
        default='oscillation',
        choices=['oscillation', 'pinknoise'],
        help='Simulation family to generate.',
    )
    parser.add_argument(
        '--envelope-mode',
        default='shared',
        choices=['constant', 'ind', 'shared'],
        help='Envelope mode for oscillation simulations.',
    )
    parser.add_argument(
        '--sim-fs',
        type=float,
        default=500,
        help='Sampling rate of the simulated traces in Hz.',
    )
    parser.add_argument(
        '--sim-duration-s',
        type=float,
        default=600,
        help='Simulation duration in seconds.',
    )
    parser.add_argument(
        '--freqs-hz',
        nargs='+',
        type=float,
        default=[12, 30, 70],
        help='Oscillation center frequencies in Hz.',
    )
    parser.add_argument(
        '--base-amplitudes',
        nargs='+',
        type=float,
        default=[30, 28, 25],
        help='Baseline oscillation amplitudes for each active frequency.',
    )
    parser.add_argument(
        '--phases-rad',
        nargs='+',
        type=float,
        default=[0.0, 0.6, 1.1],
        help='Oscillation phases in radians for each active frequency.',
    )
    parser.add_argument(
        '--envelope-scales',
        nargs='+',
        type=float,
        default=[12, 8, 4],
        help='Envelope fluctuation scale for each active frequency.',
    )
    parser.add_argument(
        '--smooth-window-s',
        type=float,
        default=0.5,
        help='Smoothing window for the envelope fluctuations in seconds.',
    )
    parser.add_argument(
        '--additive-noise-sd',
        type=float,
        default=0.0,
        help='Standard deviation of additive white noise applied to the trace.',
    )
    parser.add_argument(
        '--pink-alpha',
        type=float,
        default=1.0,
        help='Power-law exponent alpha used by the pink-noise PSD, P(f) ~ 1/f^alpha.',
    )
    parser.add_argument(
        '--pink-sd',
        type=float,
        default=1.0,
        help='Target standard deviation of each generated pink-noise trace.',
    )
    parser.add_argument(
        '--pink-mean',
        type=float,
        default=0.0,
        help='Mean offset added to each generated pink-noise trace after scaling.',
    )
    parser.add_argument(
        '--pink-fmin-hz',
        type=float,
        default=None,
        help='Low-frequency floor used in 1/f^alpha shaping. Defaults to the first nonzero FFT bin.',
    )
    parser.add_argument(
        '--n-seeds',
        type=int,
        default=10,
        help='Number of random seeds to generate when --seeds is not provided.',
    )
    parser.add_argument(
        '--seed-start',
        type=int,
        default=0,
        help='First seed used when generating a contiguous seed range.',
    )
    parser.add_argument(
        '--seeds',
        nargs='+',
        type=int,
        help='Optional explicit list of seeds. Overrides --n-seeds and --seed-start.',
    )
    parser.add_argument(
        '--output-mat',
        type=str,
        help='Optional output path for the MAT file.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite an existing output file.',
    )
    return parser.parse_args()


def resolve_seeds(params):
    if params.seeds is not None:
        seeds = np.asarray(params.seeds, dtype=int)
    else:
        seeds = np.arange(params.seed_start, params.seed_start + params.n_seeds, dtype=int)

    if seeds.size < 2:
        raise ValueError(
            'Use at least 2 seeds so the saved lfp_ds array stays channel-like for ProcessedLFP.'
        )
    return seeds


def build_output_path(params, seeds):
    if params.output_mat:
        return params.output_mat

    if params.method == 'oscillation':
        method_tag = params.envelope_mode
    elif params.method == 'pinknoise':
        method_tag = f'alpha-{params.pink_alpha:g}'
    else:
        method_tag = params.method

    filename = (
        f'sim_{params.method}_{method_tag}'
        f'_fs-{params.sim_fs:g}'
        f'_dur-{params.sim_duration_s:g}'
        f'_nseeds-{seeds.size}.mat'
    )
    return os.path.join('data', 'simulation', filename)


def build_oscillation_params(params, seeds):
    return {
        'method': params.method,
        'envelope_mode': params.envelope_mode,
        'sim_fs': float(params.sim_fs),
        'sim_duration_s': float(params.sim_duration_s),
        'freqs_hz': list(params.freqs_hz),
        'base_amplitudes': list(params.base_amplitudes),
        'phases_rad': list(params.phases_rad),
        'envelope_scales': list(params.envelope_scales),
        'smooth_window_s': float(params.smooth_window_s),
        'additive_noise_sd': float(params.additive_noise_sd),
        'seeds': seeds.tolist(),
    }


def build_pinknoise_params(params, seeds):
    return {
        'method': params.method,
        'sim_fs': float(params.sim_fs),
        'sim_duration_s': float(params.sim_duration_s),
        'pink_alpha': float(params.pink_alpha),
        'pink_sd': float(params.pink_sd),
        'pink_mean': float(params.pink_mean),
        'pink_fmin_hz': None if params.pink_fmin_hz is None else float(params.pink_fmin_hz),
        'seeds': seeds.tolist(),
    }


def build_simulation_params(params, seeds):
    if params.method == 'oscillation':
        return build_oscillation_params(params, seeds)
    if params.method == 'pinknoise':
        return build_pinknoise_params(params, seeds)

    raise NotImplementedError(f'Unsupported simulation method: {params.method}')


def generate_oscillation(seeds, sim_params):
    sim_time = np.arange(0, sim_params['sim_duration_s'], 1 / sim_params['sim_fs'])
    sim_kwargs = dict(
        sim_time=sim_time,
        freqs_hz=np.asarray(sim_params['freqs_hz'], dtype=float),
        base_amplitudes=np.asarray(sim_params['base_amplitudes'], dtype=float),
        phases_rad=np.asarray(sim_params['phases_rad'], dtype=float),
        envelope_scales=np.asarray(sim_params['envelope_scales'], dtype=float),
        smooth_window_s=float(sim_params['smooth_window_s']),
        additive_noise_sd=float(sim_params['additive_noise_sd']),
    )

    traces = []
    envelopes = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        trace, envelope = generate_oscillation_trace(
            **sim_kwargs,
            rng=rng,
            envelope_mode=sim_params['envelope_mode'],
        )
        traces.append(trace)
        envelopes.append(envelope)

    return {
        'traces': np.column_stack(traces),
        'sim_time': sim_time,
        'extra_mat_fields': {
            'active_freqs_hz': np.asarray(sim_params['freqs_hz'], dtype=float),
            'base_amplitudes': np.asarray(sim_params['base_amplitudes'], dtype=float),
            'phases_rad': np.asarray(sim_params['phases_rad'], dtype=float),
            'envelope_scales': np.asarray(sim_params['envelope_scales'], dtype=float),
            'smooth_window_s': float(sim_params['smooth_window_s']),
            'additive_noise_sd': float(sim_params['additive_noise_sd']),
            'envelope_mode': np.asarray([sim_params['envelope_mode']], dtype=object),
            'envelopes': np.stack(envelopes, axis=-1),
        },
    }


def generate_pinknoise(seeds, sim_params):
    sim_time = np.arange(0, sim_params['sim_duration_s'], 1 / sim_params['sim_fs'])

    traces = []
    fft_freqs_hz = None
    target_power = None
    effective_fmin_hz = None
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        trace, fft_freqs_hz, target_power, effective_fmin_hz = generate_pinknoise_trace(
            sim_time=sim_time,
            rng=rng,
            alpha=float(sim_params['pink_alpha']),
            target_sd=float(sim_params['pink_sd']),
            mean=float(sim_params['pink_mean']),
            fmin_hz=sim_params['pink_fmin_hz'],
        )
        traces.append(trace)

    return {
        'traces': np.column_stack(traces),
        'sim_time': sim_time,
        'extra_mat_fields': {
            'pink_alpha': float(sim_params['pink_alpha']),
            'pink_sd': float(sim_params['pink_sd']),
            'pink_mean': float(sim_params['pink_mean']),
            'pink_fmin_hz': float(effective_fmin_hz),
            'pink_fft_freqs_hz': np.asarray(fft_freqs_hz, dtype=float),
            'pink_target_power': np.asarray(target_power, dtype=float),
        },
    }


def generate(params):
    seeds = resolve_seeds(params)
    sim_params = build_simulation_params(params, seeds)

    if params.method == 'oscillation':
        sim_data = generate_oscillation(seeds, sim_params)
    elif params.method == 'pinknoise':
        sim_data = generate_pinknoise(seeds, sim_params)
    else:
        raise NotImplementedError(f'Unsupported simulation method: {params.method}')

    sim_data.update(
        seeds=seeds,
        sim_params=sim_params,
        method=params.method,
    )
    return sim_data


def save_simulation_mat(sim_data, output_path):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    seeds = sim_data['seeds']
    sim_params = sim_data['sim_params']

    mat_payload = {
        'lfp_ds': np.asarray(sim_data['traces'], dtype=float),
        'fs': float(sim_params['sim_fs']),
        'source_fs': float(sim_params['sim_fs']),
        'subject': 'SIM',
        'emu_id': -1,
        'task': sim_data['method'],
        'channel_ids': np.asarray(seeds, dtype=int),
        'channel_names': np.asarray([f'sim_{seed}' for seed in seeds], dtype=object),
        'regions': np.asarray(['SIM'], dtype=object),
        'simulation_method': np.asarray([sim_data['method']], dtype=object),
        'simulation_params_json': json.dumps(sim_params, sort_keys=True),
        'seeds': np.asarray(seeds, dtype=int),
        'sim_time': np.asarray(sim_data['sim_time'], dtype=float),
    }
    mat_payload.update(sim_data.get('extra_mat_fields', {}))
    io.savemat(output_path, mat_payload)


def main():
    args = parse_args()
    sim_data = generate(args)
    output_path = build_output_path(args, sim_data['seeds'])

    if os.path.exists(output_path) and not args.overwrite:
        raise FileExistsError(f'Output file already exists: {output_path}')

    save_simulation_mat(sim_data, output_path)
    print(f'Saved simulated MAT file to {output_path}')
    print(
        f"Shape: {sim_data['traces'].shape[0]:,} samples x "
        f"{sim_data['traces'].shape[1]} seeds"
    )


if __name__ == '__main__':
    main()
