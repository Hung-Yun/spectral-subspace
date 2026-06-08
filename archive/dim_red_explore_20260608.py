"""
Archived exploratory dimensionality-reduction script from June 8, 2026.

This snapshot is a session-level sandbox for testing how single-channel
spectrogram structure behaves under several transforms before any formal
behavior alignment pipeline existed. At a high level it:

1. Loads one processed LFP session and computes spectrograms for selected
   channels.
2. Builds transformed spectrogram matrices such as raw power, log power,
   z-scored power, and z-scored log-power.
3. Compares PCA cumulative variance curves across transforms.
4. Computes a phase-randomized PCA null to ask whether observed low-rank
   structure is stronger than expected from the trace PSD alone.
5. Explores FA, frequency-correlation matrices, and loading patterns as a
   prototype for later analysis code.

Useful landmarks below this docstring:
- Phase randomization helper and PCA-null routine: roughly lines 25-100.
- Session configuration and spectrogram settings: roughly lines 103-130.
- Main per-channel exploration loop: roughly lines 132-248.

This file is archived as a record of exploratory dim-reduction work and is
not intended to be the long-term home for production analysis logic.
"""

#%%

import matplotlib.pyplot as plt
import numpy as np
from scipy import io

from decomposition import FAResults, PCAResults
from processed import ProcessedLFP
from spectral import get_spectrogram
from utils import fig_set, finish_plot


def phase_randomize_trace(trace, rng):
    """Preserve the trace power spectrum while randomizing temporal phase."""
    centered = np.asarray(trace, dtype=float) - np.mean(trace)
    trace_fft = np.fft.rfft(centered)
    randomized_fft = trace_fft.copy()

    randomizable = slice(1, -1 if centered.size % 2 == 0 else None)
    randomized_fft[randomizable] = np.abs(trace_fft[randomizable]) * np.exp(
        1j * rng.uniform(-np.pi, np.pi, randomized_fft[randomizable].size)
    )
    return np.fft.irfft(randomized_fft, n=centered.size) + np.mean(trace)


def compute_pca_null(trace, fs, spectrogram_kwargs, transforms, observed_matrices, n_surrogates=100, seed=0, verbose=True):
    """Compute phase-randomized PCA null curves for one trace and one set of transforms."""
    rng = np.random.default_rng(seed)
    transforms = tuple(transforms)
    observed_matrices = {
        name: np.atleast_2d(np.asarray(matrix, dtype=float))
        for name, matrix in observed_matrices.items()
    }

    def transform_power(power_by_freq_time, transform, eps=1e-12):
        matrix = np.atleast_2d(np.asarray(power_by_freq_time, dtype=float))

        def zscore_rows(rows):
            row_mean = np.nanmean(rows, axis=1, keepdims=True)
            row_std = np.nanstd(rows, axis=1, keepdims=True)
            row_std[row_std == 0] = 1
            return (rows - row_mean) / row_std

        def log_rows(rows):
            row_min = np.nanmin(rows, axis=1, keepdims=True)
            shift = np.where(row_min <= 0, -row_min + eps, 0.0)
            return np.log(rows + shift + eps)

        if transform == 'raw':
            return matrix
        if transform == 'zscore':
            return zscore_rows(matrix)
        if transform == 'log':
            return log_rows(matrix)
        if transform == 'log_zscore':
            return zscore_rows(log_rows(matrix))
        if transform == 'zscore_log':
            return log_rows(zscore_rows(matrix))
        raise ValueError(f'Unsupported transform {transform!r}.')

    observed_results = {}
    null_cumulative_variance = {}
    for name in transforms:
        pca_results = PCAResults()
        pca_results.pca_fit(X=observed_matrices[name].T)
        observed_results[name] = pca_results
        null_cumulative_variance[name] = np.empty((n_surrogates, observed_matrices[name].shape[0]), dtype=float)

    for null_idx in range(n_surrogates):
        if verbose and null_idx % 10 == 0:
            print(f'Running phase-randomized surrogate {null_idx + 1}/{n_surrogates}')

        surrogate_trace = phase_randomize_trace(trace, rng)
        _, _, surrogate_power = get_spectrogram(
            surrogate_trace,
            fs=fs,
            **spectrogram_kwargs,
        )

        for name in transforms:
            surrogate_matrix = transform_power(surrogate_power, name)
            surrogate_pca = PCAResults()
            surrogate_pca.pca_fit(X=surrogate_matrix.T)
            null_cumulative_variance[name][null_idx] = np.cumsum(surrogate_pca.explained_variance_ratio_)

    summary = {}
    for name in transforms:
        observed_curve = np.cumsum(observed_results[name].explained_variance_ratio_)
        null_curves = null_cumulative_variance[name]
        summary[name] = dict(
            observed_cumulative_variance=observed_curve,
            null_curves=null_curves,
            null_median=np.median(null_curves, axis=0),
            null_lower=np.percentile(null_curves, 2.5, axis=0),
            null_upper=np.percentile(null_curves, 97.5, axis=0),
            pc1_p_value=(np.sum(null_curves[:, 0] >= observed_curve[0]) + 1) / (n_surrogates + 1),
            observed_n_pcs_95=np.searchsorted(observed_curve, 0.95) + 1,
            null_n_pcs_95=np.array([np.searchsorted(curve, 0.95) + 1 for curve in null_curves]),
        )
    return summary


plt.figure()
plt.close()
fig_set(font_size=10, linewidth=0.8)

# data_path = 'data/simulation/sim_pinknoise_alpha-1_fs-500_dur-120_nseeds-15.mat'
# data_path = 'data/simulation/sim_oscillation_shared_fs-500_dur-300_nseeds-10.mat'
data_path = 'data/neural/EMU-0130_subj-YFA_task-WheelOfFortune_run-01_NSP-2_ds_lfp.mat'
# data_path = 'data/neural/EMU-0090_subj-YFA_task-Pacman_time-20240424_142255_NSP-2_ds_lfp.mat'

session_name = data_path.split('/')[-1].split('.')[0]
mat = io.loadmat(data_path, squeeze_me=True, struct_as_record=False)
data = {key: value for key, value in mat.items() if not key.startswith('__')}
recording = ProcessedLFP(data)

recording.plot_psd(window_s=1.0, overlap_frac=0.5, max_hz=100)

spectrogram_kwargs = dict(
    start_s=0,
    duration_s=350,
    freqs_hz=np.arange(1, 101, dtype=float),
    fwhm=0.5,
    wavelet_window_s=1.0,
)
pca_null_kwargs = dict(
    n_surrogates=100,
    seed=0,
    verbose=True,
)

channel_indices = range(20,21)
pca_null_transforms = ('raw', 'log_zscore')

for channel_idx in channel_indices:
    recording.compute_spectrogram(
        channel=channel_idx,
        **spectrogram_kwargs,
    )

    spectral_matrices = {
        name: recording.transform_data(source='spec_power', transform=name)
        for name in pca_null_transforms
    }
    spec_log = recording.transform_data(source='spec_power', transform='log')               # freqs x time
    spec_log_z = recording.transform_data(source='spec_power', transform='log_zscore')      # freqs x time

    display_step = max(int(round(recording.fs / 100)), 1)
    plt.figure(figsize=(6, 3), dpi=300)
    plt.pcolormesh(
        recording.spec_time[::display_step],
        recording.spec_freqs_hz,
        spec_log[:, ::display_step],
        cmap='viridis',
        shading='auto',
    )
    plt.colorbar(label='Log power')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(f'Spectrogram: {recording.channel_names[recording.spec_channel_idx]}')
    finish_plot(filename='spectrogram', save_dir=f'plots/exploratory/{session_name}/channel_{channel_idx}', exts=('png',))

    pca_results = PCAResults()
    plt.figure(figsize=(3, 3), dpi=300)
    for name, value in spectral_matrices.items():
        pca_results.pca_fit(X=value.T)
        plt.plot(
            np.cumsum(pca_results.explained_variance_ratio_),
            marker='o',
            label=f'{name}: D={pca_results.d_shared}',
        )
        plt.xlabel('Number of components')
        plt.ylabel('Proportion of variance explained')
        plt.legend(frameon=False, loc='best', fontsize=7)
    plt.title('PCA cumulative explained variance')
    finish_plot(
        filename='pca_cumulative_explained_variance',
        save_dir=f'plots/exploratory/{session_name}/channel_{channel_idx}',
        exts=('png',),
    )

    pca_null = compute_pca_null(
        trace=recording.spec_trace,
        fs=recording.fs,
        spectrogram_kwargs={key: value for key, value in spectrogram_kwargs.items() if key not in ('start_s', 'duration_s')},
        transforms=tuple(spectral_matrices.keys()),
        observed_matrices=spectral_matrices,
        **pca_null_kwargs,
    )

    component_numbers = np.arange(1, recording.spec_power.shape[0] + 1)
    fig, ax = plt.subplots(1, len(spectral_matrices), figsize=(3.5 * len(spectral_matrices), 3), dpi=300)
    if len(spectral_matrices) == 1:
        ax = np.array([ax])
    for ax_i, name in zip(np.ravel(ax), spectral_matrices):
        summary = pca_null[name]
        ax_i.fill_between(component_numbers,summary['null_lower'],summary['null_upper'],color='0.85',label='Phase-null 95% interval',)
        ax_i.plot(component_numbers,summary['observed_cumulative_variance'],lw=1,color='r',label='Observed',)
        ax_i.plot(component_numbers,summary['null_median'],lw=1,color='k',ls='--',label='Phase-null median',)
        ax_i.set_xlabel('Number of PCA components')
        ax_i.set_ylabel('Cumulative variance explained')
        ax_i.set_title(f'{name}: p={summary["pc1_p_value"]:.3f}, ' f'D95={summary["observed_n_pcs_95"]}' )
    np.ravel(ax)[0].legend(frameon=False, loc='best', fontsize=6)
    plt.tight_layout()
    finish_plot(filename='pca_phase_randomized_null',save_dir=f'plots/exploratory/{session_name}/channel_{channel_idx}',exts=('png',),)

    fa_results = FAResults()
    plt.figure(figsize=(3, 3), dpi=300)
    for name, value in spectral_matrices.items():
        fa_results.fa_fit(X=value.T, shared_var_thresh=0.95, max_iter=int(1e6), tol=1e-6, verbose=False)
        plt.plot(np.cumsum(fa_results.explained_variance_ratio_),marker='o',label=f'{name}: D={fa_results.d_shared}',)
        plt.xlabel('Number of components')
        plt.ylabel('Proportion of variance explained')
        plt.legend(frameon=False, loc='best', fontsize=7)
    plt.title('FA cumulative explained variance')
    finish_plot(filename='fa_cumulative_explained_variance',save_dir=f'plots/exploratory/{session_name}/channel_{channel_idx}',exts=('png',),)

    for transform in ['log_zscore', 'zscore']:
        freq_corr = recording.compute_corr(source='spec_power', transform=transform)

        plt.figure(figsize=(3.5, 3), dpi=300)
        plt.imshow(freq_corr, aspect='auto', origin='lower', cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(label='Correlation')
        plt.xticks(np.linspace(0, 100, 11))
        plt.yticks(np.linspace(0, 100, 11))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Frequency (Hz)')
        finish_plot(filename=f'correlation_{transform}', save_dir=f'plots/exploratory/{session_name}/channel_{channel_idx}', exts=('png',))

    fa_results = FAResults()
    fa_results.fa_fit(X=spec_log_z.T,n_components=10,max_iter=int(1e6),tol=1e-6,verbose=True,)

    plt.figure(figsize=(3, 3), dpi=300)
    plt.plot(fa_results.shared_var_per_unit, marker='o')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Proportion of shared variance')
    plt.title(f'FA shared var per frequency: D={fa_results.d_shared}')
    finish_plot(filename='fa_shared_var_per_unit', save_dir=f'plots/exploratory/{session_name}/channel_{channel_idx}', exts=('png',))

    fig, ax = plt.subplots(2, 5, figsize=(10, 4), dpi=300)
    for i in range(fa_results.d_shared):
        ax_i = ax.flatten()[i]
        ax_i.plot(recording.spec_freqs_hz, np.abs(fa_results.subspace[i]))
        ax_i.set_title(f'PC {i + 1}: {fa_results.explained_variance_ratio_[i]:.2f}')
        ax_i.set_xlabel('Frequency (Hz)')
        ax_i.set_ylabel('Loading')
    plt.tight_layout()
    finish_plot(filename='fa_loadings', save_dir=f'plots/exploratory/{session_name}/channel_{channel_idx}', exts=('png',))

#%%
