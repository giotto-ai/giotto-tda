# License: GNU AGPLv3

from numbers import Real
from typing import Callable

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.utils import gen_even_slices
from sklearn.utils.validation import _num_samples

from ._utils import _subdiagrams, _sample_image
from ..externals.modules.gtda_bottleneck import bottleneck_distance
from ..externals.modules.gtda_wasserstein import wasserstein_distance
from ..utils.intervals import Interval

_AVAILABLE_METRICS = {
    'bottleneck': {
        'delta': {'type': Real, 'in': Interval(0, 1, closed='both')}
        },
    'wasserstein': {
        'p': {'type': Real, 'in': Interval(1, np.inf, closed='left')},
        'delta': {'type': Real, 'in': Interval(0, 1, closed='right')}
        },
    'betti': {
        'p': {'type': Real, 'in': Interval(1, np.inf, closed='both')},
        'n_bins': {'type': int, 'in': Interval(1, np.inf, closed='left')}
        },
    'landscape': {
        'p': {'type': Real, 'in': Interval(1, np.inf, closed='both')},
        'n_bins': {'type': int, 'in': Interval(1, np.inf, closed='left')},
        'n_layers': {'type': int, 'in': Interval(1, np.inf, closed='left')}
        },
    'heat': {
        'p': {'type': Real, 'in': Interval(1, np.inf, closed='both')},
        'n_bins': {'type': int, 'in': Interval(1, np.inf, closed='left')},
        'sigma': {'type': Real, 'in': Interval(0, np.inf, closed='neither')}
        },
    'persistence_image': {
        'p': {'type': Real, 'in': Interval(1, np.inf, closed='both')},
        'n_bins': {'type': int, 'in': Interval(1, np.inf, closed='left')},
        'sigma': {'type': Real, 'in': Interval(0, np.inf, closed='neither')},
        'weight_function': {'type': (Callable, type(None))}
        },
    'silhouette': {
        'power': {'type': Real, 'in': Interval(0, np.inf, closed='right')},
        'p': {'type': Real, 'in': Interval(1, np.inf, closed='both')},
        'n_bins': {'type': int, 'in': Interval(1, np.inf, closed='left')}
        }
    }

_AVAILABLE_AMPLITUDE_METRICS = {}
for _metric, _metric_params in _AVAILABLE_METRICS.items():
    if _metric not in ['bottleneck', 'wasserstein']:
        _AVAILABLE_AMPLITUDE_METRICS[_metric] = _metric_params.copy()
    else:
        _AVAILABLE_AMPLITUDE_METRICS[_metric] = \
            {name: descr for name, descr in _metric_params.items()
             if name != 'delta'}


def betti_curves(diagrams, sampling):
    born = sampling >= diagrams[:, :, 0]
    not_dead = sampling < diagrams[:, :, 1]
    alive = np.logical_and(born, not_dead)
    betti = np.sum(alive, axis=2).T
    return betti


def landscapes(diagrams, sampling, n_layers):
    n_points = diagrams.shape[1]
    midpoints = (diagrams[:, :, 1] + diagrams[:, :, 0]) / 2.
    heights = (diagrams[:, :, 1] - diagrams[:, :, 0]) / 2.
    fibers = np.maximum(-np.abs(sampling - midpoints) + heights, 0)
    top_pos = range(-min(n_layers, n_points), 0)
    fibers.partition(top_pos, axis=2)
    fibers = np.flip(fibers[:, :, -n_layers:], axis=2)
    fibers = np.transpose(fibers, (1, 2, 0))
    pad_with = ((0, 0), (0, max(0, n_layers - n_points)), (0, 0))
    fibers = np.pad(fibers, pad_with, "constant", constant_values=0)
    return fibers


def heats(diagrams, sampling, step_size, sigma):
    # WARNING: modifies `diagrams` in place
    heats_ = \
        np.zeros((len(diagrams), len(sampling), len(sampling)), dtype=float)
    # If the step size is zero, we return a trivial image
    if step_size == 0:
        return heats_

    # Set the values outside of the sampling range
    first_sampling, last_sampling = sampling[0, 0, 0], sampling[-1, 0, 0]
    diagrams[diagrams < first_sampling] = first_sampling
    diagrams[diagrams > last_sampling] = last_sampling

    # Calculate the value of `sigma` in pixel units, threshold for numerical
    # reasons if it's too large.
    sigma_pixel = sigma / step_size
    sigma_pixel = min(sigma_pixel, 10**5 * len(sampling))

    for i, diagram in enumerate(diagrams):
        nontrivial_points_idx = np.flatnonzero(diagram[:, 1] != diagram[:, 0])
        diagram_nontrivial_pixel_coords = np.array(
            (diagram - first_sampling) / step_size, dtype=int
            )[nontrivial_points_idx]
        image = heats_[i]
        _sample_image(image, diagram_nontrivial_pixel_coords)
        gaussian_filter(image, sigma_pixel, mode="constant", output=image)

    heats_ -= np.transpose(heats_, (0, 2, 1))
    heats_ /= (step_size ** 2)
    heats_ = np.rot90(heats_, k=1, axes=(1, 2))
    return heats_


def persistence_images(diagrams, sampling, step_size, sigma, weights):
    # For persistence images, `sampling` is a tall matrix with two columns
    # (the first for birth and the second for persistence), and `step_size` is
    # a 2d array
    # WARNING: modifies `diagrams` in place
    persistence_images_ = \
        np.zeros((len(diagrams), len(sampling), len(sampling)), dtype=float)
    # If either step size is zero, we return a trivial image
    if (step_size == 0).any():
        return persistence_images_

    # Transform diagrams from (birth, death, dim) to (birth, persistence, dim)
    diagrams[:, :, 1] -= diagrams[:, :, 0]

    sigma_pixel = []
    first_samplings = sampling[0]
    last_samplings = sampling[-1]
    for ax in [0, 1]:
        diagrams_ax = diagrams[:, :, ax]
        # Set the values outside of the sampling range
        diagrams_ax[diagrams_ax < first_samplings[ax]] = first_samplings[ax]
        diagrams_ax[diagrams_ax > last_samplings[ax]] = last_samplings[ax]
        # Calculate the value of the component of `sigma` in pixel units
        sigma_pixel.append(sigma / step_size[ax])

    # Sample the image, apply the weights, smoothen
    for i, diagram in enumerate(diagrams):
        nontrivial_points_idx = np.flatnonzero(diagram[:, 1])
        diagram_nontrivial_pixel_coords = np.array(
            (diagram - first_samplings) / step_size, dtype=int
            )[nontrivial_points_idx]
        image = persistence_images_[i]
        _sample_image(image, diagram_nontrivial_pixel_coords)
        image *= weights
        gaussian_filter(image, sigma_pixel, mode="constant", output=image)

    persistence_images_ = np.rot90(persistence_images_, k=1, axes=(1, 2))
    persistence_images_ /= np.product(step_size)
    return persistence_images_


def silhouettes(diagrams, sampling, power, **kwargs):
    """Input: a batch of persistence diagrams with a sampling (3d array
    returned by _bin) of a one-dimensional range.
    """
    sampling = np.transpose(sampling, axes=(1, 2, 0))
    weights = np.diff(diagrams, axis=2)
    if power > 8.:
        weights = weights / np.max(weights, axis=1, keepdims=True)
    weights = weights ** power
    total_weights = np.sum(weights, axis=1)
    # Next line is a trick to avoid NaNs when computing `fibers_weighted_sum`
    total_weights[total_weights == 0.] = np.inf
    midpoints = (diagrams[:, :, [1]] + diagrams[:, :, [0]]) / 2.
    heights = (diagrams[:, :, [1]] - diagrams[:, :, [0]]) / 2.
    fibers = np.maximum(-np.abs(sampling - midpoints) + heights, 0)
    fibers_weighted_sum = np.sum(weights * fibers, axis=1) / total_weights
    return fibers_weighted_sum


def bottleneck_distances(diagrams_1, diagrams_2, delta=0.01, **kwargs):
    return np.array([[bottleneck_distance(
        diagram_1[diagram_1[:, 0] != diagram_1[:, 1]],
        diagram_2[diagram_2[:, 0] != diagram_2[:, 1]],
        delta) for diagram_2 in diagrams_2] for diagram_1 in diagrams_1])


def wasserstein_distances(diagrams_1, diagrams_2, p=2, delta=0.01, **kwargs):
    return np.array([[wasserstein_distance(
        diagram_1[diagram_1[:, 0] != diagram_1[:, 1]],
        diagram_2[diagram_2[:, 0] != diagram_2[:, 1]],
        p, delta,) for diagram_2 in diagrams_2] for diagram_1 in diagrams_1])


def betti_distances(
        diagrams_1, diagrams_2, sampling, step_size, p=2., **kwargs
        ):
    step_size_factor = step_size ** (1 / p)
    are_arrays_equal = np.array_equal(diagrams_1, diagrams_2)
    betti_curves_1 = betti_curves(diagrams_1, sampling)
    if are_arrays_equal:
        distances = pdist(betti_curves_1, "minkowski", p=p)
        distances *= step_size_factor
        return squareform(distances)
    betti_curves_2 = betti_curves(diagrams_2, sampling)
    distances = cdist(betti_curves_1, betti_curves_2, "minkowski", p=p)
    distances *= step_size_factor
    return distances


def landscape_distances(
        diagrams_1, diagrams_2, sampling, step_size, p=2., n_layers=1,
        **kwargs
        ):
    step_size_factor = step_size ** (1 / p)
    n_samples_1, n_points_1 = diagrams_1.shape[:2]
    n_layers_1 = min(n_layers, n_points_1)
    if np.array_equal(diagrams_1, diagrams_2):
        ls_1 = landscapes(diagrams_1, sampling, n_layers_1).\
            reshape(n_samples_1, -1)
        distances = pdist(ls_1, "minkowski", p=p)
        distances *= step_size_factor
        return squareform(distances)
    n_samples_2, n_points_2 = diagrams_2.shape[:2]
    n_layers_2 = min(n_layers, n_points_2)
    n_layers = max(n_layers_1, n_layers_2)
    ls_1 = landscapes(diagrams_1, sampling, n_layers).\
        reshape(n_samples_1, -1)
    ls_2 = landscapes(diagrams_2, sampling, n_layers).\
        reshape(n_samples_2, -1)
    distances = cdist(ls_1, ls_2, "minkowski", p=p)
    distances *= step_size_factor
    return distances


def heat_distances(
        diagrams_1, diagrams_2, sampling, step_size, sigma=0.1, p=2., **kwargs
        ):
    # WARNING: `heats` modifies `diagrams` in place
    step_size_factor = step_size ** (2 / p)
    are_arrays_equal = np.array_equal(diagrams_1, diagrams_2)
    heats_1 = heats(diagrams_1, sampling, step_size, sigma).\
        reshape(len(diagrams_1), -1)
    if are_arrays_equal:
        distances = pdist(heats_1, "minkowski", p=p)
        distances *= step_size_factor
        return squareform(distances)
    heats_2 = heats(diagrams_2, sampling, step_size, sigma).\
        reshape(len(diagrams_2), -1)
    distances = cdist(heats_1, heats_2, "minkowski", p=p)
    distances *= step_size_factor
    return distances


def persistence_image_distances(
        diagrams_1, diagrams_2, sampling, step_size, sigma=0.1,
        weight_function=np.ones_like, p=2., **kwargs
        ):
    # For persistence images, `sampling` is a tall matrix with two columns
    # (the first for birth and the second for persistence), and `step_size` is
    # a 2d array
    weights = weight_function(sampling[:, 1])
    step_sizes_factor = np.product(step_size) ** (1 / p)
    # WARNING: `persistence_images` modifies `diagrams` in place
    are_arrays_equal = np.array_equal(diagrams_1, diagrams_2)
    persistence_images_1 = \
        persistence_images(diagrams_1, sampling, step_size, sigma, weights).\
        reshape(len(diagrams_1), -1)
    if are_arrays_equal:
        distances = pdist(persistence_images_1, "minkowski", p=p)
        distances *= step_sizes_factor
        return squareform(distances)
    persistence_images_2 = persistence_images(
        diagrams_2, sampling, step_size, sigma, weights
        ).reshape(len(diagrams_2), -1)
    distances = cdist(
        persistence_images_1, persistence_images_2, "minkowski", p=p
        )
    distances *= step_sizes_factor
    return distances


def silhouette_distances(
        diagrams_1, diagrams_2, sampling, step_size, power=1., p=2., **kwargs
        ):
    step_size_factor = step_size ** (1 / p)
    are_arrays_equal = np.array_equal(diagrams_1, diagrams_2)
    silhouettes_1 = silhouettes(diagrams_1, sampling, power)
    if are_arrays_equal:
        distances = pdist(silhouettes_1, 'minkowski', p=p)
        distances *= step_size_factor
        return squareform(distances)
    silhouettes_2 = silhouettes(diagrams_2, sampling, power)
    distances = cdist(silhouettes_1, silhouettes_2, 'minkowski', p=p)
    distances *= step_size_factor
    return distances


implemented_metric_recipes = {
    "bottleneck": bottleneck_distances,
    "wasserstein": wasserstein_distances,
    "landscape": landscape_distances,
    "betti": betti_distances,
    "heat": heat_distances,
    "persistence_image": persistence_image_distances,
    'silhouette': silhouette_distances
    }


def _parallel_pairwise(
        X1, X2, metric, metric_params, homology_dimensions, n_jobs
        ):
    metric_func = implemented_metric_recipes[metric]
    effective_metric_params = metric_params.copy()
    none_dict = {dim: None for dim in homology_dimensions}
    samplings = effective_metric_params.pop("samplings", none_dict)
    step_sizes = effective_metric_params.pop("step_sizes", none_dict)
    if metric in ["heat", "persistence_image"]:
        parallel_kwargs = {"mmap_mode": "c"}
    else:
        parallel_kwargs = {}

    n_columns = len(X2)
    distance_matrices = Parallel(n_jobs=n_jobs, **parallel_kwargs)(
        delayed(metric_func)(
            _subdiagrams(X1, [dim], remove_dim=True),
            _subdiagrams(X2[s], [dim], remove_dim=True),
            sampling=samplings[dim],
            step_size=step_sizes[dim],
            **effective_metric_params
            )
        for dim in homology_dimensions
        for s in gen_even_slices(n_columns, effective_n_jobs(n_jobs))
        )

    distance_matrices = np.concatenate(distance_matrices, axis=1)
    distance_matrices = np.stack(
        [distance_matrices[:, i * n_columns:(i + 1) * n_columns]
         for i in range(len(homology_dimensions))],
        axis=2)
    return distance_matrices


def bottleneck_amplitudes(diagrams, **kwargs):
    half_lifetimes = (diagrams[:, :, 1] - diagrams[:, :, 0]) / 2.
    return np.linalg.norm(half_lifetimes, axis=1, ord=np.inf)


def wasserstein_amplitudes(diagrams, p=2., **kwargs):
    half_lifetimes = (diagrams[:, :, 1] - diagrams[:, :, 0]) / 2.
    return np.linalg.norm(half_lifetimes, axis=1, ord=p)


def betti_amplitudes(diagrams, sampling, step_size, p=2., **kwargs):
    step_size_factor = step_size ** (1 / p)
    bcs = betti_curves(diagrams, sampling)
    amplitudes = np.linalg.norm(bcs, axis=1, ord=p)
    amplitudes *= step_size_factor
    return amplitudes


def landscape_amplitudes(
        diagrams, sampling, step_size, p=2., n_layers=1, **kwargs
        ):
    step_size_factor = step_size ** (1 / p)
    ls = landscapes(diagrams, sampling, n_layers).\
        reshape(len(diagrams), -1)
    amplitudes = np.linalg.norm(ls, axis=1, ord=p)
    amplitudes *= step_size_factor
    return amplitudes


def heat_amplitudes(diagrams, sampling, step_size, sigma=0.1, p=2., **kwargs):
    # WARNING: `heats` modifies `diagrams` in place
    step_size_factor = step_size ** (2 / p)
    heats_ = heats(diagrams, sampling, step_size, sigma).\
        reshape(len(diagrams), -1)
    amplitudes = np.linalg.norm(heats_, axis=1, ord=p)
    amplitudes *= step_size_factor
    return amplitudes


def persistence_image_amplitudes(
        diagrams, sampling, step_size, sigma=0.1, weight_function=np.ones_like,
        p=2., **kwargs
        ):
    # For persistence images, `sampling` is a tall matrix with two columns
    # (the first for birth and the second for persistence), and `step_size` is
    # a 2d array
    weights = weight_function(sampling[:, 1])
    step_sizes_factor = np.product(step_size) ** (1 / p)
    # WARNING: `persistence_images` modifies `diagrams` in place
    persistence_images_ = persistence_images(
        diagrams, sampling, step_size, sigma, weights
        ).reshape(len(diagrams), -1)
    amplitudes = np.linalg.norm(persistence_images_, axis=1, ord=p)
    amplitudes *= step_sizes_factor
    return amplitudes


def silhouette_amplitudes(
        diagrams, sampling, step_size, power=1., p=2., **kwargs
        ):
    step_size_factor = step_size ** (1 / p)
    silhouettes_ = silhouettes(diagrams, sampling, power)
    amplitudes = np.linalg.norm(silhouettes_, axis=1, ord=p)
    amplitudes *= step_size_factor
    return amplitudes


implemented_amplitude_recipes = {
    "bottleneck": bottleneck_amplitudes,
    "wasserstein": wasserstein_amplitudes,
    "landscape": landscape_amplitudes,
    "betti": betti_amplitudes,
    "heat": heat_amplitudes,
    "persistence_image": persistence_image_amplitudes,
    'silhouette': silhouette_amplitudes
    }


def _parallel_amplitude(X, metric, metric_params, homology_dimensions, n_jobs):
    amplitude_func = implemented_amplitude_recipes[metric]
    effective_metric_params = metric_params.copy()
    none_dict = {dim: None for dim in homology_dimensions}
    samplings = effective_metric_params.pop("samplings", none_dict)
    step_sizes = effective_metric_params.pop("step_sizes", none_dict)
    if metric in ["heat", "persistence_image"]:
        parallel_kwargs = {"mmap_mode": "c"}
    else:
        parallel_kwargs = {}

    amplitude_arrays = Parallel(n_jobs=n_jobs, **parallel_kwargs)(
        delayed(amplitude_func)(
            _subdiagrams(X[s], [dim], remove_dim=True),
            sampling=samplings[dim],
            step_size=step_sizes[dim],
            **effective_metric_params
            )
        for dim in homology_dimensions
        for s in gen_even_slices(_num_samples(X), effective_n_jobs(n_jobs))
        )

    amplitude_arrays = np.concatenate(amplitude_arrays).\
        reshape(len(homology_dimensions), len(X)).T

    return amplitude_arrays
