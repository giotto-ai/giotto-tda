# License: GNU AGPLv3

from numbers import Real
from types import FunctionType

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
        'delta': {'type': Real, 'in': Interval(0, 1, closed='both')}},
    'wasserstein': {
        'p': {'type': Real, 'in': Interval(1, np.inf, closed='left')},
        'delta': {'type': Real, 'in': Interval(0, 1, closed='right')}},
    'betti': {
        'p': {'type': Real, 'in': Interval(1, np.inf, closed='both')},
        'n_bins': {'type': int, 'in': Interval(1, np.inf, closed='left')}},
    'landscape': {
        'p': {'type': Real, 'in': Interval(1, np.inf, closed='both')},
        'n_bins': {'type': int, 'in': Interval(1, np.inf, closed='left')},
        'n_layers': {'type': int, 'in': Interval(1, np.inf, closed='left')}},
    'heat': {
        'p': {'type': Real, 'in': Interval(1, np.inf, closed='both')},
        'n_bins': {'type': int, 'in': Interval(1, np.inf, closed='left')},
        'sigma': {'type': Real, 'in': Interval(0, np.inf, closed='neither')}},
    'persistence_image': {
        'p': {'type': Real, 'in': Interval(1, np.inf, closed='both')},
        'n_bins': {'type': int, 'in': Interval(1, np.inf, closed='left')},
        'sigma': {'type': Real, 'in': Interval(0, np.inf, closed='neither')},
        'weight_function': {'type': FunctionType, 'in': None}},
    'silhouette': {
        'power': {'type': Real, 'in': Interval(0, np.inf, closed='right')},
        'p': {'type': Real, 'in': Interval(1, np.inf, closed='both')},
        'n_bins': {'type': int, 'in': Interval(1, np.inf, closed='left')}}}

_AVAILABLE_AMPLITUDE_METRICS = dict()
for metric, metric_params in _AVAILABLE_METRICS.items():
    if metric not in ['bottleneck', 'wasserstein']:
        _AVAILABLE_AMPLITUDE_METRICS[metric] = metric_params.copy()
    else:
        _AVAILABLE_AMPLITUDE_METRICS[metric] = \
            {name: descr for name, descr in metric_params.items()
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


def _heat(image, sampled_diag, sigma):
    _sample_image(image, sampled_diag)  # modifies `heat` inplace
    image[:] = gaussian_filter(image, sigma, mode="reflect")


def heats(diagrams, sampling, step_size, sigma):
    heats_ = np.zeros((diagrams.shape[0],
                       sampling.shape[0], sampling.shape[0]))

    diagrams[diagrams < sampling[0, 0]] = sampling[0, 0]
    diagrams[diagrams > sampling[-1, 0]] = sampling[-1, 0]
    diagrams = np.array((diagrams - sampling[0, 0]) / step_size, dtype=int)

    [_heat(heats_[i], sampled_diag, sigma)
        for i, sampled_diag in enumerate(diagrams)]

    heats_ = heats_ - np.transpose(heats_, (0, 2, 1))
    heats_ = np.rot90(heats_, k=1, axes=(1, 2))
    return heats_


def persistence_images(diagrams, sampling, step_size, weights, sigma):
    persistence_images_ = np.zeros(
        (diagrams.shape[0], sampling.shape[0], sampling.shape[0]))
    # Transform diagrams from (birth, death, dim) to (birth, persistence, dim)
    diagrams[:, :, 1] = diagrams[:, :, 1] - diagrams[:, :, 0]

    for axis in [0, 1]:
        # Set the values outside of the sampling range to the sampling range.
        diagrams[:, :, axis][diagrams[:, :, axis] < sampling[0, axis]] = \
            sampling[0, axis]
        diagrams[:, :, axis][diagrams[:, :, axis] > sampling[-1, axis]] = \
            sampling[-1, axis]
        # Convert into pixel
        diagrams[:, :, axis] = np.array(
            (diagrams[:, :, axis] - sampling[0, axis]) / step_size[axis],
            dtype=int)
    # Sample the image
    [_sample_image(persistence_images_[i], sampled_diag)
     for i, sampled_diag in enumerate(diagrams)]

    # Apply the weights
    persistence_images_ *= weights / np.max(weights)

    # Smoothen the weighted-image
    for i, image in enumerate(persistence_images_):
        persistence_images_[i] = gaussian_filter(image, sigma, mode="reflect")

    persistence_images_ = np.rot90(persistence_images_, k=1, axes=(1, 2))
    return persistence_images_


def silhouettes(diagrams, sampling, power, **kwargs):
    """Input: a batch of persistence diagrams with a sampling (3d array
    returned by _bin) of a one-dimensional range.
    """
    sampling = np.transpose(sampling, axes=(1, 2, 0))
    weights = np.diff(diagrams, axis=2)[:, :, [0]]
    if power > 8.:
        weights = weights/np.max(weights, axis=1, keepdims=True)
    weights = weights**power
    total_weights = np.sum(weights, axis=1)
    midpoints = (diagrams[:, :, [1]] + diagrams[:, :, [0]]) / 2.
    heights = (diagrams[:, :, [1]] - diagrams[:, :, [0]]) / 2.
    fibers = np.maximum(-np.abs(sampling - midpoints) + heights, 0)
    fibers_weighted_sum = np.sum(weights*fibers, axis=1)/total_weights
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


def betti_distances(diagrams_1, diagrams_2, sampling,
                    step_size, p=2., **kwargs):
    betti_curves_1 = betti_curves(diagrams_1, sampling)
    if np.array_equal(diagrams_1, diagrams_2):
        unnorm_dist = squareform(pdist(betti_curves_1, "minkowski", p=p))
        return (step_size ** (1 / p)) * unnorm_dist
    betti_curves_2 = betti_curves(diagrams_2, sampling)
    unnorm_dist = cdist(betti_curves_1, betti_curves_2, "minkowski", p=p)
    return (step_size ** (1 / p)) * unnorm_dist


def landscape_distances(diagrams_1, diagrams_2, sampling, step_size,
                        p=2., n_layers=1, **kwargs):
    n_samples_1, n_points_1 = diagrams_1.shape[:2]
    n_layers_1 = min(n_layers, n_points_1)
    if np.array_equal(diagrams_1, diagrams_2):
        ls_1 = landscapes(diagrams_1, sampling,
                          n_layers_1).reshape(n_samples_1, -1)
        unnorm_dist = squareform(pdist(ls_1, "minkowski", p=p))
        return (step_size ** (1 / p)) * unnorm_dist
    n_samples_2, n_points_2 = diagrams_2.shape[:2]
    n_layers_2 = min(n_layers, n_points_2)
    n_layers = max(n_layers_1, n_layers_2)
    ls_1 = landscapes(diagrams_1, sampling,
                      n_layers).reshape(n_samples_1, -1)
    ls_2 = landscapes(diagrams_2, sampling,
                      n_layers).reshape(n_samples_2, -1)
    unnorm_dist = cdist(ls_1, ls_2, "minkowski", p=p)
    return (step_size ** (1 / p)) * unnorm_dist


def heat_distances(diagrams_1, diagrams_2, sampling, step_size,
                   sigma=1., p=2., **kwargs):
    heat_1 = heats(diagrams_1, sampling, step_size, sigma).reshape(
        diagrams_1.shape[0], -1)
    if np.array_equal(diagrams_1, diagrams_2):
        unnorm_dist = squareform(pdist(heat_1, "minkowski", p=p))
        return (step_size ** (1 / p)) * unnorm_dist
    heat_2 = heats(diagrams_2, sampling, step_size, sigma).\
        reshape(diagrams_2.shape[0], -1)
    unnorm_dist = cdist(heat_1, heat_2, "minkowski", p=p)
    return (step_size ** (1 / p)) * unnorm_dist


def persistence_image_distances(diagrams_1, diagrams_2, sampling, step_size,
                                weight_function=lambda x: x, sigma=1., p=2.,
                                **kwargs):
    sampling_ = np.copy(sampling.reshape((-1,)))
    weights = weight_function(sampling_ - sampling_[0])
    persistence_image_1 = persistence_images(diagrams_1, sampling_, step_size,
                                             weights, sigma).reshape(
                                                 diagrams_1.shape[0], -1)
    if np.array_equal(diagrams_1, diagrams_2):
        unnorm_dist = squareform(pdist(persistence_image_1, "minkowski", p=p))
        return (step_size ** (1 / p)) * unnorm_dist
    persistence_image_2 = persistence_images(diagrams_2, sampling_, step_size,
                                             weights, sigma,).reshape(
                                                 diagrams_2.shape[0], -1)
    unnorm_dist = cdist(persistence_image_1, persistence_image_2,
                        "minkowski", p=p)
    return (step_size ** (1 / p)) * unnorm_dist


def silhouette_distances(diagrams_1, diagrams_2, sampling, step_size,
                         power=2., p=2., **kwargs):
    silhouette_1 = silhouettes(diagrams_1, sampling, power)
    if np.array_equal(diagrams_1, diagrams_2):
        unnorm_dist = squareform(pdist(silhouette_1, 'minkowski', p=p))
    else:
        silhouette_2 = silhouettes(diagrams_2, sampling, power)
        unnorm_dist = cdist(silhouette_1, silhouette_2, 'minkowski', p=p)
    return (step_size ** (1 / p)) * unnorm_dist


implemented_metric_recipes = {
    "bottleneck": bottleneck_distances,
    "wasserstein": wasserstein_distances,
    "landscape": landscape_distances,
    "betti": betti_distances,
    "heat": heat_distances,
    "persistence_image": persistence_image_distances,
    'silhouette': silhouette_distances,
}


def _matrix_wrapper(distance_func, distance_matrices, slice_, dim,
                    *args, **kwargs):
    distance_matrices[:, slice_, int(dim)] = distance_func(*args, **kwargs)


def _parallel_pairwise(X1, X2, metric, metric_params,
                       homology_dimensions, n_jobs):
    metric_func = implemented_metric_recipes[metric]
    effective_metric_params = metric_params.copy()
    none_dict = {dim: None for dim in homology_dimensions}
    samplings = effective_metric_params.pop("samplings", none_dict)
    step_sizes = effective_metric_params.pop("step_sizes", none_dict)

    if X2 is None:
        X2 = X1

    distance_matrices = Parallel(n_jobs=n_jobs)(
        delayed(metric_func)(_subdiagrams(X1, [dim], remove_dim=True),
                             _subdiagrams(X2[s], [dim], remove_dim=True),
                             sampling=samplings[dim],
                             step_size=step_sizes[dim],
                             **effective_metric_params)
        for dim in homology_dimensions
        for s in gen_even_slices(X2.shape[0], effective_n_jobs(n_jobs)))

    distance_matrices = np.concatenate(distance_matrices, axis=1)
    distance_matrices = np.stack(
        [distance_matrices[:, i * X2.shape[0]:(i + 1) * X2.shape[0]]
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
    bcs = betti_curves(diagrams, sampling)
    return (step_size ** (1 / p)) * np.linalg.norm(bcs, axis=1, ord=p)


def landscape_amplitudes(diagrams, sampling, step_size, p=2., n_layers=1,
                         **kwargs):
    ls = landscapes(diagrams, sampling, n_layers).\
        reshape(len(diagrams), -1)
    return (step_size ** (1 / p)) * np.linalg.norm(ls, axis=1, ord=p)


def heat_amplitudes(diagrams, sampling, step_size, sigma=1., p=2., **kwargs):
    heat = heats(diagrams, sampling, step_size, sigma)
    return np.linalg.norm(heat, axis=(1, 2), ord=p)


def persistence_image_amplitudes(diagrams, sampling, step_size,
                                 weight_function=lambda x: x, sigma=1., p=2.,
                                 **kwargs):
    persistence_image = persistence_images(diagrams, sampling, step_size,
                                           weight_function, sigma)
    return np.linalg.norm(persistence_image, axis=(1, 2), ord=p)


def silhouette_amplitudes(diagrams, sampling, step_size, power=2., p=2.,
                          **kwargs):
    sht = silhouettes(diagrams, sampling, power)
    return (step_size ** (1 / p)) * np.linalg.norm(sht, axis=1, ord=p)


implemented_amplitude_recipes = {
    "bottleneck": bottleneck_amplitudes,
    "wasserstein": wasserstein_amplitudes,
    "landscape": landscape_amplitudes,
    "betti": betti_amplitudes,
    "heat": heat_amplitudes,
    "persistence_image": persistence_images,
    'silhouette': silhouette_amplitudes,
}


def _arrays_wrapper(amplitude_func, amplitude_arrays, slice_, dim,
                    *args, **kwargs):
    amplitude_arrays[slice_, int(dim)] = amplitude_func(*args, **kwargs)


def _parallel_amplitude(X, metric, metric_params, homology_dimensions, n_jobs):
    amplitude_func = implemented_amplitude_recipes[metric]
    effective_metric_params = metric_params.copy()
    none_dict = {dim: None for dim in homology_dimensions}
    samplings = effective_metric_params.pop("samplings", none_dict)
    step_sizes = effective_metric_params.pop("step_sizes", none_dict)

    amplitude_arrays = Parallel(n_jobs=n_jobs)(
        delayed(amplitude_func)(
            _subdiagrams(X, [dim], remove_dim=True)[s],
            sampling=samplings[dim], step_size=step_sizes[dim],
            **effective_metric_params)
        for dim in homology_dimensions
        for s in gen_even_slices(_num_samples(X), effective_n_jobs(n_jobs)))

    amplitude_arrays = (np.concatenate(amplitude_arrays).reshape(
        len(homology_dimensions), X.shape[0]).T)

    return amplitude_arrays
