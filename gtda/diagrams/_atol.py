from numbers import Real

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans, MiniBatchKMeans
from gtda.utils.intervals import Interval


_AVAILABLE_QUANTISERS = {
    'KMeans': {
        'n_clusters': {'type': int, 'in': Interval(1, np.inf, closed='left')},
        'init': {'type': str, 'in': ['k-means++', 'random']},
        'n_init': {'type': int, 'in': Interval(1, np.inf, closed='left')},
        'max_iter': {'type': int, 'in': Interval(1, np.inf, closed='left')},
        'tol': {'type': Real, 'in': Interval(0, 1, closed='both')},
        'precompute_distances': {'type': (str, bool), 'in': ['auto']},
        'verbose': {'type': int},
        'random_state': {'type': (int, type(None), np.random.RandomState),
                         'in': Interval(0, np.inf, closed='both')},
        'copy_x': {'type': bool},
        'algorithm': {'type': str, 'in': ['auto', 'full', 'elkan']},
    },
    'MiniBatchKMeans': {
        'n_clusters': {'type': int, 'in': Interval(1, np.inf, closed='left')},
        'init': {'type': str, 'in': ['k-means++', 'random']},
        'max_iter': {'type': int, 'in': Interval(1, np.inf, closed='left')},
        'batch_size': {'type': int, 'in': Interval(1, np.inf, closed='left')},
        'verbose': {'type': int},
        'compute_labels': {'type': bool},
        'random_state': {'type': (int, type(None), np.random.RandomState),
                         'in': Interval(0, np.inf, closed='both')},
        'tol': {'type': Real, 'in': Interval(0, 1, closed='both')},
        'max_no_improvement': {'type': int,
                               'in': Interval(1, np.inf, closed='left')},
        'init_size': {'type': (int, type(None)),
                      'in': Interval(1, np.inf, closed='left')},
        'n_init': {'type': int, 'in': Interval(1, np.inf, closed='left')},
        'reassignment_ratio': {'type': Real,
                               'in': Interval(0, 1, closed='both')},
    },
}


implemented_quantiser_recipes = {
    'KMeans': KMeans,
    'MiniBatchKMeans': MiniBatchKMeans,
}


_AVAILABLE_CONTRAST_FUNCTIONS = {
    'gaussian': {},
    'laplacian': {},
    'indicator': {},
}


def gaussian_contrast(measure, centers, inertias):
    return np.exp(
        - pairwise_distances(measure, Y=centers, squared=True) / inertias**2
    )


def laplacian_contrast(measure, centers, inertias):
    return np.exp(
        - pairwise_distances(measure, Y=centers) / inertias
    )


def indicator_contrast(measure, centers, inertias):
    return np.clip(
        2 - pairwise_distances(measure, Y=centers) / inertias, 0, 1
    )


implemented_contrast_recipes = {
    'gaussian': gaussian_contrast,
    'laplacian': laplacian_contrast,
    'indicator': indicator_contrast,
}

_AVAILABLE_WEIGHT_FUNCTIONS = {
    'one': {},
    'uniform': {},
}


def one_weight(measure):
    return np.ones(shape=measure.shape[0])


def uniform_weight(measure):
    return np.ones(shape=measure.shape[0]) / measure.shape[0]


implemented_weight_recipes = {
    'one': one_weight,
    'uniform': uniform_weight,
}
