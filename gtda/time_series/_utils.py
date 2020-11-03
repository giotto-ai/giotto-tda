"""Utility functions for time series processing."""
# License: GNU AGPLv3

from functools import partial

import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.neighbors import NearestNeighbors


def _time_delay_embedding(X, time_delay=1, dimension=2, stride=1,
                          flatten=False, ensure_last_value=True):
    if hasattr(X, 'shape') and hasattr(X, 'ndim'):  # ndarray input
        n_timestamps = X.shape[-1]
        n_points, offset = \
            divmod(n_timestamps - time_delay * (dimension - 1) - 1, stride)
        n_points += 1
        if n_points <= 0:
            raise ValueError(
                f"Not enough time stamps ({n_timestamps}) to produce at least "
                f"one {dimension}-dimensional vector under the current choice "
                f"of time delay ({time_delay})."
                )
        indices = np.tile(np.arange(0, time_delay * dimension, time_delay),
                          (n_points, 1))
        indices += np.arange(n_points)[:, None] * stride
        if ensure_last_value:
            indices += offset

        X_embedded = X[..., indices]
        if flatten and (X.ndim > 2):
            transpose_axes = (0, *range(1, X.ndim)[::-1], X.ndim)
            X_embedded = np.transpose(X_embedded, axes=transpose_axes).\
                reshape(len(X), -1, dimension * np.prod(X.shape[1:-1]))
    else:  # list of ndarray input
        func = partial(_time_delay_embedding, time_delay=time_delay,
                       dimension=dimension, stride=stride, flatten=flatten,
                       ensure_last_value=ensure_last_value)
        X_embedded = [func(x[None, ...])[0] for x in X]

    return X_embedded


def _mutual_information(X, time_delay, n_bins):
    """Calculate the mutual information given the time delay."""
    contingency = np.histogram2d(X[:-time_delay], X[time_delay:],
                                 bins=n_bins)[0]
    mutual_information = mutual_info_score(None, None,
                                           contingency=contingency)
    return mutual_information


def _false_nearest_neighbors(X, time_delay, dimension, stride=1):
    """Calculate the number of false nearest neighbours in a certain
    embedding dimension, based on heuristics."""
    X_embedded = _time_delay_embedding(X, time_delay=time_delay,
                                       dimension=dimension, stride=stride)

    neighbor = \
        NearestNeighbors(n_neighbors=2, algorithm='auto').fit(X_embedded)
    distances, indices = neighbor.kneighbors(X_embedded)
    distance = distances[:, 1]
    X_first_nbhrs = X[indices[:, 1]]

    epsilon = 2. * np.std(X)
    tolerance = 10

    neg_dim_delay = - dimension * time_delay
    distance_slice = distance[:neg_dim_delay]
    X_rolled = np.roll(X, neg_dim_delay)
    X_rolled_slice = slice(len(X) - len(X_embedded), neg_dim_delay)
    X_first_nbhrs_rolled = np.roll(X_first_nbhrs, neg_dim_delay)

    neighbor_abs_diff = np.abs(
        X_rolled[X_rolled_slice] - X_first_nbhrs_rolled[:neg_dim_delay]
        )

    false_neighbor_ratio = np.divide(
        neighbor_abs_diff, distance_slice,
        out=np.zeros_like(neighbor_abs_diff, dtype=float),
        where=(distance_slice != 0)
        )
    false_neighbor_criteria = false_neighbor_ratio > tolerance

    limited_dataset_criteria = distance_slice < epsilon

    n_false_neighbors = \
        np.sum(false_neighbor_criteria * limited_dataset_criteria)
    return n_false_neighbors
