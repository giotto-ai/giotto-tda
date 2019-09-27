# Authors: Guillaume Tauzin <guillaume.tauzin@epfl.ch>
#          Umberto Lupo <u.lupo@l2f.ch>
# License: TBD

import numpy as np


def _rotate_clockwise(X):
    rot_mat = (np.sqrt(2) / 2.) * np.array([[1, -1], [1, 1]])
    return np.dot(X, rot_mat)


def _rotate_anticlockwise(X):
    rot_mat = (np.sqrt(2) / 2.) * np.array([[1, 1], [-1, 1]])
    return np.dot(X, rot_mat)


def _pad(X, max_betti_numbers):
    X_padded = {dim: np.pad(
        X[dim],
        ((0, 0), (0, max_betti_numbers[dim] - X[dim].shape[1]),
        (0, 0)), 'constant') for dim in X.keys()}
    return X_padded


def _sort(X_scaled, homology_dimensions):
    indices = {dim: np.argsort(X_scaled[dim][:, :, 1] -
                                     X_scaled[dim][:, :, 0], axis=1)
               for dim in homology_dimensions}
    indices = {
        dim: np.stack([indices[dim], indices[dim]],
                            axis=2) for dim in homology_dimensions}
    X_sorted = {
        dim: np.flip(np.take_along_axis(X_scaled[dim],
                                              indices[dim],
                                              axis=1), axis=1)
        for dim in homology_dimensions}
    return {**X_sorted, **{dim: X_scaled[dim]
                          for dim in set(X_scaled.keys()) -
                          homology_dimensions}}


def _filter(X_scaled, homology_dimensions, cutoff):
    X_filtered = {dim: X_scaled[dim].copy()
                 for dim in homology_dimensions}
    mask = {dim: (np.sqrt(2) / 2.) * (X[:, :, 1] - X[:, :, 0]) <=
            cutoff for dim, X in X_filtered.items()}

    for dim, X in X_filtered.items():
        X[mask[dim], :] = [0, 0]

    max_points = {
        dim: np.max(np.sum(X[:, :, 1] != 0,
                                 axis=1)) for dim, X in X_filtered.items()
        }
    X_filtered = {dim: X[:, :max_points[dim], :] for dim,
                 X in X_filtered.items()}
    return {**X_filtered, **{dim: X_scaled[dim]
                            for dim in set(X_scaled.keys()) -
                            homology_dimensions}}


def _create_linspaces(X, n_sampled_values=100, **kw_args):
    min_vals = {dim: np.min(X[dim][:, :, 0]) if X[dim].size else np.inf
                for dim in X.keys()}
    glob_min_val = min(list(min_vals.values()))
    min_vals = {dim: min_vals[dim]
                     if (min_vals[dim] != np.inf) else glob_min_val
                for dim in X.keys()}

    max_vals = {dim: np.max(X[dim][:, :, 1]) if X[dim].size else -np.inf
                for dim in X.keys()}
    glob_max_val = max(list(max_vals.values()))
    max_vals = {dim: max_vals[dim]
                     if (max_vals[dim] != -np.inf) else glob_max_val
                for dim in X.keys()}
    max_vals = {dim: max_vals[dim]
                     if (max_vals[dim] != min_vals[dim]) else glob_max_val
                for dim in X.keys()}

    num_segments = n_sampled_values + 1
    step_sizes = {dim: (max_vals[dim] - min_vals[dim])/num_segments
                  for dim in X.keys()}
    linspaces = {dim: np.linspace(min_vals[dim], max_vals[dim],
                                  num=num_segments, endpoint=False)[1:]
                 for dim in X.keys()}

    return linspaces, step_sizes
