"""Utility functions for diagrams."""
# License: GNU AGPLv3

import numpy as np


def _rotate_clockwise(X):
    rot_mat = (np.sqrt(2) / 2.) * np.array([[1, -1, 0], [1, 1, 0], [0, 0, 1]])
    return np.dot(X, rot_mat)


def _rotate_anticlockwise(X):
    rot_mat = (np.sqrt(2) / 2.) * np.array([[1, 1, 0], [-1, 1, 0], [0, 0, 1]])
    return np.dot(X, rot_mat)


def _subdiagrams(X, homology_dimensions, remove_dim=False):
    for dim in homology_dimensions:
        Xs = X[X[:, :, 2] == dim]
        Xs = Xs.reshape(X.shape[0], -1, 3)
    if remove_dim:
        Xs = Xs[:, :, :2]
    return Xs


def _pad(X, max_diagram_sizes):
    X_padded = {dim: np.pad(
        X[dim],
        ((0, 0), (0, max_diagram_sizes[dim] - X[dim].shape[1]),
         (0, 0)), 'constant') for dim in X.keys()}
    return X_padded


def _sort(Xs):
    indices = np.argsort(Xs[:, :, 1] - Xs[:, :, 0], axis=1)
    indices = np.stack([indices, indices, indices], axis=2)
    Xs = np.flip(np.take_along_axis(Xs, indices, axis=1), axis=1)
    return Xs


def _filter(Xs, filtered_homology_dimensions, cutoff):
    homology_dimensions = sorted(list(set(Xs[0, :, 2])))
    unfiltered_homology_dimensions = sorted(list(
        set(homology_dimensions) - set(filtered_homology_dimensions)))

    if len(unfiltered_homology_dimensions) == 0:
        Xf = np.empty((Xs.shape[0], 0, 3), dtype=Xs.dtype)
    else:
        Xf = _subdiagrams(Xs, unfiltered_homology_dimensions)

    for dim in filtered_homology_dimensions:
        Xdim = _subdiagrams(Xs, [dim])
        min_value = np.min(Xdim[:, :, 0])
        mask = np.sqrt(2) / 2. * (Xdim[:, :, 1] - Xdim[:, :, 0]) <= cutoff
        Xdim[mask, :] = [min_value, min_value, dim]
        max_points = np.max(np.sum(Xs[:, :, 1] != 0, axis=1))
        Xdim = Xdim[:, :max_points, :]
        Xf = np.concatenate([Xf, Xdim], axis=1)
    return Xf


def _discretize(X, n_values=100, **kw_args):
    homology_dimensions = sorted(list(set(X[0, :, 2])))

    min_vals = {dim: np.min(_subdiagrams(X, [dim], remove_dim=True)[:, :, 0])
                for dim in homology_dimensions}
    max_vals = {dim: np.max(_subdiagrams(X, [dim], remove_dim=True)[:, :, 1])
                for dim in homology_dimensions}
    global_max_val = max(list(max_vals.values()))
    max_vals = {
        dim: max_vals[dim] if
        (max_vals[dim] != min_vals[dim]) else
        global_max_val for dim in homology_dimensions}

    samplings = {}
    step_sizes = {}
    for dim in homology_dimensions:
        samplings[dim], step_sizes[dim] = np.linspace(min_vals[dim],
                                                      max_vals[dim],
                                                      retstep=True,
                                                      num=n_values)
        samplings[dim] = samplings[dim][:, None, None]
    return samplings, step_sizes
