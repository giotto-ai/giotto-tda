"""Utility functions for diagrams."""
# License: GNU AGPLv3

import numpy as np


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


def _sample_image(image, sampled_diag):
    unique, counts = np.unique(sampled_diag, axis=0, return_counts=True)
    unique = tuple(tuple(row) for row in unique.astype(np.int).T)
    image[unique] = counts


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
        mask = (Xdim[:, :, 1] - Xdim[:, :, 0]) <= cutoff
        Xdim[mask, :] = [min_value, min_value, dim]
        max_points = np.max(np.sum(Xs[:, :, 1] != 0, axis=1))
        Xdim = Xdim[:, :max_points, :]
        Xf = np.concatenate([Xf, Xdim], axis=1)
    return Xf


def _bin(X, metric, n_bins=100, **kw_args):
    homology_dimensions = sorted(list(set(X[0, :, 2])))
    # For some vectorizations, we force the values to be the same + widest
    sub_diags = {dim: _subdiagrams(X, [dim], remove_dim=True)
                 for dim in homology_dimensions}
    # For persistence images, move into birth-persistence
    if metric == 'persistence_image':
        for dim in homology_dimensions:
            sub_diags[dim][:, :, [1]] = sub_diags[dim][:, :, [1]] \
                - sub_diags[dim][:, :, [0]]
    min_vals = {dim: np.min(sub_diags[dim], axis=(0, 1))
                for dim in homology_dimensions}
    max_vals = {dim: np.max(sub_diags[dim], axis=(0, 1))
                for dim in homology_dimensions}

    if metric in ['landscape', 'betti', 'heat', 'silhouette']:
        #  Taking the min(resp. max) of a tuple `m` amounts to extracting
        #  the birth (resp. death) value
        min_vals = {d: np.array(2*[np.min(m)]) for d, m in min_vals.items()}
        max_vals = {d: np.array(2*[np.max(m)]) for d, m in max_vals.items()}

    # Scales between axes should be kept the same, but not between dimension
    all_max_values = np.stack(list(max_vals.values()))
    if len(homology_dimensions) == 1:
        all_max_values = all_max_values.reshape(1, -1)
    global_max_val = np.max(all_max_values, axis=0)
    max_vals = {dim: np.array([max_vals[dim][k] if
                               (max_vals[dim][k] != min_vals[dim][k])
                               else global_max_val[k] for k in range(2)])
                for dim in homology_dimensions}

    samplings = {}
    step_sizes = {}
    for dim in homology_dimensions:
        samplings[dim], step_sizes[dim] = np.linspace(min_vals[dim],
                                                      max_vals[dim],
                                                      retstep=True,
                                                      num=n_bins)
    if metric in ['landscape', 'betti', 'heat', 'silhouette']:
        for dim in homology_dimensions:
            samplings[dim] = samplings[dim][:, [0], None]
            step_sizes[dim] = step_sizes[dim][0]
    return samplings, step_sizes


def _calculate_weights(X, weight_function, samplings, **kw_args):
    weights = {dim: weight_function(samplings[dim][:, 1])
               for dim in samplings.keys()}
    return weights
