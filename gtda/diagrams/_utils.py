"""Utility functions for diagrams."""
# License: GNU AGPLv3

from functools import reduce
from operator import iconcat

import numpy as np


def _subdiagrams(X, homology_dimensions, remove_dim=False):
    """For each diagram in a collection, extract the subdiagrams in a given
    list of homology dimensions. It is assumed that all diagrams in X contain
    the same number of points in each homology dimension."""
    n = len(X)
    if len(homology_dimensions) == 1:
        Xs = X[X[:, :, 2] == homology_dimensions[0]].reshape(n, -1, 3)
    else:
        Xs = np.concatenate([X[X[:, :, 2] == dim].reshape(n, -1, 3)
                             for dim in homology_dimensions],
                            axis=1)
    if remove_dim:
        Xs = Xs[:, :, :2]
    return Xs


def _pad(X, max_diagram_sizes):
    X_padded = {dim: np.pad(
        X[dim],
        ((0, 0), (0, max_diagram_sizes[dim] - X[dim].shape[1]),
         (0, 0)), 'constant') for dim in X.keys()}
    return X_padded


def _sample_image(image, sampled_diag):
    # NOTE: Modifies `image` in-place
    unique, counts = np.unique(sampled_diag, axis=0, return_counts=True)
    unique = tuple(tuple(row) for row in unique.astype(np.int).T)
    image[unique] = counts


def _filter(X, filtered_homology_dimensions, cutoff):
    n = len(X)
    homology_dimensions = sorted(list(set(X[0, :, 2])))
    unfiltered_homology_dimensions = [dim for dim in homology_dimensions if
                                      dim not in filtered_homology_dimensions]

    if len(unfiltered_homology_dimensions) == 0:
        Xuf = np.empty((n, 0, 3), dtype=X.dtype)
    else:
        Xuf = _subdiagrams(X, unfiltered_homology_dimensions)

    cutoff_mask = X[:, :, 1] - X[:, :, 0] > cutoff
    Xf = []
    for dim in filtered_homology_dimensions:
        dim_mask = X[:, :, 2] == dim
        indices = np.nonzero(np.logical_and(dim_mask, cutoff_mask))
        if not indices[0].size:
            Xdim = np.tile([0., 0., dim], (n, 1, 1))
        else:
            unique, counts = np.unique(indices[0], return_counts=True)
            max_n_points = np.max(counts)
            X_indices = X[indices]
            min_value = np.min(X_indices[:, 0])
            Xdim = np.tile([min_value, min_value, dim], (n, max_n_points, 1))
            Xdim[indices[0], reduce(iconcat, map(range, counts), [])] = \
                X_indices
        Xf.append(Xdim)

    Xf.append(Xuf)
    Xf = np.concatenate(Xf, axis=1)
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
