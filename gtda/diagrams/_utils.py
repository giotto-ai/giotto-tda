"""Utility functions for diagrams."""
# License: GNU AGPLv3

import numpy as np


def _homology_dimensions_to_sorted_ints(homology_dimensions):
    return tuple(
        sorted([int(dim) if dim != np.inf else dim
                for dim in homology_dimensions])
        )


def _subdiagrams(X, homology_dimensions, remove_dim=False):
    """For each diagram in a collection, extract the subdiagrams in a given
    list of homology dimensions. It is assumed that all diagrams in X contain
    the same number of points in each homology dimension."""
    n_samples = len(X)
    X_0 = X[0]

    def _subdiagrams_single_homology_dimension(homology_dimension):
        n_features_in_dim = np.sum(X_0[:, 2] == homology_dimension)
        try:
            # In this case, reshape ensures copy
            Xs = X[X[:, :, 2] == homology_dimension].\
                reshape(n_samples, n_features_in_dim, 3)
            return Xs
        except ValueError as e:
            if e.args[0].lower().startswith("cannot reshape array"):
                raise ValueError(
                    f"All persistence diagrams in the collection must have "
                    f"the same number of birth-death-dimension triples in any "
                    f"given homology dimension. This is not true in homology "
                    f"dimension {homology_dimension}. Trivial triples for "
                    f"which birth = death may be added or removed to fulfill "
                    f"this requirement."
                )
            else:
                raise e

    if len(homology_dimensions) == 1:
        Xs = _subdiagrams_single_homology_dimension(homology_dimensions[0])
    else:
        # np.concatenate will also create a copy
        Xs = np.concatenate(
            [_subdiagrams_single_homology_dimension(dim)
             for dim in homology_dimensions],
            axis=1
            )
    if remove_dim:
        Xs = Xs[:, :, :2]
    return Xs


def _sample_image(image, diagram_pixel_coords):
    # WARNING: Modifies `image` in-place
    unique, counts = \
        np.unique(diagram_pixel_coords, axis=0, return_counts=True)
    unique = tuple(tuple(row) for row in unique.astype(int).T)
    image[unique] = counts


def _multirange(counts):
    """Given a 1D array of positive integers, generate an array equal to
    np.concatenate([np.arange(c) for c in counts]), but in a faster and more
    memory-efficient way."""
    cumsum = np.cumsum(counts)
    reset_index = cumsum[:-1]
    incr = np.ones(cumsum[-1], dtype=int)
    incr[0] = 0

    # For each index in reset_index, we insert the negative value necessary
    # to offset the cumsum in the last line
    incr[reset_index] = 1 - counts[:-1]
    incr.cumsum(out=incr)

    return incr


def _filter(X, filtered_homology_dimensions, cutoff):
    n = len(X)
    homology_dimensions = sorted(np.unique(X[0, :, 2]))
    unfiltered_homology_dimensions = [dim for dim in homology_dimensions if
                                      dim not in filtered_homology_dimensions]

    if len(unfiltered_homology_dimensions) == 0:
        Xuf = np.empty((n, 0, 3), dtype=X.dtype)
    else:
        Xuf = _subdiagrams(X, unfiltered_homology_dimensions)

    # Compute a global 2D cutoff mask once
    cutoff_mask = X[:, :, 1] - X[:, :, 0] > cutoff
    Xf = []
    for dim in filtered_homology_dimensions:
        # Compute a 2D mask for persistence pairs in dimension dim
        dim_mask = X[:, :, 2] == dim
        # Need the indices relative to X of persistence triples in dimension
        # dim surviving the cutoff
        indices = np.nonzero(np.logical_and(dim_mask, cutoff_mask))
        if not indices[0].size:
            Xdim = np.tile([0., 0., dim], (n, 1, 1))
        else:
            # A unique element k is repeated N times *consecutively* in
            # indices[0] iff there are exactly N valid persistence triples
            # in the k-th diagram
            unique, counts = np.unique(indices[0], return_counts=True)
            max_n_points = np.max(counts)
            # Make a global 2D array of all valid triples
            X_indices = X[indices]
            min_value = np.min(X_indices[:, 0])  # For padding
            # Initialise the array of filtered subdiagrams in dimension m
            Xdim = np.tile([min_value, min_value, dim], (n, max_n_points, 1))
            # Since repeated indices in indices[0] are consecutive and we know
            # the counts per unique index, we can fill the top portion of
            # each 2D array entry of Xdim with the filtered triples from the
            # corresponding entry of X
            Xdim[indices[0], _multirange(counts)] = X_indices
        Xf.append(Xdim)

    Xf.append(Xuf)
    Xf = np.concatenate(Xf, axis=1)
    return Xf


def _bin(X, metric, n_bins=100, homology_dimensions=None, **kw_args):
    if homology_dimensions is None:
        homology_dimensions = sorted(np.unique(X[0, :, 2]))
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
        samplings[dim], step_sizes[dim] = np.linspace(
            min_vals[dim], max_vals[dim], retstep=True, num=n_bins
            )
    if metric in ['landscape', 'betti', 'heat', 'silhouette']:
        for dim in homology_dimensions:
            samplings[dim] = samplings[dim][:, [0], None]
            step_sizes[dim] = step_sizes[dim][0]
    return samplings, step_sizes


def _make_homology_dimensions_mapping(homology_dimensions,
                                      homology_dimensions_ref):
    """`homology_dimensions_ref` is assumed to be a sorted tuple as is e.g.
    :attr:`homology_dimensions_` for several transformers."""
    if homology_dimensions is None:
        homology_dimensions_mapping = list(enumerate(homology_dimensions_ref))
    else:
        homology_dimensions_mapping = []
        for dim in homology_dimensions:
            if dim not in homology_dimensions_ref:
                raise ValueError(f"All homology dimensions must be in "
                                 f"{homology_dimensions_ref}; {dim} is not.")
            else:
                homology_dimensions_arr = np.array(homology_dimensions_ref)
                inv_idx = np.flatnonzero(homology_dimensions_arr == dim)[0]
                homology_dimensions_mapping.append((inv_idx, dim))
    return homology_dimensions_mapping
