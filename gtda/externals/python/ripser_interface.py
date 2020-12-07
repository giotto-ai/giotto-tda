"""
MIT License
Copyright (c) 2018 Christopher Tralie and Nathaniel Saul
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import gc
from warnings import warn

import numpy as np
from scipy import sparse
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import pairwise_distances

from ..modules import gtda_ripser, gtda_ripser_coeff, gtda_collapser


def DRFDM(DParam, maxHomDim, thresh=-1, coeff=2, do_cocycles=0):
    if coeff == 2:
        ret = gtda_ripser.rips_dm(DParam, DParam.shape[0], coeff, maxHomDim,
                                  thresh, do_cocycles)
    else:
        ret = gtda_ripser_coeff.rips_dm(DParam, DParam.shape[0], coeff,
                                        maxHomDim, thresh, do_cocycles)
    return ret


def DRFDMSparse(I, J, V, N, maxHomDim, thresh=-1, coeff=2, do_cocycles=0):
    if coeff == 2:
        ret = gtda_ripser.rips_dm_sparse(I, J, V, I.size, N, coeff, maxHomDim,
                                         thresh, do_cocycles)
    else:
        ret = gtda_ripser_coeff.rips_dm_sparse(I, J, V, I.size, N, coeff,
                                               maxHomDim, thresh, do_cocycles)
    return ret


def dpoint2pointcloud(X, i, metric):
    """Return the distance from the ith point in a Euclidean point cloud
    to the rest of the points.

    Parameters
    ----------
    X: ndarray (n_samples, n_features)
        A numpy array of data

    i: int
        The index of the point from which to return all distances

    metric: string or callable
        The metric to use when calculating distance between instances in a
        feature array

    """
    ds = pairwise_distances(X, X[i, :][None, :], metric=metric).flatten()
    ds[i] = 0
    return ds


def get_greedy_perm(X, n_perm=None, metric="euclidean"):
    """Compute a furthest point sampling permutation of a set of points

    Parameters
    ----------
    X: ndarray (n_samples, n_features)
        A numpy array of either data or distance matrix

    n_perm: int
        Number of points to take in the permutation

    metric: string or callable
        The metric to use when calculating distance between instances in a
        feature array

    Returns
    -------
    idx_perm: ndarray(n_perm)
        Indices of points in the greedy permutation

    lambdas: ndarray(n_perm)
        Covering radii at different points

    dperm2all: ndarray(n_perm, n_samples)
        Distances from points in the greedy permutation to points
        in the original point set

    """
    if not n_perm:
        n_perm = X.shape[0]
    # By default, takes the first point in the list to be the
    # first point in the permutation, but could be random
    idx_perm = np.zeros(n_perm, dtype=np.int64)
    lambdas = np.zeros(n_perm)
    if metric == 'precomputed':
        dpoint2all = lambda i: X[i, :]
    else:
        dpoint2all = lambda i: dpoint2pointcloud(X, i, metric)
    ds = dpoint2all(0)
    dperm2all = [ds]
    for i in range(1, n_perm):
        idx = np.argmax(ds)
        idx_perm[i] = idx
        lambdas[i - 1] = ds[idx]
        dperm2all.append(dpoint2all(idx))
        ds = np.minimum(ds, dperm2all[-1])
    lambdas[-1] = np.max(ds)
    dperm2all = np.array(dperm2all)
    return idx_perm, lambdas, dperm2all


def _resolve_symmetry_conflicts(coo):
    """Given a sparse matrix in COO format, filter out any entry at location
    (i, j) strictly below the diagonal if the entry at (j, i) is also
    stored."""
    _row, _col, _data = coo.row, coo.col, coo.data

    below_diag = _col < _row
    # Check if there is anything below the main diagonal
    if below_diag.any():
        # Initialize filtered COO data with information in the upper triangle
        in_upper_triangle = np.logical_not(below_diag)
        row = _row[in_upper_triangle]
        col = _col[in_upper_triangle]
        data = _data[in_upper_triangle]

        # Filter out entries below the diagonal for which entries at
        # transposed positions are already available
        upper_triangle_indices = set(zip(row, col))
        additions = tuple(
            zip(*((j, i, x) for (i, j, x) in zip(_row[below_diag],
                                                 _col[below_diag],
                                                 _data[below_diag])
                  if (j, i) not in upper_triangle_indices))
            )
        # Add surviving entries below the diagonal to final COO data
        if additions:
            row_add, col_add, data_add = additions
            row = np.concatenate([row, row_add])
            col = np.concatenate([col, col_add])
            data = np.concatenate([data, data_add])

        return row, col, data
    else:
        return _row, _col, _data


def ripser(X, maxdim=1, thresh=np.inf, coeff=2, metric="euclidean",
           n_perm=None, collapse_edges=False):
    """Compute persistence diagrams for X data array using Ripser [1]_.

    If X is not a distance matrix, it will be converted to a distance matrix
    using the chosen metric.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        A numpy array of either data or distance matrix. Can also be a sparse
        distance matrix of type scipy.sparse

    maxdim : int, optional, default: ``1``
        Maximum homology dimension computed. Will compute all dimensions lower
        than and equal to this value. For 1, H_0 and H_1 will be computed.

    thresh : float, optional, default: ``numpy.inf``
        Maximum distances considered when constructing filtration. If
        ``numpy.inf``, compute the entire filtration.

    coeff : int prime, optional, default: ``2``
        Compute homology with coefficients in the prime field Z/pZ for p=coeff.

    metric : string or callable, optional, default: ``'euclidean'``
        The metric to use when calculating distance between instances in a
        feature array. If set to ``'precomputed'``, input data is interpreted
        as a distance matrix or of adjacency matrices of a weighted undirected
        graph. If a string, it must be one of the options allowed by
        :func:`scipy.spatial.distance.pdist` for its metric parameter, or a
        or a metric listed in
        :obj:`sklearn.pairwise.PAIRWISE_DISTANCE_FUNCTIONS`, including
        ``'euclidean'``, ``'manhattan'`` or ``'cosine'``. If a callable, it
        should take pairs of vectors (1D arrays) as input and, for each two
        vectors in a pair, it should return a scalar indicating the
        distance/dissimilarity between them.

    n_perm : int or None, optional, default: ``None``
        The number of points to subsample in a "greedy permutation", or a
        furthest point sampling of the points. These points will be used in
        lieu of the full point cloud for a faster computation, at the expense
        of some accuracy, which can be bounded as a maximum bottleneck distance
        to all diagrams on the original point set.

    collapse_edges : bool, optional, default: ``False``
        Whether to use the edge collapse algorithm as described in [2]_ prior
        to calling ``ripser``.

    Returns
    -------
    A dictionary holding all of the results of the computation
    {
        'dgms': list (size maxdim) of ndarray (n_pairs, 2)
            A list of persistence diagrams, one for each dimension less
            than maxdim. Each diagram is an ndarray of size (n_pairs, 2)
            with the first column representing the birth time and the
            second column representing the death time of each pair.
        'num_edges': int
            The number of edges added during the computation
        'dperm2all': None or ndarray (n_perm, n_samples)
            ``None`` if n_perm is ``None``. Otherwise, the distance from all
            points in the permutation to all points in the dataset.
        'idx_perm': ndarray(n_perm) if n_perm > 0
            Index into the original point cloud of the points used
            as a subsample in the greedy permutation
        'r_cover': float
            Covering radius of the subsampled points.
            If n_perm <= 0, then the full point cloud was used and this is 0
    }

    Notes
    -----
    `Ripser <https://github.com/Ripser/ripser>`_ is used as a C++ backend
    for computing Vietoris–Rips persistent homology. Python bindings were
    modified for performance from the `ripser.py
    <https://github.com/scikit-tda/ripser.py>`_ package.

    `GUDHI <https://github.com/GUDHI/gudhi-devel>`_ is used as a C++ backend
    for the edge collapse algorithm described in [2]_.

    References
    ----------
    .. [1] U. Bauer, "Ripser: efficient computation of Vietoris–Rips
           persistence barcodes", 2019; `arXiv:1908.02518
           <https://arxiv.org/abs/1908.02518>`_.

    .. [2] J.-D. Boissonnat and S. Pritam, "Edge Collapse and Persistence of
           Flag Complexes"; in *36th International Symposium on Computational
           Geometry (SoCG 2020)*, pp. 19:1–19:15, Schloss
           Dagstuhl-Leibniz–Zentrum für Informatik, 2020;
           `DOI: 10.4230/LIPIcs.SoCG.2020.19
           <https://doi.org/10.4230/LIPIcs.SoCG.2020.19>`_.

    """
    if n_perm and sparse.issparse(X):
        raise Exception("Greedy permutation is not supported for sparse "
                        "distance matrices")
    if n_perm and n_perm > X.shape[0]:
        raise Exception("Number of points in greedy permutation is greater "
                        "than number of points in the point cloud")
    if n_perm and n_perm < 0:
        raise Exception("There should be a strictly positive number of points "
                        "in the greedy permutation")

    idx_perm = np.arange(X.shape[0])
    r_cover = 0.0
    if n_perm:
        idx_perm, lambdas, dperm2all = \
            get_greedy_perm(X, n_perm=n_perm, metric=metric)
        r_cover = lambdas[-1]
        dm = dperm2all[:, idx_perm]
    else:
        if metric == 'precomputed':
            dm = X
        else:
            dm = pairwise_distances(X, metric=metric)
        dperm2all = None

    n_points = max(dm.shape)
    if (dm.diagonal() != 0).any():
        if collapse_edges:
            warn("Edge collapses are not supported when any of the diagonal "
                 "entries are non-zero. Computing persistent homology without "
                 "using edge collapse.")
            collapse_edges = False
        if not sparse.issparse(dm):
            # If any of the diagonal elements are nonzero, convert to sparse
            # format, because currently that's the only format that handles
            # nonzero births
            dm = sparse.coo_matrix(dm)

    is_sparse = sparse.issparse(dm)
    if is_sparse or collapse_edges:
        if is_sparse:
            row, col, data = _resolve_symmetry_conflicts(dm.tocoo())
            if collapse_edges:
                row, col, data = \
                    gtda_collapser.flag_complex_collapse_edges_coo(row,
                                                                   col,
                                                                   data,
                                                                   thresh)
        else:
            row, col, data = \
                gtda_collapser.flag_complex_collapse_edges_dense(dm, thresh)

        res = DRFDMSparse(np.asarray(row, dtype=np.int32, order="C"),
                          np.asarray(col, dtype=np.int32, order="C"),
                          np.asarray(data, dtype=np.float32, order="C"),
                          n_points,
                          maxdim,
                          thresh,
                          coeff)
    else:
        # Only consider strict upper diagonal
        DParam = squareform(dm, checks=False).astype(np.float32)
        # Run garbage collector to free up memory taken by `dm`
        del dm
        gc.collect()
        res = DRFDM(DParam, maxdim, thresh, coeff)

    # Unwrap persistence diagrams
    dgms = res.births_and_deaths_by_dim
    for dim in range(len(dgms)):
        N = int(len(dgms[dim]) / 2)
        dgms[dim] = np.reshape(np.array(dgms[dim]), [N, 2])

    ret = {"dgms": dgms,
           "num_edges": res.num_edges,
           "dperm2all": dperm2all,
           "idx_perm": idx_perm,
           "r_cover": r_cover}

    return ret
