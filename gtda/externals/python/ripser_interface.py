from scipy import sparse
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from ..modules import gtda_ripser, gtda_ripser_coeff


def DRFDM(DParam, maxHomDim, thresh=-1, coeff=2, do_cocycles=1):
    if coeff == 2:
        ret = gtda_ripser.rips_dm(DParam, DParam.shape[0], coeff, maxHomDim,
                                  thresh, do_cocycles)
    else:
        ret = gtda_ripser_coeff.rips_dm(DParam, DParam.shape[0], coeff,
                                        maxHomDim, thresh, do_cocycles)
    ret_rips = {}
    ret_rips.update({"births_and_deaths_by_dim": ret.births_and_deaths_by_dim})
    ret_rips.update({"num_edges": ret.num_edges})
    return ret_rips


def DRFDMSparse(I, J, V, N, maxHomDim, thresh=-1, coeff=2, do_cocycles=1):
    if coeff == 2:
        ret = gtda_ripser.rips_dm_sparse(I, J, V, I.size, N, coeff, maxHomDim,
                                         thresh, do_cocycles)
    else:
        ret = gtda_ripser_coeff.rips_dm_sparse(I, J, V, I.size, N, coeff,
                                               maxHomDim, thresh, do_cocycles)
    ret_rips = {}
    ret_rips.update({"births_and_deaths_by_dim": ret.births_and_deaths_by_dim})
    ret_rips.update({"num_edges": ret.num_edges})
    return ret_rips


def dpoint2pointcloud(X, i, metric):
    """
    Return the distance from the ith point in a Euclidean point cloud
    to the rest of the points
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
    """
    Compute a furthest point sampling permutation of a set of points
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
    return (idx_perm, lambdas, dperm2all)


def ripser(X, maxdim=1, thresh=np.inf, coeff=2, metric="euclidean",
           n_perm=None):
    """Compute persistence diagrams for X data array. If X is not a distance
    matrix, it will be converted to a distance matrix using the chosen metric.

    Parameters
    ----------
    X: ndarray (n_samples, n_features)
        A numpy array of either data or distance matrix.
        Can also be a sparse distance matrix of type scipy.sparse
    maxdim: int, optional, default 1
        Maximum homology dimension computed. Will compute all dimensions
        lower than and equal to this value.
        For 1, H_0 and H_1 will be computed.
    thresh: float, default infinity
        Maximum distances considered when constructing filtration.
        If infinity, compute the entire filtration.
    coeff: int prime, default 2
        Compute homology with coefficients in the prime field Z/pZ for p=coeff.
    metric: string or callable
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        specified in pairwise_distances, including "euclidean", "manhattan",
        or "cosine". Alternatively, if metric is a callable function, it is
        called on each pair of instances (rows) and the resulting value
        recorded. The callable should take two arrays from X as input and
        return a value indicating the distance between them.

    n_perm: int
        The number of points to subsample in a "greedy permutation,"
        or a furthest point sampling of the points.  These points
        will be used in lieu of the full point cloud for a faster
        computation, at the expense of some accuracy, which can
        be bounded as a maximum bottleneck distance to all diagrams
        on the original point set
    Returns
    -------
    A dictionary holding all of the results of the computation
    {'dgms': list (size maxdim) of ndarray (n_pairs, 2)
        A list of persistence diagrams, one for each dimension less
        than maxdim. Each diagram is an ndarray of size (n_pairs, 2)
        with the first column representing the birth time and the
        second column representing the death time of each pair.
     'num_edges': int
        The number of edges added during the computation
     'dperm2all': ndarray(n_samples, n_samples) or ndarray (n_perm, n_samples) if n_perm
        The distance matrix used in the computation if n_perm is none.
        Otherwise, the distance from all points in the permutation to
        all points in the dataset
     'idx_perm': ndarray(n_perm) if n_perm > 0
        Index into the original point cloud of the points used
        as a subsample in the greedy permutation
     'r_cover': float
        Covering radius of the subsampled points.
        If n_perm <= 0, then the full point cloud was used and this is 0
    }

    """
    if n_perm and sparse.issparse(X):
        raise Exception(
            "Greedy permutation is not supported for sparse distance matrices"
        )
    if n_perm and n_perm > X.shape[0]:
        raise Exception(
            "Number of points in greedy permutation is greater"
            + " than number of points in the point cloud"
        )
    if n_perm and n_perm < 0:
        raise Exception(
            "Should be a strictly positive number of points in the greedy "
            "permutation"
        )

    idx_perm = np.arange(X.shape[0])
    r_cover = 0.0
    if n_perm:
        idx_perm, lambdas, dperm2all = get_greedy_perm(
            X, n_perm=n_perm, metric=metric
        )
        r_cover = lambdas[-1]
        dm = dperm2all[:, idx_perm]
    else:
        if metric == 'precomputed':
            dm = X
        else:
            dm = pairwise_distances(X, metric=metric)
        dperm2all = dm

    n_points = dm.shape[0]
    if not sparse.issparse(dm) and np.sum(np.abs(dm.diagonal()) > 0) > 0:
        # If any of the diagonal elements are nonzero,
        # convert to sparse format, because currently
        # that's the only format that handles nonzero
        # births
        dm = sparse.coo_matrix(dm)

    if sparse.issparse(dm):
        coo = dm.tocoo()
        res = DRFDMSparse(
            coo.row.astype(dtype=np.int32, order="C"),
            coo.col.astype(dtype=np.int32, order="C"),
            np.array(coo.data, dtype=np.float32, order="C"),
            n_points,
            maxdim,
            thresh,
            coeff,
        )
    else:
        I, J = np.meshgrid(np.arange(n_points), np.arange(n_points))
        DParam = np.array(dm[I > J], dtype=np.float32)
        res = DRFDM(DParam, maxdim, thresh, coeff)

    # Unwrap persistence diagrams
    dgms = res["births_and_deaths_by_dim"]
    for dim in range(len(dgms)):
        N = int(len(dgms[dim]) / 2)
        dgms[dim] = np.reshape(np.array(dgms[dim]), [N, 2])

    ret = {
        "dgms": dgms,
        "num_edges": res["num_edges"],
        "dperm2all": dperm2all,
        "idx_perm": idx_perm,
        "r_cover": r_cover,
    }
    return ret
