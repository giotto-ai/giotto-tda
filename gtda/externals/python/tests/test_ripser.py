import numpy as np
import scipy as sp
from .. import ripser

X = np.random.random((50, 50))
# Without this line, there might be different results because GUDHI
# assumes zeros in the diagonal
np.fill_diagonal(X, 0)
maxdim = 2


def test_with_collapser():
    diags_collapsed = ripser(
        X,
        metric='precomputed',
        maxdim=maxdim,
        collapse_edges=True)['dgms']
    diags_not_collapsed = ripser(
        X,
        metric='precomputed',
        maxdim=maxdim,
        collapse_edges=False)['dgms']

    for i in range(maxdim):
        assert np.array_equal(diags_collapsed[i], diags_not_collapsed[i])


def test_with_collapser_with_tresh():
    thresh = 0.1
    diags_collapsed_thresh = ripser(
        X,
        metric='precomputed',
        maxdim=maxdim,
        thresh=thresh,
        collapse_edges=True)['dgms']
    diags_not_collapsed_thresh = ripser(
        X, metric='precomputed', maxdim=maxdim, thresh=thresh,
        collapse_edges=False)['dgms']

    for i in range(maxdim):
        assert np.array_equal(diags_collapsed_thresh[i],
                              diags_not_collapsed_thresh[i])


def test_with_collapser_coo():
    diags_collapsed = ripser(
        sp.sparse.coo_matrix(X),
        metric='precomputed',
        maxdim=maxdim,
        collapse_edges=True)['dgms']
    diags_not_collapsed = ripser(
        sp.sparse.coo_matrix(X),
        metric='precomputed',
        maxdim=maxdim,
        collapse_edges=False)['dgms']

    for i in range(maxdim):
        assert np.array_equal(diags_collapsed[i], diags_not_collapsed[i])


def test_with_collapser_coo_thresh():
    thresh = 0.1
    diags_collapsed = ripser(
        sp.sparse.coo_matrix(X),
        metric='precomputed',
        maxdim=maxdim,
        thresh=thresh,
        collapse_edges=True)['dgms']
    diags_not_collapsed = ripser(
        sp.sparse.coo_matrix(X),
        metric='precomputed',
        maxdim=maxdim,
        thresh=thresh,
        collapse_edges=False)['dgms']

    for i in range(maxdim):
        assert np.array_equal(diags_collapsed[i], diags_not_collapsed[i])
