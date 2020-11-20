import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, integers, composite
from numpy.testing import assert_almost_equal
from scipy.sparse import coo_matrix

from gtda.externals import ripser


@composite
def get_dense_distance_matrices(draw):
    """Generate 2d dense square arrays of floats, with zero along the
    diagonal."""
    shapes = draw(integers(min_value=2, max_value=30))
    distance_matrix = draw(arrays(dtype=np.float,
                                  elements=floats(allow_nan=False,
                                                  allow_infinity=True,
                                                  min_value=0),
                                  shape=(shapes, shapes), unique=False))
    np.fill_diagonal(distance_matrix, 0)
    return distance_matrix


@composite
def get_sparse_distance_matrices(draw):
    """Generate 2d sparse matrices of floats, with zero along the diagonal."""
    shapes = draw(integers(min_value=2, max_value=40))
    distance_matrix = draw(arrays(dtype=np.float,
                                  elements=floats(allow_nan=False,
                                                  allow_infinity=True,
                                                  min_value=0),
                                  shape=(shapes, shapes), unique=False))
    distance_matrix = np.triu(distance_matrix, k=1)
    distance_matrix = coo_matrix(distance_matrix)
    row, col, data = \
        distance_matrix.row, distance_matrix.col, distance_matrix.data
    not_inf_idx = data != np.inf
    row = row[not_inf_idx]
    col = col[not_inf_idx]
    data = data[not_inf_idx]
    shape = (np.max(row) + 1, np.max(col) + 1) if not_inf_idx.any() else (0, 0)
    distance_matrix = coo_matrix((data, (row, col)), shape=shape)
    return distance_matrix


@pytest.mark.parametrize('thresh', [False, True])
@pytest.mark.parametrize('coeff', [2, 7])
@settings(deadline=500)
@given(distance_matrix=get_dense_distance_matrices())
def test_collapse_consistent_with_no_collapse_dense(thresh,
                                                    coeff, distance_matrix):
    thresh = np.max(distance_matrix) / 2 if thresh else np.inf
    maxdim = 3
    pd_collapse = ripser(distance_matrix, thresh=thresh, maxdim=maxdim,
                         coeff=coeff, metric='precomputed',
                         collapse_edges=True)['dgms']
    pd_no_collapse = ripser(distance_matrix, thresh=thresh, maxdim=maxdim,
                            coeff=coeff, metric='precomputed',
                            collapse_edges=False)['dgms']
    for i in range(maxdim + 1):
        pd_collapse[i] = np.sort(pd_collapse[i], axis=0)
        pd_no_collapse[i] = np.sort(pd_no_collapse[i], axis=0)
        assert_almost_equal(pd_collapse[i], pd_no_collapse[i])


@pytest.mark.parametrize('thresh', [False, True])
@pytest.mark.parametrize('coeff', [2, 7])
@settings(deadline=500)
@given(distance_matrix=get_sparse_distance_matrices())
def test_collapse_consistent_with_no_collapse_coo(thresh,
                                                  coeff, distance_matrix):
    if thresh and distance_matrix.nnz:
        thresh = np.max(distance_matrix) / 2
    else:
        thresh = np.inf
    maxdim = 3
    pd_collapse = ripser(distance_matrix, thresh=thresh, maxdim=maxdim,
                         coeff=coeff, metric='precomputed',
                         collapse_edges=True)['dgms']
    pd_no_collapse = ripser(distance_matrix, thresh=thresh, maxdim=maxdim,
                            coeff=coeff, metric='precomputed',
                            collapse_edges=False)['dgms']
    for i in range(maxdim + 1):
        pd_collapse[i] = np.sort(pd_collapse[i], axis=0)
        pd_no_collapse[i] = np.sort(pd_no_collapse[i], axis=0)
        assert_almost_equal(pd_collapse[i], pd_no_collapse[i])


def test_coo_results_independent_of_order():
    """Regression test for PR #465"""
    data = np.array([6., 8., 2., 4., 5., 9., 10., 3., 1., 1.])
    row = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3])
    col = np.array([4, 1, 3, 2, 4, 3, 2, 3, 4, 4])
    dm = coo_matrix((data, (row, col)))
    diagrams = ripser(dm, metric="precomputed")['dgms']
    diagrams_csr = ripser(dm.tocsr(), metric="precomputed")['dgms']
    expected = [np.array([[0., 1.],
                          [0., 1.],
                          [0., 2.],
                          [0., 5.],
                          [0., np.inf]]),
                np.array([], dtype=np.float64).reshape(0, 2)]
    for i in range(2):
        assert np.array_equal(diagrams[i], expected[i])
        assert np.array_equal(diagrams_csr[i], expected[i])
