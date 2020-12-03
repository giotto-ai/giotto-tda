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
    """Generate 2d upper triangular sparse matrices of floats, with zero along
    the diagonal."""
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
    shape_kwargs = {} if data.size else {"shape": (0, 0)}
    distance_matrix = coo_matrix((data, (row, col)), **shape_kwargs)
    return distance_matrix


@settings(deadline=500)
@given(distance_matrix=get_sparse_distance_matrices())
def test_coo_below_diagonal_and_mixed_same_as_above(distance_matrix):
    """Test that if we feed sparse matrices representing the same undirected
    weighted graph we obtain the same results regardless of whether all entries
    are above the diagonal, all are below the diagonal, or neither.
    Furthermore, test that conflicts between stored data in the upper and lower
    triangle are resolved in favour of the upper triangle."""
    ripser_kwargs = {"maxdim": 2, "metric": "precomputed"}

    pd_above = ripser(distance_matrix, **ripser_kwargs)['dgms']

    pd_below = ripser(distance_matrix.T, **ripser_kwargs)['dgms']

    _row, _col, _data = (distance_matrix.row, distance_matrix.col,
                         distance_matrix.data)
    coo_shape_kwargs = {} if _data.size else {"shape": (0, 0)}
    to_transpose_mask = np.full(len(_row), False)
    to_transpose_mask[np.random.choice(np.arange(len(_row)),
                                       size=len(_row) // 2,
                                       replace=False)] = True
    row = np.concatenate([_col[to_transpose_mask], _row[~to_transpose_mask]])
    col = np.concatenate([_row[to_transpose_mask], _col[~to_transpose_mask]])
    distance_matrix_mixed = coo_matrix((_data, (row, col)), **coo_shape_kwargs)
    pd_mixed = ripser(distance_matrix_mixed, **ripser_kwargs)['dgms']

    row = np.concatenate([row, _row[to_transpose_mask]])
    col = np.concatenate([col, _col[to_transpose_mask]])
    data = np.random.random(len(row))
    data[:len(_row)] = _data
    distance_matrix_conflicts = coo_matrix((data, (row, col)),
                                           **coo_shape_kwargs)
    pd_conflicts = ripser(distance_matrix_conflicts, **ripser_kwargs)['dgms']

    for i in range(ripser_kwargs["maxdim"] + 1):
        pd_above[i] = np.sort(pd_above[i], axis=0)
        pd_below[i] = np.sort(pd_below[i], axis=0)
        pd_mixed[i] = np.sort(pd_mixed[i], axis=0)
        pd_conflicts[i] = np.sort(pd_conflicts[i], axis=0)
        assert_almost_equal(pd_above[i], pd_below[i])


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
