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
    dm = draw(arrays(dtype=np.float,
                     elements=floats(allow_nan=False,
                                     allow_infinity=True,
                                     min_value=0),
                     shape=(shapes, shapes), unique=False))
    np.fill_diagonal(dm, 0)
    return dm


@composite
def get_sparse_distance_matrices(draw):
    """Generate 2d upper triangular sparse matrices of floats, with zero along
    the diagonal."""
    shapes = draw(integers(min_value=2, max_value=40))
    dm = draw(arrays(dtype=np.float,
                     elements=floats(allow_nan=False,
                                     allow_infinity=True,
                                     min_value=0),
                     shape=(shapes, shapes), unique=False))
    dm = np.triu(dm, k=1)
    dm = coo_matrix(dm)
    row, col, data = dm.row, dm.col, dm.data
    not_inf_idx = data != np.inf
    row = row[not_inf_idx]
    col = col[not_inf_idx]
    data = data[not_inf_idx]
    shape_kwargs = {} if data.size else {"shape": (0, 0)}
    dm = coo_matrix((data, (row, col)), **shape_kwargs)
    return dm


@settings(deadline=500)
@given(dm=get_sparse_distance_matrices())
def test_coo_below_diagonal_and_mixed_same_as_above(dm):
    """Test that if we feed sparse matrices representing the same undirected
    weighted graph we obtain the same results regardless of whether all entries
    are above the diagonal, all are below the diagonal, or neither.
    Furthermore, test that conflicts between stored data in the upper and lower
    triangle are resolved in favour of the upper triangle."""
    ripser_kwargs = {"maxdim": 2, "metric": "precomputed"}

    pd_above = ripser(dm, **ripser_kwargs)['dgms']

    pd_below = ripser(dm.T, **ripser_kwargs)['dgms']

    _row, _col, _data = dm.row, dm.col, dm.data
    coo_shape_kwargs = {} if _data.size else {"shape": (0, 0)}
    to_transpose_mask = np.full(len(_row), False)
    to_transpose_mask[np.random.choice(np.arange(len(_row)),
                                       size=len(_row) // 2,
                                       replace=False)] = True
    row = np.concatenate([_col[to_transpose_mask], _row[~to_transpose_mask]])
    col = np.concatenate([_row[to_transpose_mask], _col[~to_transpose_mask]])
    dm_mixed = coo_matrix((_data, (row, col)), **coo_shape_kwargs)
    pd_mixed = ripser(dm_mixed, **ripser_kwargs)['dgms']

    row = np.concatenate([row, _row[to_transpose_mask]])
    col = np.concatenate([col, _col[to_transpose_mask]])
    data = np.random.random(len(row))
    data[:len(_row)] = _data
    dm_conflicts = coo_matrix((data, (row, col)), **coo_shape_kwargs)
    pd_conflicts = ripser(dm_conflicts, **ripser_kwargs)['dgms']

    for i in range(ripser_kwargs["maxdim"] + 1):
        pd_above[i] = np.sort(pd_above[i], axis=0)
        pd_below[i] = np.sort(pd_below[i], axis=0)
        pd_mixed[i] = np.sort(pd_mixed[i], axis=0)
        pd_conflicts[i] = np.sort(pd_conflicts[i], axis=0)
        assert_almost_equal(pd_above[i], pd_below[i])


@pytest.mark.parametrize('thresh', [False, True])
@pytest.mark.parametrize('coeff', [2, 7])
@settings(deadline=500)
@given(dm=get_dense_distance_matrices())
def test_collapse_consistent_with_no_collapse_dense(thresh, coeff, dm):
    thresh = np.max(dm) / 2 if thresh else np.inf
    maxdim = 3
    pd_collapse = ripser(dm, thresh=thresh, maxdim=maxdim, coeff=coeff,
                         metric='precomputed', collapse_edges=True)['dgms']
    pd_no_collapse = ripser(dm, thresh=thresh, maxdim=maxdim, coeff=coeff,
                            metric='precomputed', collapse_edges=False)['dgms']
    for i in range(maxdim + 1):
        pd_collapse[i] = np.sort(pd_collapse[i], axis=0)
        pd_no_collapse[i] = np.sort(pd_no_collapse[i], axis=0)
        assert_almost_equal(pd_collapse[i], pd_no_collapse[i])


@pytest.mark.parametrize('thresh', [False, True])
@pytest.mark.parametrize('coeff', [2, 7])
@settings(deadline=500)
@given(dm=get_sparse_distance_matrices())
def test_collapse_consistent_with_no_collapse_coo(thresh, coeff, dm):
    if thresh and dm.nnz:
        thresh = np.max(dm) / 2
    else:
        thresh = np.inf
    maxdim = 3
    pd_collapse = ripser(dm, thresh=thresh, maxdim=maxdim, coeff=coeff,
                         metric='precomputed', collapse_edges=True)['dgms']
    pd_no_collapse = ripser(dm, thresh=thresh, maxdim=maxdim, coeff=coeff,
                            metric='precomputed', collapse_edges=False)['dgms']
    for i in range(maxdim + 1):
        pd_collapse[i] = np.sort(pd_collapse[i], axis=0)
        pd_no_collapse[i] = np.sort(pd_no_collapse[i], axis=0)
        assert_almost_equal(pd_collapse[i], pd_no_collapse[i])


def test_collapser_with_negative_weights():
    """Test that collapser works as expected when some of the vertex and edge
    weights are negative."""
    n_points = 20
    dm = np.random.random((n_points, n_points))
    np.fill_diagonal(dm, -np.random.random(n_points))
    dm -= 0.2
    dm_sparse = coo_matrix(dm)

    maxdim = 2
    pd_collapse_dense = ripser(dm, metric='precomputed', maxdim=maxdim,
                               collapse_edges=True)['dgms']
    pd_collapse_sparse = ripser(dm_sparse, metric='precomputed',
                                maxdim=maxdim, collapse_edges=True)['dgms']
    pd_no_collapse = ripser(dm, metric='precomputed', maxdim=maxdim,
                            collapse_edges=False)['dgms']

    for i in range(maxdim + 1):
        pd_collapse_dense[i] = np.sort(pd_collapse_dense[i], axis=0)
        pd_collapse_sparse[i] = np.sort(pd_collapse_dense[i], axis=0)
        pd_no_collapse[i] = np.sort(pd_no_collapse[i], axis=0)
        assert_almost_equal(pd_collapse_dense[i], pd_no_collapse[i])
        assert_almost_equal(pd_collapse_sparse[i], pd_no_collapse[i])


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
