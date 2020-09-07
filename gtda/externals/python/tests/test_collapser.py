#!/usr/bin/env python

""" Test comes from
https://github.com/GUDHI/gudhi-devel/blob/master/src/Collapse/example/edge_collapse_basic_example.cpp
"""

import numpy as np
import pytest
from gtda.externals.modules.gtda_collapser import \
    flag_complex_collapse_edges_dense, \
    flag_complex_collapse_edges_sparse, \
    flag_complex_collapse_edges_coo
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, integers, composite
from numpy.testing import assert_almost_equal
from scipy.sparse import coo_matrix, csr_matrix

from gtda.externals import ripser

X = np.array([[0, 1, 1.],
              [1, 2, 1.],
              [2, 3, 1.],
              [3, 0, 1.],
              [0, 2, 2.],
              [1, 3, 2.]], dtype=np.int32)
tX = np.transpose(X)


def check_collapse(collapsed, removed):
    coo = collapsed.tocoo()
    cooT = np.array([coo.row, coo.col, coo.data]).transpose()
    for elem in removed:
        if (cooT == elem).all(axis=1).any():
            return False
    return True


def test_simple_csr_example():
    X = csr_matrix((tX[2], (tX[0], tX[1])))
    coo_ = flag_complex_collapse_edges_sparse(X)
    coo = coo_matrix((coo_[2], (coo_[0], coo_[1])))
    assert check_collapse(coo, [[1, 3, 2]])


def test_simple_coo_example():
    coo_ = flag_complex_collapse_edges_coo(
        tX[0], tX[1], tX[2])
    coo = coo_matrix((coo_[2], (coo_[0], coo_[1])))
    assert check_collapse(coo, [[1, 3, 2]])


def test_simple_dense_example():
    data = csr_matrix((tX[2], (tX[0], tX[1]))).toarray()
    coo_ = flag_complex_collapse_edges_dense(data)
    coo = coo_matrix((coo_[2], (coo_[0], coo_[1])))
    assert check_collapse(coo, [[1, 3, 2]])


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
    shapes = draw(integers(min_value=2, max_value=30))
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
    shape = (np.max(row) + 1, np.max(col) + 1) if not_inf_idx.sum() else (0, 0)
    distance_matrix = coo_matrix((data, (row, col)), shape=shape)
    return distance_matrix


@pytest.mark.parametrize('thresh', [False, True])
@given(distance_matrix=get_dense_distance_matrices())
def test_collapse_consistent_with_no_collapse_dense(thresh, distance_matrix):
    thresh = np.max(distance_matrix) / 2 if thresh else np.inf
    maxdim = 3
    pd_collapse = ripser(distance_matrix, thresh=thresh, maxdim=maxdim,
                         metric='precomputed', collapse_edges=True)['dgms']
    pd_no_collapse = ripser(distance_matrix, thresh=thresh, maxdim=maxdim,
                            metric='precomputed', collapse_edges=False)['dgms']
    for i in range(maxdim + 1):
        pd_collapse[i] = np.sort(pd_collapse[i], axis=0)
        pd_no_collapse[i] = np.sort(pd_no_collapse[i], axis=0)
        assert_almost_equal(pd_collapse[i], pd_no_collapse[i])


@pytest.mark.parametrize('thresh', [False, True])
@given(distance_matrix=get_sparse_distance_matrices())
def test_collapse_consistent_with_no_collapse_sparse(thresh, distance_matrix):
    if thresh and distance_matrix.nnz:
        thresh = np.max(distance_matrix) / 2
    else:
        thresh = np.inf
    maxdim = 3
    pd_collapse = ripser(distance_matrix, thresh=thresh, maxdim=maxdim,
                         metric='precomputed', collapse_edges=True)['dgms']
    pd_no_collapse = ripser(distance_matrix, thresh=thresh, maxdim=maxdim,
                            metric='precomputed', collapse_edges=False)['dgms']
    for i in range(maxdim + 1):
        pd_collapse[i] = np.sort(pd_collapse[i], axis=0)
        pd_no_collapse[i] = np.sort(pd_no_collapse[i], axis=0)
        assert_almost_equal(pd_collapse[i], pd_no_collapse[i])
