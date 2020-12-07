#!/usr/bin/env python

""" Test comes from
https://github.com/GUDHI/gudhi-devel/blob/master/src/Collapse/example/edge_collapse_basic_example.cpp
"""

import numpy as np
from gtda.externals.modules.gtda_collapser import \
    flag_complex_collapse_edges_dense, \
    flag_complex_collapse_edges_sparse, \
    flag_complex_collapse_edges_coo
from scipy.sparse import coo_matrix, csr_matrix

X = np.array([[0, 1, 1.],
              [1, 2, 1.],
              [2, 3, 1.],
              [0, 3, np.inf],
              [3, 0, 1.],
              [0, 2, 2.],
              [1, 3, 2.]])
tX = np.transpose(X)
tX = np.array([tX[0].astype(np.int32), tX[1].astype(np.int32), tX[2]])
X_expected_row = [0, 1, 2]
X_expected_col = [1, 2, 3]
X_expected_data = [1.0, 1.0, 1.0]


def check_collapse(collapsed, removed):
    coo = collapsed.tocoo()
    cooT = np.array([coo.row, coo.col, coo.data]).transpose()
    for elem in removed:
        if (cooT == elem).all(axis=1).any():
            return False
    return True


def test_simple_csr_example():
    X_ = csr_matrix((tX[2], (tX[0].astype(np.int32),
                             tX[1].astype(np.int32)))).toarray()
    coo_ = flag_complex_collapse_edges_sparse(X_)
    coo = coo_matrix((coo_[2], (coo_[0], coo_[1])))
    assert check_collapse(coo, [[1, 3, 2]])


def test_simple_coo_example():
    coo_ = flag_complex_collapse_edges_coo(
        tX[0], tX[1], tX[2])
    coo = coo_matrix((coo_[2], (coo_[0], coo_[1])))
    assert check_collapse(coo, [[1, 3, 2]])


def test_simple_dense_example():
    data = csr_matrix((tX[2], (tX[0].astype(np.int32),
                               tX[1].astype(np.int32)))).toarray()
    coo_ = flag_complex_collapse_edges_dense(data)
    coo = coo_matrix((coo_[2], (coo_[0], coo_[1])))
    assert check_collapse(coo, [[1, 3, 2]])


def test_csr_expected_output():
    X_ = csr_matrix((tX[2], (tX[0].astype(np.int32),
                             tX[1].astype(np.int32)))).toarray()
    coo_ = flag_complex_collapse_edges_sparse(X_)
    coo = coo_matrix((coo_[2], (coo_[0], coo_[1])))
    assert np.equal(coo.row, X_expected_row).all()
    assert np.equal(coo.col, X_expected_col).all()
    assert np.equal(coo.data, X_expected_data).all()


def test_coo_expected_output():
    coo_ = flag_complex_collapse_edges_coo(
        tX[0], tX[1], tX[2])
    coo = coo_matrix((coo_[2], (coo_[0], coo_[1])))
    print(coo)
    assert np.equal(coo.row, X_expected_row).all()
    assert np.equal(coo.col, X_expected_col).all()
    assert np.equal(coo.data, X_expected_data).all()


def test_dense_expected_output():
    data = csr_matrix((tX[2], (tX[0].astype(np.int32),
                               tX[1].astype(np.int32)))).toarray()
    coo_ = flag_complex_collapse_edges_dense(data)
    coo = coo_matrix((coo_[2], (coo_[0], coo_[1])))
    assert np.equal(coo.row, X_expected_row).all()
    assert np.equal(coo.col, X_expected_col).all()
    assert np.equal(coo.data, X_expected_data).all()
