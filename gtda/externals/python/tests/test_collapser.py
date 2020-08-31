#!/usr/bin/env python

""" Test comes from
https://github.com/GUDHI/gudhi-devel/blob/master/src/Collapse/example/edge_collapse_basic_example.cpp
"""


import numpy as np
import scipy
from scipy.sparse import coo_matrix, csr_matrix
from gtda.externals.modules.gtda_collapser import flag_complex_collapse_edges


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
            print(elem)
            return False

    return True


def test_simple_csr_example():
    X = csr_matrix((tX[2], (tX[0], tX[1])))
    coo_ = flag_complex_collapse_edges(X)
    coo = coo_matrix((coo_[2], (coo_[0], coo_[1])))
    assert check_collapse(coo,
                          [[1, 3, 2]])


def test_simple_coo_example():
    coo_ = flag_complex_collapse_edges(
        tX[0], tX[1], tX[2])
    coo = coo_matrix((coo_[2], (coo_[0], coo_[1])))
    assert check_collapse(coo,
                          [[1, 3, 2]])


def test_simple_dense_example():
    data = csr_matrix((tX[2], (tX[0], tX[1]))).tocoo()
    coo_ = flag_complex_collapse_edges(data)
    coo = coo_matrix((coo_[2], (coo_[0], coo_[1])))
    assert check_collapse(coo,
                          [[1, 3, 2]])
