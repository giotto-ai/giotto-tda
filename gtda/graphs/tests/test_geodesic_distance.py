"""Testing for GraphGeodesicDistance."""

import warnings

import numpy as np
import plotly.io as pio
import pytest
from numpy.ma import masked_array
from numpy.testing import assert_almost_equal
from scipy.sparse import csr_matrix
from sklearn.exceptions import NotFittedError

from gtda.graphs import GraphGeodesicDistance

pio.renderers.default = "plotly_mimetype"


X_ggd = []

X_ggd_float = np.array([
    np.array([[0., 1., 3., 0., 0.],
              [1., 0., 5., 0., 0.],
              [3., 5., 0., 4., 0.],
              [0., 0., 4., 0., 0.],
              [0., 0., 0., 0., 0.]]),
    np.array([[0., 1., 3., 0., np.inf],
              [1., 0., 1., 0., np.inf],
              [3., 1., 0., 4., np.inf],
              [0., 0., 4., 0., np.inf],
              [np.inf, np.inf, np.inf, np.inf, 0.]])
    ])
X_ggd_float_res = np.array([
    np.zeros(X_ggd_float[0].shape, dtype=float),
    np.array([[0., 0., 1., 0., np.inf],
              [0., 0., 1., 0., np.inf],
              [1., 1., 0., 1., np.inf],
              [0., 0., 1., 0., np.inf],
              [np.inf, np.inf, np.inf, np.inf, 0.]])
    ])
X_ggd.append((X_ggd_float, X_ggd_float_res))

X_ggd_float_list = list(X_ggd_float)
X_ggd.append((X_ggd_float_list, X_ggd_float_res))

X_ggd_bool = [np.array([[False, True, False],
                        [True, False, False],
                        [False, False, False]])]
X_ggd_bool_res = np.array([[[0., 1., np.inf],
                            [1., 0., np.inf],
                            [np.inf, np.inf, 0.]]])
X_ggd.append((X_ggd_bool, X_ggd_bool_res))

X_ggd_int = [X_ggd_bool[0].astype(int)]
X_ggd_int_res = np.zeros((1, *X_ggd_int[0].shape), dtype=float)
X_ggd.append((X_ggd_int, X_ggd_int_res))

x_ggd_float = X_ggd_bool[0].astype(float)
X_ggd.append(([x_ggd_float], X_ggd_int_res))

X_ggd.append(
    ([masked_array(x_ggd_float, mask=x_ggd_float == np.inf)], X_ggd_int_res)
    )

X_ggd_csr_int = [csr_matrix(X_ggd_int[0])]
X_ggd.append((X_ggd_csr_int, X_ggd_bool_res))

X_ggd_csr_int_rectang = [csr_matrix(X_ggd_int[0])]
X_ggd_csr_int_rectang[0].resize(2, 3)
X_ggd.append((X_ggd_csr_int_rectang, X_ggd_bool_res))

X_ggd_coo_int_rectang = [X_ggd_csr_int_rectang[0].tocoo()]
X_ggd.append((X_ggd_coo_int_rectang, X_ggd_bool_res))

X_ggd_csr_int_with_zeros = [
    csr_matrix(([1, 1, 0, 0], ([0, 1, 0, 2], [1, 0, 2, 0])))
    ]
X_ggd_csr_int_with_zeros_res = np.array([[[0., 1., 0.],
                                          [1., 0., 1.],
                                          [0., 1., 0.]]])
X_ggd.append((X_ggd_csr_int_with_zeros, X_ggd_csr_int_with_zeros_res))

X_ggd_csr_bool_with_False = [
    csr_matrix(([True, True, False, False], ([0, 1, 0, 2], [1, 0, 2, 0])))
    ]
X_ggd.append((X_ggd_csr_bool_with_False, X_ggd_csr_int_with_zeros_res))


def test_ggd_not_fitted():
    ggd = GraphGeodesicDistance()

    with pytest.raises(NotFittedError):
        ggd.transform(X_ggd)


def test_ggd_fit_transform_plot():
    X = X_ggd[0][0]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Methods .*")
        GraphGeodesicDistance().fit_transform_plot(X, sample=0)


@pytest.mark.parametrize("X, X_res", X_ggd)
@pytest.mark.parametrize("method", ["auto", "FW", "D", "J", "BF"])
def test_ggd_transform_undirected(X, X_res, method):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Methods .*")
        ggd = GraphGeodesicDistance(directed=False, method=method)
        X_ft = ggd.fit_transform(X)

    assert_almost_equal(X_ft, X_res)


def test_parallel_ggd_transform():
    X = X_ggd[0][0]
    ggd = GraphGeodesicDistance(n_jobs=1)
    ggd_parallel = GraphGeodesicDistance(n_jobs=2)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Methods .*")
        assert_almost_equal(ggd.fit_transform(X),
                            ggd_parallel.fit_transform(X))
