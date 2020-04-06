"""Testing for persistent homology on grids."""
# License: GNU AGPLv3

import numpy as np
import plotly.io as pio
import pytest
from numpy.testing import assert_almost_equal
from sklearn.exceptions import NotFittedError

from gtda.homology import CubicalPersistence

pio.renderers.default = 'plotly_mimetype'

X = np.array([[[2., 2.47942554],
               [2.47942554, 2.84147098],
               [2.98935825, 2.79848711],
               [2.79848711, 2.41211849],
               [2.41211849, 1.92484888]]])

X_cp_res = np.array([[[1.9248489, 2.9893582, 0.],
                      [2., 2.79848711, 0],
                      [0., 0., 1]]])

X_cp_res_periodic = np.array([[[1.9248489, 2.9893582, 0.],
                               [2., 2.9893582, 1.],
                               [2.7984871, 2.9893582, 1.],
                               [2.7984871, 2.841471, 1.]]])


def test_cp_not_fitted():
    cp = CubicalPersistence()

    with pytest.raises(NotFittedError):
        cp.transform(X)


@pytest.mark.parametrize('hom_dims', [None, (0,), (1,), (0, 1)])
def test_cp_fit_transform_plot(hom_dims):
    CubicalPersistence().fit_transform_plot(
        X, sample=0, homology_dimensions=hom_dims)


@pytest.mark.parametrize("periodic_dimensions, expected",
                         [(None, X_cp_res),
                          (np.array([False, False]), X_cp_res),
                          (np.array([True, True]), X_cp_res_periodic)])
def test_cp_transform(periodic_dimensions, expected):
    cp = CubicalPersistence(periodic_dimensions=periodic_dimensions)
    assert_almost_equal(cp.fit_transform(X), expected)
