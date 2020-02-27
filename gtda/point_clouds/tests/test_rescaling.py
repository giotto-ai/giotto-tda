"""Testing for rescaling transfomers."""
# License: GNU AGPLv3

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from sklearn.exceptions import NotFittedError

from gtda.point_clouds import ConsistentRescaling, ConsecutiveRescaling

Xr = np.array([[[0, 0], [1, 2], [5, 6]]])


def test_consistent_not_fitted():
    cr = ConsistentRescaling()

    with pytest.raises(NotFittedError):
        cr.transform(Xr)


def test_consistent_transform():
    cr = ConsistentRescaling()
    Xres = np.array([[[0., 1., 2.19601308],
                      [1., 0., 1.59054146],
                      [2.19601308, 1.59054146, 0.]]])

    assert_almost_equal(cr.fit_transform(Xr), Xres)


def test_consecutive_not_fitted():
    cr = ConsecutiveRescaling()

    with pytest.raises(NotFittedError):
        cr.transform(Xr)


def test_consecutive_transform():
    cr = ConsecutiveRescaling()
    Xres = np.array([[[0., 0., 7.81024968],
                      [2.23606798, 0., 0.],
                      [7.81024968, 5.65685425, 0.]]])

    assert_almost_equal(cr.fit_transform(Xr), Xres)
