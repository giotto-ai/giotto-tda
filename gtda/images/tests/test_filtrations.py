"""Testing for binary image filtrations."""
# License: GNU AGPLv3

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from sklearn.exceptions import NotFittedError

from gtda.images import HeightFiltration

images_2D = np.stack([np.ones((3, 4)),
                      np.concatenate([np.ones((3, 2)), np.zeros((3, 2))],
                                     axis=1),
                      np.zeros((3, 4))], axis=0)

images_3D = np.stack([np.ones((3, 4, 2)),
                      np.concatenate([np.ones((3, 2, 2)),
                                      np.zeros((3, 2, 2))], axis=1),
                      np.zeros((3, 4, 2))], axis=0)


def test_height_not_fitted():
    height = HeightFiltration()
    with pytest.raises(NotFittedError):
        height.transform(images_2D)


def test_height_errors():
    direction = 'a'
    height = HeightFiltration(direction=direction)
    with pytest.raises(TypeError):
        height.fit(images_2D)


images_2D_height = np.array(
    [[[0., 0.70710678, 1.41421356, 2.12132034],
      [0.70710678, 1.41421356, 2.12132034, 2.82842712],
      [1.41421356, 2.12132034, 2.82842712, 3.53553391]],
     [[0., 0.70710678, 4.53553391, 4.53553391],
      [0.70710678, 1.41421356, 4.53553391, 4.53553391],
      [1.41421356, 2.12132034, 4.53553391, 4.53553391]],
     [[4.53553391, 4.53553391, 4.53553391, 4.53553391],
      [4.53553391, 4.53553391, 4.53553391, 4.53553391],
      [4.53553391, 4.53553391, 4.53553391, 4.53553391]]])


images_3D_height = np.array(
    [[[[0., 0.70710678], [0.70710678, 1.41421356],
       [1.41421356, 2.12132034], [2.12132034, 2.82842712]],
      [[0., 0.70710678], [0.70710678, 1.41421356],
       [1.41421356, 2.12132034], [2.12132034, 2.82842712]],
      [[0., 0.70710678], [0.70710678, 1.41421356],
       [1.41421356, 2.12132034], [2.12132034, 2.82842712]]],
     [[[0., 0.70710678], [0.70710678, 1.41421356],
       [3.82842712, 3.82842712], [3.82842712, 3.82842712]],
      [[0., 0.70710678], [0.70710678, 1.41421356],
       [3.82842712, 3.82842712], [3.82842712, 3.82842712]],
      [[0., 0.70710678], [0.70710678, 1.41421356],
       [3.82842712, 3.82842712], [3.82842712, 3.82842712]]],
     [[[3.82842712, 3.82842712], [3.82842712, 3.82842712],
       [3.82842712, 3.82842712], [3.82842712, 3.82842712]],
      [[3.82842712, 3.82842712], [3.82842712, 3.82842712],
       [3.82842712, 3.82842712], [3.82842712, 3.82842712]],
      [[3.82842712, 3.82842712], [3.82842712, 3.82842712],
       [3.82842712, 3.82842712], [3.82842712, 3.82842712]]]])


@pytest.mark.parametrize("direction, images, expected",
                         [(None, images_2D, images_2D_height),
                          ([1, 1], images_2D, images_2D_height),
                          ([1, 0, 1], images_3D, images_3D_height)])
def test_height_transform(direction, images, expected):
    height = HeightFiltration(direction=direction)

    assert_almost_equal(height.fit_transform(images),
                        expected)
