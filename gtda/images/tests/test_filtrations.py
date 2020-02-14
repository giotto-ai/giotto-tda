"""Testing for binary image filtrations."""
# License: GNU AGPLv3

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from sklearn.exceptions import NotFittedError

from gtda.images import HeightFiltration, RadialFiltration

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


def test_radial_not_fitted():
    radial = RadialFiltration()
    with pytest.raises(NotFittedError):
        radial.transform(images_2D)


def test_radial_errors():
    center = 'a'
    radial = RadialFiltration(center=center)
    with pytest.raises(TypeError):
        radial.fit(images_2D)


images_2D_radial = np.array(
    [[[0., 1., 2., 3.],
      [1., 1.41421356, 2.23606798, 3.16227766],
      [2., 2.23606798, 2.82842712, 3.60555128]],
     [[0., 1., 4.60555128, 4.60555128],
      [1., 1.41421356, 4.60555128, 4.60555128],
      [2., 2.23606798, 4.60555128, 4.60555128]],
     [[4.60555128, 4.60555128, 4.60555128, 4.60555128],
      [4.60555128, 4.60555128, 4.60555128, 4.60555128],
      [4.60555128, 4.60555128, 4.60555128, 4.60555128]]])


images_3D_radial = np.array(
    [[[[1.41421356, 1.], [1., 0.],
       [1.41421356, 1.], [2.23606798, 2.]],
      [[1.73205081, 1.41421356], [1.41421356, 1.],
       [1.73205081, 1.41421356], [2.44948974, 2.23606798]],
      [[2.44948974, 2.23606798], [2.23606798, 2.],
       [2.44948974, 2.23606798], [3., 2.82842712]]],
     [[[1.41421356, 1.], [1., 0.], [4., 4.], [4., 4.]],
      [[1.73205081, 1.41421356], [1.41421356, 1.], [4., 4.], [4., 4.]],
      [[2.44948974, 2.23606798], [2.23606798, 2.], [4., 4.], [4., 4.]]],
     [[[4., 4.], [4., 4.],  [4., 4.], [4., 4.]],
      [[4., 4.], [4., 4.], [4., 4.], [4., 4.]],
      [[4., 4.], [4., 4.], [4., 4.], [4., 4.]]]])


@pytest.mark.parametrize("center, images, expected",
                         [(None, images_2D, images_2D_radial),
                          ([0, 0], images_2D, images_2D_radial),
                          ([1, 0, 1], images_3D, images_3D_radial)])
def test_radial_transform(center, images, expected):
    radial = RadialFiltration(center=center)

    print(radial.fit_transform(images))

    assert_almost_equal(radial.fit_transform(images),
                        expected)
