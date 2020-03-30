"""Testing for binary image filtrations."""
# License: GNU AGPLv3

import numpy as np
import plotly.io as pio
import pytest
from numpy.testing import assert_almost_equal
from sklearn.exceptions import NotFittedError

from gtda.images import HeightFiltration, RadialFiltration, \
    DilationFiltration, ErosionFiltration, SignedDistanceFiltration

pio.renderers.default = 'plotly_mimetype'

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
                          (np.asarray([1, 1]), images_2D, images_2D_height),
                          (np.asarray([1, 0, 1]), images_3D,
                           images_3D_height)])
def test_height_transform(direction, images, expected):
    height = HeightFiltration(direction=direction)

    assert_almost_equal(height.fit_transform(images),
                        expected)


def test_height_fit_transform_plot():
    HeightFiltration().fit_transform_plot(images_2D, sample=0)


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
     [[[4., 4.], [4., 4.], [4., 4.], [4., 4.]],
      [[4., 4.], [4., 4.], [4., 4.], [4., 4.]],
      [[4., 4.], [4., 4.], [4., 4.], [4., 4.]]]])


@pytest.mark.parametrize("center, images, expected",
                         [(None, images_2D, images_2D_radial),
                          (np.asarray([0, 0]), images_2D, images_2D_radial),
                          (np.asarray([1, 0, 1]), images_3D,
                           images_3D_radial)])
def test_radial_transform(center, images, expected):
    radial = RadialFiltration(center=center)

    assert_almost_equal(radial.fit_transform(images),
                        expected)


def test_radial_fit_transform_plot():
    RadialFiltration().fit_transform_plot(images_2D, sample=0)


def test_dilation_not_fitted():
    dilation = DilationFiltration()
    with pytest.raises(NotFittedError):
        dilation.transform(images_2D)


def test_dilation_errors():
    n_iterations = 'a'
    dilation = DilationFiltration(n_iterations=n_iterations)
    with pytest.raises(TypeError):
        dilation.fit(images_2D)


images_2D_dilation = np.array(
    [[[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]],
     [[0., 0., 1., 2.], [0., 0., 1., 2.], [0., 0., 1., 2.]],
     [[7., 7., 7., 7.], [7., 7., 7., 7.], [7., 7., 7., 7.]]])


images_3D_dilation = np.array(
    [[[[0., 0.], [0., 0.], [0., 0.], [0., 0.]],
      [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],
      [[0., 0.], [0., 0.], [0., 0.], [0., 0.]]],
     [[[0., 0.], [0., 0.], [1., 1.], [9., 9.]],
      [[0., 0.], [0., 0.], [1., 1.], [9., 9.]],
      [[0., 0.], [0., 0.], [1., 1.], [9., 9.]]],
     [[[9., 9.], [9., 9.], [9., 9.], [9., 9.]],
      [[9., 9.], [9., 9.], [9., 9.], [9., 9.]],
      [[9., 9.], [9., 9.], [9., 9.], [9., 9.]]]])


@pytest.mark.parametrize("n_iterations, images, expected",
                         [(None, images_2D, images_2D_dilation),
                          (100, images_2D, images_2D_dilation),
                          (1, images_3D, images_3D_dilation)])
def test_dilation_transform(n_iterations, images, expected):
    dilation = DilationFiltration(n_iterations=n_iterations)

    assert_almost_equal(dilation.fit_transform(images),
                        expected)


def test_dilation_fit_transform_plot():
    DilationFiltration().fit_transform_plot(images_2D, sample=0)


def test_erosion_not_fitted():
    erosion = ErosionFiltration()
    with pytest.raises(NotFittedError):
        erosion.transform(images_2D)


def test_erosion_errors():
    n_iterations = 'a'
    erosion = ErosionFiltration(n_iterations=n_iterations)
    with pytest.raises(TypeError):
        erosion.fit(images_2D)


images_2D_erosion = np.array(
    [[[7., 7., 7., 7.], [7., 7., 7., 7.], [7., 7., 7., 7.]],
     [[2., 1., 0., 0.], [2., 1., 0., 0.], [2., 1., 0., 0.]],
     [[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]]])


images_3D_erosion = np.array(
    [[[[9., 9.], [9., 9.], [9., 9.], [9., 9.]],
      [[9., 9.], [9., 9.], [9., 9.], [9., 9.]],
      [[9., 9.], [9., 9.], [9., 9.], [9., 9.]]],
     [[[9., 9.], [1., 1.], [0., 0.], [0., 0.]],
      [[9., 9.], [1., 1.], [0., 0.], [0., 0.]],
      [[9., 9.], [1., 1.], [0., 0.], [0., 0.]]],
     [[[0., 0.], [0., 0.], [0., 0.], [0., 0.]],
      [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],
      [[0., 0.], [0., 0.], [0., 0.], [0., 0.]]]])


@pytest.mark.parametrize("n_iterations, images, expected",
                         [(None, images_2D, images_2D_erosion),
                          (100, images_2D, images_2D_erosion),
                          (1, images_3D, images_3D_erosion)])
def test_erosion_transform(n_iterations, images, expected):
    erosion = ErosionFiltration(n_iterations=n_iterations)

    assert_almost_equal(erosion.fit_transform(images),
                        expected)


def test_erosion_fit_transform_plot():
    ErosionFiltration().fit_transform_plot(images_2D, sample=0)


def test_signed_not_fitted():
    signed = SignedDistanceFiltration()
    with pytest.raises(NotFittedError):
        signed.transform(images_2D)


def test_signed_errors():
    n_iterations = 'a'
    signed = SignedDistanceFiltration(n_iterations=n_iterations)
    with pytest.raises(TypeError):
        signed.fit(images_2D)


images_2D_signed = np.array(
    [[[7., 7., 7., 7.], [7., 7., 7., 7.], [7., 7., 7., 7.]],
     [[1., 0., -1., -2.], [1., 0., -1., -2.], [1., 0., -1., -2.]],
     [[-7., -7., -7., -7.], [-7., -7., -7., -7.], [-7., -7., -7., -7.]]])

images_3D_signed = np.array(
    [[[[9., 9.], [9., 9.], [9., 9.], [9., 9.]],
      [[9., 9.], [9., 9.], [9., 9.], [9., 9.]],
      [[9., 9.], [9., 9.], [9., 9.], [9., 9.]]],
     [[[1., 1.], [0., 0.], [-1., -1.], [-2., -2.]],
      [[1., 1.], [0., 0.], [-1., -1.], [-2., -2.]],
      [[1., 1.], [0., 0.], [-1., -1.], [-2., -2.]]],
     [[[-9., -9.], [-9., -9.], [-9., -9.], [-9., -9.]],
      [[-9., -9.], [-9., -9.], [-9., -9.], [-9., -9.]],
      [[-9., -9.], [-9., -9.], [-9., -9.], [-9., -9.]]]])


@pytest.mark.parametrize("n_iterations, images, expected",
                         [(None, images_2D, images_2D_signed),
                          (100, images_2D, images_2D_signed),
                          (2, images_3D, images_3D_signed)])
def test_signed_transform(n_iterations, images, expected):
    signed = SignedDistanceFiltration(n_iterations=n_iterations)

    assert_almost_equal(signed.fit_transform(images),
                        expected)


def test_signed_fit_transform_plot():
    SignedDistanceFiltration().fit_transform_plot(images_2D, sample=0)
