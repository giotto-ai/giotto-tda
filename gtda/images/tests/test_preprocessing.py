"""Testing for images preprocessors."""
# License: GNU AGPLv3

import numpy as np
import plotly.io as pio
import pytest
from numpy.testing import assert_almost_equal, assert_equal
from sklearn.exceptions import NotFittedError

from gtda.images import Binarizer, Inverter, Padder, ImageToPointCloud

pio.renderers.default = 'plotly_mimetype'

images_2D = np.stack([
    np.ones((7, 8)),
    np.concatenate([np.ones((7, 4)), np.zeros((7, 4))], axis=1),
    np.zeros((7, 8))], axis=0)

images_3D = np.stack([
    np.ones((7, 8, 4)),
    np.concatenate([np.ones((7, 4, 4)), np.zeros((7, 4, 4))], axis=1),
    np.zeros((7, 8, 4))], axis=0)


def test_binarizer_not_fitted():
    binarizer = Binarizer()
    with pytest.raises(NotFittedError):
        binarizer.transform(images_2D)


def test_binarizer_errors():
    threshold = 'a'
    binarizer = Binarizer(threshold=threshold)
    with pytest.raises(TypeError):
        binarizer.fit(images_2D)


@pytest.mark.parametrize("threshold, expected",
                         [(0.65, images_2D),
                          (0.53, images_3D)])
def test_binarizer_transform(threshold, expected):
    binarizer = Binarizer(threshold=threshold)

    assert_almost_equal(binarizer.fit_transform(expected),
                        expected)


def test_binarizer_fit_transform_plot():
    Binarizer().fit_transform_plot(images_2D, sample=0)


def test_inverter_not_fitted():
    inverter = Inverter()
    with pytest.raises(NotFittedError):
        inverter.transform(images_2D)


images_2D_inverted = np.stack(
    [np.zeros((7, 8)),
     np.concatenate([np.zeros((7, 4)), np.ones((7, 4))], axis=1),
     np.ones((7, 8))], axis=0)

images_3D_inverted = np.stack(
    [np.zeros((7, 8, 4)),
     np.concatenate([np.zeros((7, 4, 4)), np.ones((7, 4, 4))], axis=1),
     np.ones((7, 8, 4))], axis=0)


@pytest.mark.parametrize("images, expected",
                         [(images_2D, images_2D_inverted),
                          (images_3D, images_3D_inverted)])
def test_inverter_transform(images, expected):
    inverter = Inverter()

    assert_almost_equal(inverter.fit_transform(images),
                        expected)


def test_inverter_fit_transform_plot():
    Inverter().fit_transform_plot(images_2D, sample=0)


def test_padder_not_fitted():
    padder = Padder()
    with pytest.raises(NotFittedError):
        padder.transform(images_2D)


@pytest.mark.parametrize("images, paddings",
                         [(images_2D, np.array([1, 1], dtype=np.int)),
                          (images_2D, None),
                          (images_3D, np.array([2, 2, 2], dtype=np.int))])
def test_padder_transform(images, paddings):
    padder = Padder(paddings=paddings)

    if paddings is None:
        expected_shape = np.asarray(images.shape[1:]) + 2
    else:
        expected_shape = images.shape[1:] + 2 * paddings

    assert_equal(padder.fit_transform(images).shape[1:],
                 expected_shape)


def test_padder_fit_transform_plot():
    Padder().fit_transform_plot(images_2D, sample=0)


images_2D_small = np.stack([
    np.ones((3, 2)),
    np.concatenate([np.ones((3, 1)), np.zeros((3, 1))], axis=1),
    np.zeros((3, 2))], axis=0)

images_3D_small = np.stack([
    np.ones((3, 2, 2)),
    np.concatenate([np.ones((3, 1, 2)), np.zeros((3, 1, 2))], axis=1),
    np.zeros((3, 2, 2))], axis=0)


def test_img2pc_not_fitted():
    img2pc = ImageToPointCloud()
    with pytest.raises(NotFittedError):
        img2pc.transform(images_2D)


images_2D_img2pc = list(
    [np.array([[0., 2.], [1., 2.], [0., 1.], [1., 1.], [0., 0.], [1., 0.]]),
     np.array([[0., 2.], [0., 1.], [0., 0.]]),
     np.array([[]])
     ])

images_3D_img2pc = list(
    [np.array([[0., 2., 0.], [0., 2., 1.],
              [1., 2., 0.], [1., 2., 1.],
              [0., 1., 0.], [0., 1., 1.],
              [1., 1., 0.], [1., 1., 1.],
              [0., 0., 0.], [0., 0., 1.],
              [1., 0., 0.], [1., 0., 1.]]),
     np.array([[0., 2., 0.], [0., 2., 1.],
               [0., 1., 0.], [0., 1., 1.],
               [0., 0., 0.], [0., 0., 1.]]),
     np.array([[]])])


def compare_arrays_as_sets(a1, a2):
    """ A helper function to compare two point_clouds.
    They should have the same points, but not necessarily in the same order.
    """
    def to_set_of_elements(a):
        return set([tuple(p) for p in a])
    as1, as2 = [to_set_of_elements(a) for a in [a1, a2]]
    return (as1 <= as2) and (as1 >= as2)


@pytest.mark.parametrize("images, expected",
                         [(images_2D_small, images_2D_img2pc),
                          (images_3D_small, images_3D_img2pc)])
def test_img2pc_transform(images, expected):
    img2pc = ImageToPointCloud()
    results = img2pc.fit_transform(images)

    all(compare_arrays_as_sets(res, expected)
        for res, expected in zip(results,
                                 expected))


@pytest.mark.parametrize("images", [images_2D, images_3D])
def test_img2pc_fit_transform_plot(images):
    ImageToPointCloud().fit_transform_plot(images, sample=0)
