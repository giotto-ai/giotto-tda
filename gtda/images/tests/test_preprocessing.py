"""Testing for images preprocessors."""
# License: GNU AGPLv3

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from sklearn.exceptions import NotFittedError

from giotto.images import Binarizer

images_2D = np.stack([np.ones((7, 8)),
                      np.concatenate([np.ones((7, 4)), np.zeros((7, 4))],
                                     axis=1),
                      np.zeros((7, 8))], axis=0)

images_3D = np.stack([np.ones((7, 8, 4)),
                      np.concatenate([np.ones((7, 4, 4)),
                                      np.zeros((7, 4, 4))], axis=1),
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
