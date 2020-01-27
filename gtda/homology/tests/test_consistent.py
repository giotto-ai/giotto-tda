"""Testing for consistent homology."""
# License: GNU AGPLv3

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from sklearn.exceptions import NotFittedError

from gtda.homology import ConsistentRescaling

X_cr = np.array([[[0, 0], [1, 2], [5, 6]]])


def test_rescaling_not_fitted():
    cr = ConsistentRescaling()

    with pytest.raises(NotFittedError):
        cr.transform(X_cr)


def test_rescaling_transform():
    cr = ConsistentRescaling()
    X_rescaled = np.array([[[0., 1., 2.19601308],
                            [1., 0., 1.59054146],
                            [2.19601308, 1.59054146, 0.]]])

    assert_almost_equal(cr.fit_transform(X_cr), X_rescaled)
