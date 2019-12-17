"""Testing for ReleventHoles"""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from giotto.diagrams import RelevantHoles

X_rh = np.array([[[0, 1, 0], [0, 4, 0], [2, 3, 1], [0, 0, 2]],
                 [[0, 3, 0], [0, 2, 0], [1, 2, 1], [0, 0, 2]]])


def test_throw_exception_when_not_fitted():
    rh = RelevantHoles()

    with pytest.raises(NotFittedError):
        rh.transform(X_rh)


@pytest.mark.parametrize(('threshold_type', 'threshold_fraction', 'X_res'),
                         [('rel', 0.3, np.array([[1, 1, 0], [2, 1, 0]])),
                          ('abs', 0.3, np.array([[2, 1, 0], [2, 1, 0]]))])
def test_output(threshold_type, threshold_fraction, X_res):
    rh = RelevantHoles(threshold_type=threshold_type,
                       threshold_fraction=threshold_fraction)

    assert (rh.fit_transform(X_rh) == X_res).all()
