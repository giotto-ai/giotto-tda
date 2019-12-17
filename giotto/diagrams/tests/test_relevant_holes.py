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


def test_output_for_relative_theshold():
    rh_rel = RelevantHoles(threshold_type='rel', threshold_fraction=0.3)

    # ground truth
    X_rel_res = np.array([[1, 1, 0], [2, 1, 0]])

    assert (rh_rel.fit_transform(X_rh) == X_rel_res).all()


def test_output_for_absolute_threshold():
    rh_abs = RelevantHoles(threshold_type='abs', threshold_fraction=0.3)

    # ground truth
    X_abs_res = np.array([[2, 1, 0], [2, 1, 0]])

    assert (rh_abs.fit_transform(X_rh) == X_abs_res).all()
