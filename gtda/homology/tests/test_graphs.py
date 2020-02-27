"""Testing for persistent homology on grid."""
# License: GNU AGPLv3

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from sklearn.exceptions import NotFittedError

from gtda.homology import FlagPersistence

pc = np.array([[[2., 2.47942554], [2.47942554, 2.84147098],
               [2.98935825, 2.79848711], [2.79848711, 2.41211849],
               [2.41211849, 1.92484888]]])


def test_fp_params():
    coeff = 'not_defined'
    fp = FlagPersistence(coeff=coeff)

    with pytest.raises(ValueError):
        fp.fit_transform(pc)


def test_fp_not_fitted():
    fp = FlagPersistence()

    with pytest.raises(NotFittedError):
        fp.transform(pc)


pc_fp_res = np.array([[[0., 0.43094373, 0], [0., 0.5117411, 0],
                       [0., 0.60077095, 0], [0., 0.62186205, 0],
                       [0.69093919, 0.80131882, 1]]])


def test_fp_transform():
    fp = FlagPersistence()

    assert_almost_equal(fp.fit_transform(pc), pc_fp_res)
