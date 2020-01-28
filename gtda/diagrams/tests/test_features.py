"""Testing for PersistenceEntropy"""
# License: GNU AGPLv3

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from sklearn.exceptions import NotFittedError

from gtda.diagrams import PersistenceEntropy

diagram = np.array([[[0, 1, 0], [2, 3, 0], [4, 6, 1], [2, 6, 1]]])


def test_pe_not_fitted():
    pe = PersistenceEntropy()

    with pytest.raises(NotFittedError):
        pe.transform(diagram)


def test_pe_transform():
    pe = PersistenceEntropy()
    diagram_res = np.array([[0.69314718, 0.63651417]])

    assert_almost_equal(pe.fit_transform(diagram), diagram_res)


def test_pi_not_fitted():
    pi = PersistentImage()

    with pytest.raises(NotFittedError):
        pi.transform(diagram)


def test_pi_transform():
    pi = PersistentImage()
    diagram_res = np.array([[0.69314718, 0.63651417]])

    assert_almost_equal(pi.fit_transform(diagram), diagram_res)
