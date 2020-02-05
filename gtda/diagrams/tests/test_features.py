"""Testing for PersistenceEntropy"""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from sklearn.exceptions import NotFittedError

from gtda.diagrams import PersistenceEntropy, Silhouette

X_pe = np.array([[[0, 1, 0], [2, 3, 0], [4, 6, 1], [2, 6, 1]]])


def test_pe_not_fitted():
    pe = PersistenceEntropy()

    with pytest.raises(NotFittedError):
        pe.transform(X_pe)


def test_pe_transform():
    pe = PersistenceEntropy()
    X_pe_res = np.array([[0.69314718, 0.63651417]])

    assert_almost_equal(pe.fit_transform(X_pe), X_pe_res)


def test_silhouette_transform():
    sht = Silhouette(n_values=31, order=1.)
    X_sht_res = np.array([0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.2, 0.15, 0.1,
                          0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0., 0.05,
                          0.1, 0.15, 0.2, 0.25, 0.2, 0.15, 0.1, 0.05, 0.])

    assert_almost_equal(sht.fit_transform(X_pe)[0][0], X_sht_res)
