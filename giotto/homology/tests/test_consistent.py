"""Testing for ConsistentRescaling"""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from sklearn.exceptions import NotFittedError
from sklearn.utils.testing import assert_raise_message

from giotto.homology import ConsistentRescaling

X = np.array([[[0, 0], [1, 2], [5, 6]]])

X_rescaled = np.array([[[0., 1., 2.19601308],
                        [1., 0., 1.59054146],
                        [2.19601308, 1.59054146, 0.]]])


@pytest.fixture()
def cr():
    return ConsistentRescaling()


def test_rescaling_init():
    metric = 'euclidean'
    n_neighbor = 1
    n_jobs = 1
    cr = ConsistentRescaling(metric=metric, n_neighbor=n_neighbor,
                             n_jobs=n_jobs)
    assert cr.get_params()['metric'] == metric
    assert cr.get_params()['n_neighbor'] == n_neighbor
    assert cr.get_params()['n_jobs'] == n_jobs


def test_rescaling_not_fitted(cr):
    msg = ("This %s instance is not fitted yet. Call \'fit\'"
           " with appropriate arguments before using this method.")
    assert_raise_message(NotFittedError, msg % 'ConsistentRescaling',
                         cr.transform, X)


def test_rescaling_transform(cr):
    assert_almost_equal(cr.fit_transform(X), X_rescaled)
