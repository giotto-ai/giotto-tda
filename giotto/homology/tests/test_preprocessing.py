import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from giotto.homology.preprocessing import LocalNeighborhood

X = np.array([[1.3, 2.5, 3.1, 4.5, 5.4],
              [6.5, 7.2, 8.2, 9.4, 10.7],
              [2.7, 3.3, 1.1, 4.5, 5.6],
              [1.4, 5.6, 4.3, 1.2, 8.1]])


def test_non_existent_metric():
    ln = LocalNeighborhood(metric='wrong_metric')

    with pytest.raises(ValueError):
        ln.fit_transform(X)


def test_k_min_greater_k_max():
    ln = LocalNeighborhood(k_min=10, k_max=5)

    with pytest.raises(ValueError):
        ln.fit(X)


def test_local_neighborhood_not_fitted():
    ln = LocalNeighborhood()

    with pytest.raises(NotFittedError):
        ln.transform(X)


def test_correct_shape():
    ln = LocalNeighborhood(k_min=2, k_max=3, dist_percentage=0.9)
    Xt = ln.fit_transform(X)
    Xt_expected = [[[1.3, 2.5, 3.1, 4.5, 5.4],
                    [2.7, 3.3, 1.1, 4.5, 5.6],
                    [1.4, 5.6, 4.3, 1.2, 8.1]],
                   [[6.5, 7.2, 8.2, 9.4, 10.7],
                    [1.4, 5.6, 4.3, 1.2, 8.1],
                    [1.4, 5.6, 4.3, 1.2, 8.1]],
                   [[1.3, 2.5, 3.1, 4.5, 5.4],
                    [2.7, 3.3, 1.1, 4.5, 5.6],
                    [1.4, 5.6, 4.3, 1.2, 8.1]],
                   [[1.3, 2.5, 3.1, 4.5, 5.4],
                    [2.7, 3.3, 1.1, 4.5, 5.6],
                    [1.4, 5.6, 4.3, 1.2, 8.1]]]

    np.testing.assert_array_equal(Xt, Xt_expected)


def test_zero_distance_one_element():
    ln = LocalNeighborhood(k_min=1, k_max=3, dist_percentage=0)
    Xt = ln.fit_transform(X)
    Xt_expected = np.expand_dims(X, axis=1)

    np.testing.assert_array_equal(Xt, Xt_expected)
