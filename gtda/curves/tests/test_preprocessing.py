"""Testing for curves preprocessing."""

import pytest
import numpy as np
import plotly.io as pio
from numpy.testing import assert_almost_equal
from sklearn.exceptions import NotFittedError
from giotto.curves import Derivative

pio.renderers.default = 'plotly_mimetype'

np.random.seed(0)
X = np.random.rand(1, 2, 5)


def test_derivative_not_fitted():
    d = Derivative()

    with pytest.raises(NotFittedError):
        d.transform(X)


X_res = {
    1: np.array([[[ 0.16637586, -0.11242599, -0.05788019, -0.12122838],
                  [-0.2083069, 0.45418579, 0.07188976, -0.58022124]]]),
    2: np.array([[[-0.27880185, 0.0545458, -0.06334819],
                  [ 0.66249269, -0.38229603, -0.652111]]]),
}

@pytest.mark.parametrize('order', [1, 2])
def test_derivative_transform(order):
    d = Derivative(order)

    assert_almost_equal(d.fit_transform(X), X_res[order])


def test_consistent_fit_transform_plot():
    d = Derivative()
    Xt = d.fit_transform(X)
    d.plot(Xt)
