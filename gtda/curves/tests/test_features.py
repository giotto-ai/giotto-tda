"""Testing for curves features extraction."""

import pytest
import numpy as np
import plotly.io as pio
from numpy.testing import assert_almost_equal
from sklearn.exceptions import NotFittedError
from gtda.curves import StandardFeature

pio.renderers.default = 'plotly_mimetype'

np.random.seed(0)
X = np.random.rand(3, 2, 20)


def test_standard_not_fitted():
    sf = StandardFeature()

    with pytest.raises(NotFittedError):
        sf.transform(X)


@pytest.mark.parametrize('function', ['argmax', 'argmin', 'min', 'max', 'mean',
                                      'std', 'median', 'average'])
def test_standard_shape(function):
    sf = StandardFeature(function=function)
    Xt = sf.fit_transform(X)
    assert Xt.shape == X.shape[:2]


X_res = {
    'flatten': X.reshape(3, 40),
    'argmax': np.array([[8, 0],
                        [12, 12],
                        [9, 3]]),
    'argmin': np.array([[16, 14],
                        [3, 15],
                        [19, 8]]),
    'min': np.array([[0.0202184, 0.0187898],
                     [0.06022547, 0.03918779],
                     [0.00469548, 0.22308163]]),
    'max': np.array([[0.96366276, 0.97861834],
                     [0.98837384, 0.97676109],
                     [0.9292962, 0.96218855]]),
    'mean': np.array([[0.58155482, 0.56784552],
                      [0.39983943, 0.40521714],
                      [0.40951229, 0.61738194]]),
    'std': np.array([[0.27591522, 0.26865653],
                     [0.23900448, 0.31701912],
                     [0.27368227, 0.23340901]]),
    'median': np.array([[0.58540397, 0.61451486],
                        [0.36160934, 0.28947358],
                        [0.36641597, 0.63492923]]),
    'average': np.array([[0.58155482, 0.56784552],
                         [0.39983943, 0.40521714],
                         [0.40951229, 0.61738194]]),
}


@pytest.mark.parametrize('function', ['flatten', 'argmax', 'argmin', 'min',
                                      'max', 'mean', 'std', 'median',
                                      'average'])
def test_standard_transform(function):
    sf = StandardFeature(function=function)

    assert_almost_equal(sf.fit_transform(X), X_res[function])
