"""Testing for feature extraction from curves."""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from sklearn.exceptions import NotFittedError

from gtda.curves import StandardFeatures

np.random.seed(0)
X = np.random.rand(3, 2, 20)


def scalar_fn(x):
    return x[0]


def vector_fn(x):
    return x


def vector_fn_2(x):
    return x[:-1]


def test_standard_not_fitted():
    sf = StandardFeatures()
    with pytest.raises(NotFittedError):
        sf.transform(X)


def test_standard_invalid_shape():
    sf = StandardFeatures()

    with pytest.raises(ValueError):
        sf.fit(np.empty((1, 2, 3, 4)))


def test_standard_transform_channels_different_from_fit_channels():
    sf = StandardFeatures()

    with pytest.raises(ValueError, match="Number of channels must be the "
                                         "same as in `fit`"):
        sf.fit(X).transform(X[:, :-1, :])


def test_standard_invalid_function_function_params():
    sf = StandardFeatures(function="wrong")
    with pytest.raises(ValueError):
        sf.fit(X)

    sf.set_params(function=0)
    with pytest.raises(TypeError):
        sf.fit(X)

    sf.set_params(function="max", function_params={"param": 2})
    with pytest.raises(KeyError):
        sf.fit(X)

    sf.set_params(function_params=[])
    with pytest.raises(TypeError, match="If `function` is a string or a "
                                        "callable function"):
        sf.fit(X)

    sf.set_params(function=["wrong", "max"])
    with pytest.raises(ValueError, match="which is not in"):
        sf.fit(X)

    sf.set_params(function=["max", "min"], function_params={})
    with pytest.raises(TypeError, match="If `function` is a list/tuple"):
        sf.fit(X)


@pytest.mark.parametrize("function", ["argmax", "argmin", "min", "max", "mean",
                                      "std", "median", "average", np.max,
                                      scalar_fn, [scalar_fn, "max"],
                                      [scalar_fn, np.max]])
def test_standard_shape_scalar_function(function):
    sf = StandardFeatures(function=function)
    Xt = sf.fit_transform(X)

    assert Xt.shape == X.shape[:2]


def test_standard_shape_function_list_with_none():
    sf = StandardFeatures(function=[None, np.max])
    Xt = sf.fit_transform(X)
    sf.set_params(function="max")

    assert_almost_equal(Xt, sf.fit_transform(X)[:, [1]])


@pytest.mark.parametrize("function", [np.max, scalar_fn, [scalar_fn, np.max]])
def test_standard_function_params_ignored(function):
    sf = StandardFeatures(function=function, function_params={"param": 2})
    sf.fit(X)


X_res = {
    "identity": X.reshape(X.shape[0], -1),
    vector_fn: X.reshape(X.shape[0], -1),
    (vector_fn, vector_fn): X.reshape(X.shape[0], -1),
    "argmax": np.array([[8, 0],
                        [12, 12],
                        [9, 3]]),
    "argmin": np.array([[16, 14],
                        [3, 15],
                        [19, 8]]),
    "min": np.array([[0.0202184, 0.0187898],
                     [0.06022547, 0.03918779],
                     [0.00469548, 0.22308163]]),
    "max": np.array([[0.96366276, 0.97861834],
                     [0.98837384, 0.97676109],
                     [0.9292962, 0.96218855]]),
    "mean": np.array([[0.58155482, 0.56784552],
                      [0.39983943, 0.40521714],
                      [0.40951229, 0.61738194]]),
    "std": np.array([[0.27591522, 0.26865653],
                     [0.23900448, 0.31701912],
                     [0.27368227, 0.23340901]]),
    "median": np.array([[0.58540397, 0.61451486],
                        [0.36160934, 0.28947358],
                        [0.36641597, 0.63492923]]),
    "average": np.array([[0.58155482, 0.56784552],
                         [0.39983943, 0.40521714],
                         [0.40951229, 0.61738194]]),
    np.max: np.array([[0.96366276, 0.97861834],
                      [0.98837384, 0.97676109],
                      [0.9292962, 0.96218855]]),
    (np.max, np.max): np.array([[0.96366276, 0.97861834],
                               [0.98837384, 0.97676109],
                               [0.9292962, 0.96218855]])
    }


@pytest.mark.parametrize("function", X_res.keys())
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_standard_transform(function, n_jobs):
    sf = StandardFeatures(function=function, n_jobs=n_jobs)

    assert_almost_equal(sf.fit_transform(X), X_res[function])


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_standard_transform_mixed_vector(n_jobs):
    sf = StandardFeatures(function=[vector_fn, vector_fn_2], n_jobs=n_jobs)
    Xt = sf.fit_transform(X)

    assert Xt.shape == (len(X), 2 * X.shape[-1] - 1)
    assert_almost_equal(Xt[:, :X.shape[2]], X[:, 0, :])
    assert_almost_equal(Xt[:, X.shape[2]:], X[:, 1, :-1])


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_standard_transform_mixed_vector_scalar(n_jobs):
    sf = StandardFeatures(function=[vector_fn, scalar_fn], n_jobs=n_jobs)
    Xt = sf.fit_transform(X)

    assert Xt.shape == (len(X), X.shape[-1] + 1)
    assert_almost_equal(Xt[:, :X.shape[2]], X[:, 0, :])

    sf.set_params(function=[None, vector_fn])
    Xt = sf.fit_transform(X)

    assert_almost_equal(Xt, X[:, 1, :])
