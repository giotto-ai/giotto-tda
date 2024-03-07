"""Testing for feature extraction from curves."""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from sklearn.exceptions import NotFittedError

from gtda.curves import StandardFeatures

# Generated on 30/09/2020 by
# np.random.seed(0); X = np.random.rand(3, 2, 20)
X = np.array([[[0.5488135, 0.71518937, 0.60276338, 0.54488318, 0.4236548,
                0.64589411, 0.43758721, 0.891773, 0.96366276, 0.38344152,
                0.79172504, 0.52889492, 0.56804456, 0.92559664, 0.07103606,
                0.0871293, 0.0202184, 0.83261985, 0.77815675, 0.87001215],
               [0.97861834, 0.79915856, 0.46147936, 0.78052918, 0.11827443,
                0.63992102, 0.14335329, 0.94466892, 0.52184832, 0.41466194,
                0.26455561, 0.77423369, 0.45615033, 0.56843395, 0.0187898,
                0.6176355, 0.61209572, 0.616934, 0.94374808, 0.6818203]],
              [[0.3595079, 0.43703195, 0.6976312, 0.06022547, 0.66676672,
                0.67063787, 0.21038256, 0.1289263, 0.31542835, 0.36371077,
                0.57019677, 0.43860151, 0.98837384, 0.10204481, 0.20887676,
                0.16130952, 0.65310833, 0.2532916, 0.46631077, 0.24442559],
               [0.15896958, 0.11037514, 0.65632959, 0.13818295, 0.19658236,
                0.36872517, 0.82099323, 0.09710128, 0.83794491, 0.09609841,
                0.97645947, 0.4686512, 0.97676109, 0.60484552, 0.73926358,
                0.03918779, 0.28280696, 0.12019656, 0.2961402, 0.11872772]],
              [[0.31798318, 0.41426299, 0.0641475, 0.69247212, 0.56660145,
                0.26538949, 0.52324805, 0.09394051, 0.5759465, 0.9292962,
                0.31856895, 0.66741038, 0.13179786, 0.7163272, 0.28940609,
                0.18319136, 0.58651293, 0.02010755, 0.82894003, 0.00469548],
               [0.67781654, 0.27000797, 0.73519402, 0.96218855, 0.24875314,
               0.57615733, 0.59204193, 0.57225191, 0.22308163, 0.95274901,
                0.44712538, 0.84640867, 0.69947928, 0.29743695, 0.81379782,
                0.39650574, 0.8811032, 0.58127287, 0.88173536, 0.69253159]]])


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


@pytest.mark.parametrize("shape", [(2,), (2, 3), (2, 3, 4, 5)])
def test_standard_invalid_shape(shape):
    sf = StandardFeatures()

    with pytest.raises(ValueError, match="Input must be 3-dimensional."):
        sf.fit(np.ones(shape))

    with pytest.raises(ValueError, match="Input must be 3-dimensional."):
        sf.fit(X).transform(np.ones(shape))


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

    sf.set_params(function_params=[{}])
    with pytest.raises(ValueError, match="`function_params` has length"):
        sf.fit(X)

    sf.set_params(function=["max"], function_params=None)
    with pytest.raises(ValueError, match="`function` has length"):
        sf.fit(X)


@pytest.mark.parametrize("function, function_params, effective_function, "
                         "effective_function_params",
                         [('max', None, np.max, {}), ('max', {}, np.max, {}),
                          (np.max, None, (np.max, np.max), ({}, {})),
                          (np.max, {}, (np.max, np.max), ({}, {})),
                          ([np.max, np.min], [{}, None],
                           (np.max, np.min), ({}, {})),
                          ([np.max, None], [{}, None],
                           (np.max, None), ({}, {}))])
def test_standard_fit_attrs(function, function_params,
                            effective_function, effective_function_params):
    sf = StandardFeatures(function=function, function_params=function_params)
    sf.fit(X)

    assert sf.n_channels_ == X.shape[1]

    assert sf.effective_function_ == effective_function \
           and sf.effective_function_params_ == effective_function_params


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


def test_standard_transform_function_params():
    weights = np.zeros(X.shape[-1])
    weights[0] = 1
    sf = StandardFeatures(function="average",
                          function_params={"weights": weights})
    Xt = sf.fit_transform(X)

    assert_almost_equal(Xt, X[:, :, 0])

    sf.set_params(function=np.average)
    Xt = sf.fit_transform(X)

    assert_almost_equal(Xt, X[:, :, 0])

    sf.set_params(function=[np.average, np.average],
                  function_params=[{"weights": weights}, {"weights": weights}])
    Xt = sf.fit_transform(X)

    assert_almost_equal(Xt, X[:, :, 0])
