"""Tests for CollectionTransformer."""
# License: GNU AGPLv3

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from sklearn.decomposition import PCA

from gtda.metaestimators import CollectionTransformer

rng = np.random.default_rng()

X_arr = rng.random((200, 100, 50))
X_list = list(X_arr)


def test_collection_transformer_input_with_nan():
    multi_pca = CollectionTransformer(PCA())
    X = X_arr.copy()
    X[0, 0, 0] = np.nan

    with pytest.raises(ValueError):
        multi_pca.fit(X)


def test_collection_transformer_invalid_transformer():
    multi_pca = CollectionTransformer(np.mean)

    with pytest.raises(TypeError):
        multi_pca.fit(X_arr)


def test_collection_transformer_is_fitted():
    multi_pca = CollectionTransformer(PCA())
    multi_pca.fit(X_arr)

    assert multi_pca._is_fitted


def test_collection_transformer_no_baseestimator_warn():
    class TestTransformer:
        def __init__(self):
            pass

        def fit_transform(self):
            pass

    test_transformer = TestTransformer()
    with pytest.warns(UserWarning):
        CollectionTransformer(test_transformer).fit(X_arr)


@pytest.mark.parametrize("X", [X_arr, X_list])
@pytest.mark.parametrize("n_jobs", [1, 2, -1])
def test_collection_transformer_fit_transform(X, n_jobs):
    n_components = 3
    pca = PCA(n_components=n_components)
    multi_pca = CollectionTransformer(pca, n_jobs=n_jobs)
    Xt = multi_pca.fit_transform(X)
    assert Xt.shape == (len(X), len(X[0]), n_components)

    first_few_outputs_actual = Xt[:10]
    first_few_outputs_exp = np.asarray([pca.fit_transform(x) for x in X[:10]])
    assert_almost_equal(first_few_outputs_actual, first_few_outputs_exp)


def test_collection_transformer_transform():
    """Test that transform is an alias of fit-transform."""
    pca = PCA()
    assert_almost_equal(CollectionTransformer(pca).fit_transform(X_arr),
                        CollectionTransformer(pca).transform(X_arr))
