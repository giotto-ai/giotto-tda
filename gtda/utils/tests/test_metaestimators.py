"""Tests for metaestimators."""
# License: GNU AGPLv3

import numpy as np
import pytest
from sklearn.decomposition import PCA
from gtda.utils import ForEachInput

rng = np.random.default_rng()

X_arr = rng.random((200, 100, 50))
X_list = list(X_arr)


def test_for_each_input_fit():
    multi_pca = ForEachInput(PCA())
    X = X_arr.copy()
    X[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        multi_pca.fit(X)


@pytest.mark.parametrize("X", [X_arr, X_list])
@pytest.mark.parametrize("n_jobs", [None, 2, -1])
def test_for_each_input_shape(X, n_jobs):
    n_components = 3
    multi_pca = ForEachInput(PCA(n_components=n_components), n_jobs=n_jobs)
    Xt = multi_pca.fit_transform(X)
    assert Xt.shape == (len(X), len(X[0]), n_components)
