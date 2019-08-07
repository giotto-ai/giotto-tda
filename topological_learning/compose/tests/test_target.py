import pytest

import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression

from topological_learning.compose import (TargetResampler, TargetResamplingClassifier,
                                          TargetResamplingRegressor)


class TestTargetResampler:
    @staticmethod
    @pytest.fixture()
    def resampler_data():
        return np.arange(0, 20)

    @pytest.mark.parametrize("step_size", [3, 5])
    def test_resampling(self, resampler_data, step_size):
        max_len_of_X = len(resampler_data) // step_size
        X = np.zeros((max_len_of_X,))

        resampled = TargetResampler(step_size=step_size, from_right=False) \
            .fit(resampler_data, X) \
            .transform(resampler_data, X)

        assert len(resampled) == max_len_of_X

    def test_fit_transform(self, resampler_data):
        max_len_of_X = len(resampler_data) // 3
        X = np.zeros((max_len_of_X,))

        resampled = TargetResampler(step_size=3, from_right=False) \
            .fit_transform(resampler_data, X, )

        assert len(resampled) == max_len_of_X

    @pytest.mark.parametrize("step_size", [1, 5])
    def test_resampling_failure(self, resampler_data, step_size):
        max_len_of_X = len(resampler_data) // step_size
        X = np.zeros((max_len_of_X + 2,))

        with pytest.raises(ValueError, match=".*Target array cannot be resampled.*"):
            _ = TargetResampler(step_size=step_size, from_right=False) \
                .fit(resampler_data, X) \
                .transform(resampler_data, X)

    @pytest.mark.parametrize("from_right", [True, False])
    def test_left_right(self, resampler_data, from_right):
        step_size = 3
        max_len_of_X = len(resampler_data) // step_size
        X = np.zeros((max_len_of_X,))

        resampled = TargetResampler(step_size=step_size, from_right=from_right) \
            .fit(resampler_data, X) \
            .transform(resampler_data, X)

        assert len(resampled) == max_len_of_X

        if from_right:
            assert resampler_data[-1] in resampled
            assert resampler_data[0] not in resampled
        else:
            assert resampler_data[-1] not in resampled
            assert resampler_data[0] in resampled

    def test_sparse_input(self, resampler_data):
        from scipy.sparse import csc_matrix
        rng = np.random.mtrand.RandomState(42)
        A = rng.randn(10, 10)

        A[:, 2 * np.arange(5)] = 0
        A_sparse = csc_matrix(A)

        resampled_dense = TargetResampler(step_size=2, from_right=False) \
            .fit(resampler_data, A) \
            .transform(resampler_data, A)
        resampled_sparse = TargetResampler(step_size=2, from_right=False) \
            .fit(resampler_data, A_sparse) \
            .transform(resampler_data, A_sparse)

        assert np.array_equal(resampled_dense, resampled_sparse)

    def test_y_dimensions(self):
        X = np.zeros((5, 3))

        y = np.arange(10)
        resampled = TargetResampler(step_size=2, from_right=False) \
            .fit(y, X) \
            .transform(y, X)
        assert resampled.shape == (5,)

        y = np.arange(10).reshape(10, 1)
        resampled = TargetResampler(step_size=2, from_right=False) \
            .fit(y, X) \
            .transform(y, X)
        assert resampled.shape == (5, 1)

        y = np.arange(20).reshape(10, 2)
        resampled = TargetResampler(step_size=2, from_right=False) \
            .fit(y, X) \
            .transform(y, X)
        assert resampled.shape == (5, 2)

        y = np.arange(30).reshape(10, 3)
        resampled = TargetResampler(step_size=2, from_right=False) \
            .fit(y, X) \
            .transform(y, X)
        assert resampled.shape == (5, 3)


class TestTargetResamplingClassifier:
    @staticmethod
    @pytest.fixture()
    def classifier():
        return LogisticRegression()

    def test_with_resample_and_classif(self, classifier):
        res = TargetResampler(step_size=3)
        trc = TargetResamplingClassifier(classifier=classifier, resampler=res)

        X = np.array([[3, 3], [1, 0], [1, 0]])
        y = np.asarray([0, 0, 0, 1, 1, 1, 1, 1, 1])
        trc.fit(X, y)
        assert 1.0 == trc.score(X, [0, 1, 1])
        np.testing.assert_array_equal(trc.predict(X), np.array([0, 1, 1]))

    def test_sparse_input(self, classifier):
        res = TargetResampler(step_size=2)
        trc = TargetResamplingClassifier(classifier=classifier, resampler=res)

        from scipy.sparse import csc_matrix
        rng = np.random.mtrand.RandomState(42)
        X = rng.randn(10, 10)

        X[:, 2 * np.arange(5)] = 0
        X_sparse = csc_matrix(X)

        y = rng.randint(0, 2, size=(23,))

        trc.fit(X, y)
        dense_pred = trc.fit(X, y).predict(X)
        sparse_pred = trc.fit(X_sparse, y).predict(X_sparse)

        np.testing.assert_array_equal(dense_pred, sparse_pred)


class TestTargetResamplingRegressor:
    @staticmethod
    @pytest.fixture()
    def regressor():
        return LinearRegression()

    def test_with_resample_and_regress(self, regressor):
        res = TargetResampler(step_size=2)
        trc = TargetResamplingRegressor(regressor=regressor, resampler=res)

        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y = np.asarray([10, 10, 20, 20, 30, 30, 40, 40])
        trc.fit(X, y)
        assert 1.0 == trc.score(X, [10, 20, 30, 40])
        np.testing.assert_array_almost_equal(trc.predict(X), np.array([10., 20., 30., 40.]),
                                             decimal=13)

    def test_sparse_input(self, regressor):
        res = TargetResampler(step_size=2)
        trc = TargetResamplingRegressor(regressor=regressor, resampler=res)

        from scipy.sparse import csc_matrix
        rng = np.random.mtrand.RandomState(42)
        X = rng.randn(10, 10)

        X[:, 2 * np.arange(5)] = 0
        X_sparse = csc_matrix(X)

        y = rng.randn(23, 2)

        trc.fit(X, y)
        dense_pred = trc.fit(X, y).predict(X)
        sparse_pred = trc.fit(X_sparse, y).predict(X_sparse)

        np.testing.assert_array_almost_equal(dense_pred, sparse_pred, decimal=13)

