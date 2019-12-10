import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from itertools import product


class OneDimensionalCover(BaseEstimator, TransformerMixin):
    def __init__(self, n_intervals=10, overlap_frac=0.1):
        self.n_intervals = n_intervals
        self.overlap_frac = overlap_frac

    def fit(self, X, y=None):
        # Allow X to have one sample only if one interval is required,
        # in which case the fitted interval will be (-np.inf, np.inf).
        min_samples = 1 if (self.n_intervals == 1) else 2
        X = check_array(X, ensure_2d=False, ensure_min_samples=min_samples)
        if (X.ndim == 2) and (X.shape[1] > 1):
            raise ValueError("X cannot have more than one column.")
        self.left_limits_, self.right_limits_ = self._find_interval_limits(X)
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ['left_limits_', 'right_limits_'])
        X = check_array(X, ensure_2d=False)
        if X.ndim == 1:
            X = X[:, None]

        Xt = (X > self.left_limits_) & (X < self.right_limits_)
        # Avoid repeating the same boolean masks (columns)
        Xt, indices = np.unique(Xt, axis=1, return_index=True)
        # Respect the order in left_limits_ and right_limits_
        Xt = Xt[:, np.argsort(indices)]
        # Remove any mask which contains only False
        nonzero_indices = np.flatnonzero(np.any(Xt, axis=0))
        Xt = Xt[:, nonzero_indices]
        return Xt

    def _find_interval_limits(self, X):
        if X.shape[0] == 1:
            return np.array([-np.inf]), np.array([np.inf])
        M, m = np.max(X), np.min(X)
        L = M - m
        # Let the length of each interval be l. The equation to solve for
        # l is (n_intervals - 1) * l * (1 - overlap_frac) + l = L.
        # The maximum left endpoint is at
        # m + (n_intervals - 1) * (1 - overlap_frac) * l
        l = L / (self.n_intervals * (1 - self.overlap_frac) +
                 self.overlap_frac)
        last = m + (self.n_intervals - 1) * (1 - self.overlap_frac) * l
        left_limits = np.linspace(m, last, num=self.n_intervals, endpoint=True)
        right_limits = left_limits + l
        left_limits[0] = -np.inf
        right_limits[-1] = np.inf
        return left_limits, right_limits

    def fitted_intervals(self):
        """Returns the open intervals computed in :meth:`fit`, as a list of
        tuples (a, b) where a < b.

        """
        check_is_fitted(self, ['left_limits_', 'right_limits_'])
        return list(zip(self.left_limits_, self.right_limits_))


class CubicalCover(BaseEstimator, TransformerMixin):
    """Calculates the cover of an n-dimensional hypercube by taking products
    of intervals of the covers of [min_n, max_n] for each of the n lenses.
    """

    def __init__(self, n_intervals=10, overlap_frac=0.1):
        self.n_intervals = n_intervals
        self.overlap_frac = overlap_frac

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        lens_count = X.shape[1]

        # calculate 1D cover of each lens
        covers = [
            OneDimensionalCover(self.n_intervals, self.overlap_frac)
            .fit_transform(X[:, lens_index, None])
            for lens_index in range(lens_count)
        ]

        # stack intervals for each cover
        intervals = [
            [cover[:, i] for i in range(cover.shape[1])] for cover in covers
        ]

        # calculate masks for pullback cover
        Xt = np.array([np.logical_and.reduce(t)
                       for t in product(*intervals)]).T

        # filter empty intervals in pullback cover
        Xt = Xt[:, np.any(Xt, axis=0)]

        return Xt
