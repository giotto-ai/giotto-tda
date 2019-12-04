import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from itertools import product


class OneDimensionalCover(BaseEstimator, TransformerMixin):
    def __init__(self, n_intervals=10, overlap_frac=0.1):
        self.n_intervals = n_intervals
        self.overlap_frac = overlap_frac

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        M, m = np.max(X), np.min(X)
        L = M - m
        # Let the length of each interval be l. The equation to solve for
        # l is (n_intervals - 1) * l * (1 - overlap_frac) + l = L.
        # The maximum left endpoint is at
        # m + (n_intervals - 1) * (1 - overlap_frac) * l
        l = L / (self.n_intervals * (1 - self.overlap_frac) +
                 self.overlap_frac)
        last = m + (self.n_intervals - 1) * (1 - self.overlap_frac) * l
        left_endpts = np.linspace(m, last, num=self.n_intervals, endpoint=True)
        right_endpts = left_endpts + l
        left_endpts[0] = -np.inf
        right_endpts[-1] = np.inf
        Xt = (X > left_endpts) & (X < right_endpts)
        return Xt


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
