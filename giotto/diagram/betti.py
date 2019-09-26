import numpy as np 
from sklearn.base import BaseEstimator, TransformerMixin
from giotto.diagram._utils import _create_linspaces
from giotto.diagram._metrics import betti_curves


class BettiCurve(BaseEstimator, TransformerMixin):
    """
    DOC TO DO
    """
    def __init__(self, n_sampled_values=100):
        self.n_sampled_values = n_sampled_values

    def fit(self, X, y=None):
        self._linspaces, _ = \
            _create_linspaces(X, n_sampled_values=self.n_sampled_values)
        return self

    def transform(self, X, y=None):
        n_samples = len(next(iter(X.values())))

        # Only parallelism is across dimensions
        bcs = Parallel(n_jobs=n_jobs)(delayed(betti_curves)(
            X[dim], self._linspaces[dim][:, None, None]) for dim in X.keys())
        bcs = np.stack(bcs, axis=1)
        linspaces = np.vstack([self._linspaces[dim] for dim in X.keys()])
        linspaces = np.tile(linspaces, (n_samples, 1, 1))
        X_transformed = np.stack([linspaces, bcs], axis=2)
        # Shape: (n_samples, n_homology_dimensions, 2, n_sampled_values)
        return X_transformed