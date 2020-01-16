import numpy as np
from joblib import Parallel, delayed
from sklearn.pipeline import FeatureUnion, _fit_transform_one, _transform_one


class ListFeatureUnion(FeatureUnion):
    def fit_transform(self, X, y=None, **fit_params):
        """Fit all transformers, transform the data and concatenate results.
        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.
        y : array-like of shape (n_samples, ...), optional
            Targets for supervised learning.
        Returns
        -------
        Xt : list of ndarray
            List of results of transformers.

        """
        results = self._parallel_func(X, y, fit_params, _fit_transform_one)
        if not results:
            # All transformers are None
            return np.zeros((X.shape[0], 0))

        Xt, transformers = zip(*results)
        self._update_transformer_list(transformers)
        Xt = list(Xt)
        return Xt

    def transform(self, X):
        """Transform X separately by each transformer, concatenate results.
        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.
        Returns
        -------
        Xt : list of ndarray
            List of results of transformers.

        """
        Xt = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, X, None, weight)
            for name, trans, weight in self._iter())
        if not Xt:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        Xt = list(Xt)
        return Xt
