"""
The module :mod:`giotto.pipeline` is an extension of scikit-learn's
and implements utilities to build a composite transformer by applying
a transformer pipeline to sliding subwindows of the input data in the
spirit of a pooling layer in a convolutional neural network.
"""

# Author: Guillaume Tauzin <guillaume.tauzin@epfl.ch>
# License: TBD

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.pipeline import Pipeline

from .compose.target import TargetResamplingClassifier, TargetResamplingRegressor

import functools


class SlidingWindowFeatureUnion(BaseEstimator, TransformerMixin):
    """Concatenates results of multiple transformer objects.
    This estimator applies a list of transformer objects in parallel to the
    input data, then concatenates the results. This is useful to combine
    several feature extraction mechanisms into a single transformer.
    Parameters of the transformer may be set using the parameter
    name after 'transformer__'.

    Parameters
    ----------
    transformer : object, required
        Transformer object to be applied to each subwindow of the data.

    width: ndarray of int, required
        Duration of the outer sliding window.

    stride: ndarray of int, default: None
        Stride of the outer sliding window.

    padding: int, default: None
        Duration of the outer sliding window.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    verbose : boolean, optional(default=False)
        If True, the time elapsed while fitting each transformer will be
        printed as it is completed.

    Attributes
    ----------
    transformer_list_ : list of transformer
        List of transformer objects to be applied to the data.

    slice_list_ : list of transformer
        List of transformer objects to be applied to the data.

    Examples
    --------
    >>> from giotto.pipeline import SlidingWindowFeatureUnion
    """
    def __init__(self, transformer, width=None, stride=None, padding=None,
                 n_jobs=None, verbose=False):
        self.transformer = transformer
        self.width = width
        self.stride = stride
        self.padding = padding
        self.n_jobs = n_jobs
        self.verbose = verbose

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.

        """
        return {'width': self.width, 'stride': self.stride,
                'padding': self.padding, 'transformer': self.transformer,
                'n_jobs': self.n_jobs}

    @staticmethod
    def _validate_params(shape_input, shape_width, shape_stride, shape_padding):
        """A class method that checks whether the hyperparameters and the input
        parameters of the :meth:`fit` are valid.

        """
        try:
            assert dimension_image == dimension_direction
        except AssertionError:
            raise ValueError("The dimension of the direction vector does not"
                             "correspond to the dimension of the image.")

    def fit(self, X, y=None):
        """Fit all transformers using X.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data, used to fit transformers.

        y : array-like, shape (n_samples, ...), optional
            Targets for supervised learning.

        Returns
        -------
        self

        """
        self._dimension = len(X.shape) - 1
        self._validate_params(self._dimension, len(self.direction))

        transformers = self._parallel_func(X, y, {}, _fit_one)

        self._update_transformer_list(transformers)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit all transformers, transform the data and concatenate results.

        Parameters
        ----------

        X : iterable or array-like, depending on transformers
            Input data to be transformed.

        y : array-like, shape (n_samples, ...), optional
            Targets for supervised learning.

        Returns
        -------
        X_t : array-like or sparse matrix, shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.

        """
        results = self._parallel_func(X, y, fit_params, _fit_transform_one)
        if not results:
            # All transformers are None
            return np.zeros((X.shape[0], 0))

        Xs, transformers = zip(*results)
        self._update_transformer_list(transformers)

        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = np.hstack(Xs)
        return Xs

    def _parallel_func(self, X, y, fit_params, func):
        """Runs func in parallel on X and y"""
        self.transformer_list = list(self.transformer_list)
        self._validate_transformers()
        transformers = list(self._iter())

        return Parallel(n_jobs=self.n_jobs)(
            delayed(func)(transformer, X, y, weight, **fit_params)
            for idx, (name, transformer, weight)
            in enumerate(transformers, 1))

    def transform(self, X):
        """Transform X separately by each transformer, concatenate results.
        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.
        Returns
        -------
        X_t : array-like or sparse matrix, shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.

        """
        n_samples = X.shape[0]
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(self.transformer_list_[i].transform)(X[:, unzip(self.slice_list_[i])], y)
            for i in range(self.n_windows_))

        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = np.hstack(Xs).reshape((n_samples, -1))
        return Xs

class PipelinePlus(Pipeline):
    def __init__(self, steps, memory=None):
        super(PipelinePlus, self).__init__(steps, memory=memory)

    def fit(self, X, y=None, store_y_resampled=False, **fit_params):
        if (isinstance(self._final_estimator, TargetResamplingClassifier)
           or isinstance(self._final_estimator, TargetResamplingRegressor)) \
           and store_y_resampled:
            fit_params[self.steps[-1][0] + '__store_y_resampled'] = True
        super(PipelinePlus, self).fit(X, y=y, **fit_params)

    def fit_transform(self, X, y=None, store_y_resampled=False, **fit_params):
        if (isinstance(self._final_estimator, TargetResamplingClassifier)
           or isinstance(self._final_estimator, TargetResamplingRegressor)) \
           and store_y_resampled:
            fit_params[self.steps[-1][0] + '__store_y_resampled'] = True
        return super(PipelinePlus, self).fit_transform(X, y=y, **fit_params)

    def fit_predict(self, X, y=None, store_y_resampled=False, **fit_params):
        if (isinstance(self._final_estimator, TargetResamplingClassifier)
           or isinstance(self._final_estimator, TargetResamplingRegressor)) \
           and store_y_resampled:
            fit_params[self.steps[-1][0] + '__store_y_resampled'] = True
        return super(PipelinePlus, self).fit_predict(X, y=y, **fit_params)

    def predict(self, X, y=None, store_y_resampled=False, **predict_params):
        if (isinstance(self._final_estimator, TargetResamplingClassifier)
           or isinstance(self._final_estimator, TargetResamplingRegressor)) \
           and store_y_resampled:
            predict_params['store_y_resampled'] = True
        return super(PipelinePlus, self).predict(X, y=y, **predict_params)

    def get_y_resampled(self):
        check_is_fitted(self._final_estimator, '_is_fitted')
        if (isinstance(self._final_estimator, TargetResamplingClassifier)
           or isinstance(self._final_estimator, TargetResamplingRegressor)):
            try:
                return self.steps[-1][1].y_resampled
            except AttributeError:
                print('The last step in the pipeline has not been fitted with'
                      'store_y_resampled=False')
