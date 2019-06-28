# Authors: Umberto Lupo <u.lupo@l2f.ch>
#          Guillaume Tauzin <guillaume.tauzin@epfl.ch>
# License: BSD 3 clause

import warnings

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, safe_indexing


class TargetResamplingClassifier(BaseEstimator, ClassifierMixin):
    """Meta-estimator to classify on a resampled target.
    Useful in classification problems whenever the correct correspondence between
    the feature and target arrays is obtained via resampling of the target array
    (in a way which might depend on the feature array).
    The computation during ``fit`` is::
        classifier.fit(X, resampler.transform(y, X)).
    The computation during ``predict`` is::
        classifier.predict(X)).
    Parameters
    ----------
    classifier : object, default=LogisticRegression()
        Classifier object such as derived from ``ClassifierMixin``. This
        classifier will automatically be cloned each time prior to fitting.
    resampler : object, default=None
        Estimator object such as derived from ``TransformerMixin``. Note that
        this resampler will be cloned during fitting.
    Attributes
    ----------
    classifier_ : object
        Fitted classifier.
    resampler_ : object
        Clone of the resampler used in ``fit``.
    Examples
    --------
    >>> import numpy as np
    >>> import topological_learning as tl
    >>> from sklearn.linear_model import LogisticRegression
    >>> from tl.compose import TargetResampler, TargetResamplingClassifier
    >>> ss = 2
    >>> res = TargetResampler(step_size=ss)
    >>> trc = TargetResamplingClassifier(classifier=LogisticRegression(),
    ...                                   resampler=res)
    >>> X = np.arange(4).reshape(-1,ss)
    >>> y = np.asarray([0,1,0])
    >>> trc.fit(X, y) # doctest: +ELLIPSIS
    TargetResamplingClassifier(...)
    >>> trc.score(X, y)
    1.0
    >>> trc.classifier_.coef_
    array([[0.45235762, 0.19441371]])
    Notes
    -----
    Internally, the target ``y`` is always converted into a 2-dimensional array
    to be used by scikit-learn transformers. At the time of prediction, the
    output will be reshaped to a have the same number of dimensions as ``y``.
    """
    def __init__(self, classifier=None, resampler=None):
        self.classifier = classifier
        self.resampler = resampler

    def get_params(self, deep=True):
        return {'classifier': self.classifier, 'resampler': self.resampler}

    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_1, ...)
            Training array, where n_samples is the number of samples and
            the total number of features is the product of all n_i's.
        y : array-like, shape (n_samples,)
            Target values.
        sample_weight : array-like, shape (n_samples,) optional
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.
        Returns
        -------
        self : object
        """
        y = check_array(y, accept_sparse=False, force_all_finite=True,
                        ensure_2d=False, dtype='numeric')

        # store the number of dimension of the target to predict an array of
        # similar shape at predict
        self._training_dim = y.ndim

        # Transformers are designed to modify X which is 2d dimensional, we
        # need to modify y accordingly.
        if y.ndim == 1:
            y_2d = y.reshape(-1, 1)
        else:
            y_2d = y
        self.resampler_ = clone(self.resampler)
        self.resampler_.fit(y_2d, X)

        # Transform y and convert back to 1d array if needed
        y_transformed = self.resampler_.transform(y_2d, X)
        # TODO Check if next if statement is necessary
        if y_transformed.ndim == 2 and y_transformed.shape[1] == 1:
            y_transformed = y_transformed.squeeze(axis=1)

        if self.classifier is None:
            from sklearn.linear_model import LogisticRegression
            self.classifier_ = LogisticRegression()
        else:
            self.classifier_ = clone(self.classifier)

        print(X.shape, y_transformed.shape)
        if sample_weight is None:
            self.classifier_.fit(X, y_transformed)
        else:
            self.classifier_.fit(X, y_transformed, sample_weight=sample_weight)

        return self

    def predict(self, X):
        """Predict using the base classifier.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_1, ...)
            Samples.
        Returns
        -------
        y_hat : array, shape = (n_samples,)
            Predicted values.
        """
        check_is_fitted(self, "classifier_")
        yhat = self.classifier_.predict(X)
        return yhat

    def predict_proba(self, X, **kwargs):
        """Returns class probability estimates for the given test data.

        Parameters
        ----------
            X: array-like, shape `(n_samples, n_1, ...)`
                Test samples where `n_samples` is the number of samples
                and the total number of features is the product of all n_i's.
            **kwargs: dictionary arguments
                Legal arguments are the arguments
                of `Sequential.predict_classes`.

        Returns
        -------
            proba: array-like, shape `(n_samples, n_outputs)`
                Class probability estimates.
                In the case of binary classification,
                to match the scikit-learn API,
                will return an array of shape `(n_samples, 2)`
                (instead of `(n_sample, 1)` as in Keras).
        """
        check_is_fitted(self, "classifier_")
        return self.classifier_.predict_proba(X)

    def score(self, X, y=None, **kwargs):
        if y.ndim == 1:
            y_2d = y.reshape(-1, 1)
        else:
            y_2d = y
        y_transformed = self.resampler_.transform(y_2d, X)
        return self.classifier_.score(X, y_transformed, **kwargs)


class TargetResamplingRegressor(BaseEstimator, RegressorMixin):
    """Meta-estimator to regress on a resampled target.
    Useful in regression problems whenever the correct correspondence between
    the feature and target arrays is obtained via resampling of the target array
    (in a way which might depend on the feature array).
    The computation during ``fit`` is::
        regressor.fit(X, resampler.transform(y, X)).
    The computation during ``predict`` is::
        regressor.predict(X)).
    Parameters
    ----------
    regressor : object, default=LinearRegression()
        Regressor object such as derived from ``RegressorMixin``. This
        regressor will automatically be cloned each time prior to fitting.
    resampler : object, default=None
        Estimator object such as derived from ``TransformerMixin``. Note that
        this resampler will be cloned during fitting.
    Attributes
    ----------
    regressor_ : object
        Fitted regressor.
    resampler_ : object
        Clone of the resampler used in ``fit``.
    Examples
    --------
    >>> import numpy as np
    >>> import topological_learning as tl
    >>> from sklearn.linear_model import LinearRegression
    >>> from tl.compose import TargetResampler, TargetResamplingRegressor
    >>> ss = 2
    >>> res = TargetResampler(step_size=ss)
    >>> trc = TargetResamplingRegressor(regressor=LinearRegression(),
    ...                                 resampler=res)
    >>> X = np.arange(4).reshape(-1,ss)
    >>> y = np.arange(3)
    >>> trc.fit(X, y) # doctest: +ELLIPSIS
    TargetResamplingRegressor(...)
    >>> trc.score(X, y)
    1.0
    >>> trc.regressor_.coef_
    array([0.25, 0.25])
    Notes
    -----
    Internally, the target ``y`` is always converted into a 2-dimensional array
    to be used by scikit-learn transformers. At the time of prediction, the
    output will be reshaped to a have the same number of dimensions as ``y``.
    """
    def __init__(self, regressor=None, resampler=None):
        self.regressor = regressor
        self.resampler = resampler

    def get_params(self, deep=True):
        return {'regressor': self.regressor, 'resampler': self.resampler}

    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_1, ...)
            Training array, where n_samples is the number of samples and
            the total number of features is the product of all n_i's.
        y : array-like, shape (n_samples,)
            Target values.
        sample_weight : array-like, shape (n_samples,) optional
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.
        Returns
        -------
        self : object
        """
        y = check_array(y, accept_sparse=False, force_all_finite=True,
                        ensure_2d=False, dtype='numeric')

        # store the number of dimension of the target to predict an array of
        # similar shape at predict
        self._training_dim = y.ndim

        # transformers are designed to modify X which is 2d dimensional, we
        # need to modify y accordingly.
        if y.ndim == 1:
            y_2d = y.reshape(-1, 1)
        else:
            y_2d = y
        self.resampler_ = clone(self.resampler)
        self.resampler_.fit(y_2d, X)

        # transform y and convert back to 1d array if needed
        y_transformed = self.resampler_.transform(y_2d, X)
        # FIXME: a FunctionTransformer can return a 1D array even when validate
        # is set to True. Therefore, we need to check the number of dimension
        # first.
        if y_transformed.ndim == 2 and y_transformed.shape[1] == 1:
            y_transformed = y_transformed.squeeze(axis=1)

        if self.regressor is None:
            from sklearn.linear_model import LinearRegression
            self.regressor_ = LinearRegression()
        else:
            self.regressor_ = clone(self.regressor)

        if sample_weight is None:
            self.regressor_.fit(X, y_transformed)
        else:
            self.regressor_.fit(X, y_transformed, sample_weight=sample_weight)

        return self

    def predict(self, X):
        """Predict using the base regressor.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_1, ...)
            Samples.
        Returns
        -------
        y_hat : array, shape = (n_samples,)
            Predicted values.
        """
        check_is_fitted(self, "regressor_")
        yhat = self.regressor_.predict(X)
        return yhat


class TargetResampler(BaseEstimator, TransformerMixin):
    """Simple resampler subsampling data at regular steps, up to a maximum
    number of samples equal to the length of another array. Convenience object
    adapted for use as a resampler inside a TargetResamplingClassifier.
    Parameters
    ----------
    step_size : int, default=1
        The difference between index values of consecutively sampled values.
    from_right : bool, default=True
        When True, we ensure that the last entry is sampled. When False, we
        ensure that the first entry is sampled.
    """
    def __init__(self, step_size=1, from_right=True):
        self.step_size = step_size
        self.from_right = from_right

    def get_params(self, deep=True):
        return {'step_size': self.step_size, 'from_right': self.from_right}

    def _validate_params(self):
        pass

    def _get_indices(self, y, max_num):
        """Obtain indices for resampling of y, up to a maximum number max_num
        of samples. Called by self.transform().
        Parameters
        ----------
        y : array-like
            Array of target values.
        max_num : int
            Maximum number of indices to be returned
        Returns
        -------
        indices : list
            List of index values. Will be modified in self.transform() if
            self.from_right is True
        """
        all_indices = list(range(0, len(y), self.step_size))
        indices = all_indices[:max_num]
        return indices

    def fit(self, y, X=None):
        """Fit the resampler.
        Parameters
        ----------
        y : array-like
            Array of target values.
        X : {array-like, sparse matrix}, default=None
            Reference array, typically a feature array.
        Returns
        -------
        self : object
        """
        self._validate_params()

        self._is_fitted = True
        return self

    def transform(self, y, X):
        """Resample y according to the length of reference array X, self.step_size
        and self.from_right.
        Parameters
        ----------
        y : array-like
            Array of target values.
        X : {array-like, sparse matrix}
            Reference array, typically a feature array.
        Returns
        -------
        y_res : array-like
            Resampling of y.
        """
        check_is_fitted(self, "_is_fitted")

        indices = self._get_indices(y, len(X))
        if len(X) > len(indices):
            raise ValueError('Target array cannot be resampled to have the same\
                              length as reference array')
        offset = int(self.from_right)*(len(y) - 1 - indices[-1])
        indices = np.asarray(indices) + offset
        y_res = y[indices]
        return y_res
