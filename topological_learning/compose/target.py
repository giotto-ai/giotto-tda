# Authors: Umberto Lupo <u.lupo@l2f.ch>
#          Guillaume Tauzin <guillaume.tauzin@epfl.ch>
# License: BSD 3 clause

import warnings

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, safe_indexing


class TargetResamplingClassifier(BaseEstimator, ClassifierMixin):
    """Meta-estimator to regress on a transformed target.
    Useful for applying a non-linear transformation in regression
    problems. This transformation can be given as a Transformer such as the
    QuantileTransformer or as a function and its inverse such as ``log`` and
    ``exp``.
    The computation during ``fit`` is::
        classifier.fit(X, func(y))
    or::
        classifier.fit(X, transformer.transform(y))
    The computation during ``predict`` is::
        inverse_func(classifier.predict(X))
    or::
        transformer.inverse_transform(classifier.predict(X))
    Read more in the :ref:`User Guide <preprocessing_targets>`.
    Parameters
    ----------
    classifier : object, default=LinearRegression()
        Classifier object such as derived from ``ClassifierMixin``. This
        classifier will automatically be cloned each time prior to fitting.
    transformer : object, default=None
        Estimator object such as derived from ``TransformerMixin``. Cannot be
        set at the same time as ``func`` and ``inverse_func``. If
        ``transformer`` is ``None`` as well as ``func`` and ``inverse_func``,
        the transformer will be an identity transformer. Note that the
        transformer will be cloned during fitting. Also, the transformer is
        restricting ``y`` to be a numpy array.
    func : function, optional
        Function to apply to ``y`` before passing to ``fit``. Cannot be set at
        the same time as ``transformer``. The function needs to return a
        2-dimensional array. If ``func`` is ``None``, the function used will be
        the identity function.
    inverse_func : function, optional
        Function to apply to the prediction of the classifier. Cannot be set at
        the same time as ``transformer`` as well. The function needs to return
        a 2-dimensional array. The inverse function is used to return
        predictions to the same space of the original training labels.
    check_inverse : bool, default=True
        Whether to check that ``transform`` followed by ``inverse_transform``
        or ``func`` followed by ``inverse_func`` leads to the original targets.
    Attributes
    ----------
    classifier_ : object
        Fitted classifier.
    transformer_ : object
        Transformer used in ``fit`` and ``predict``.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.compose import TransformedTargetClassifier
    >>> tt = TransformedTargetClassifier(classifier=LinearRegression(),
    ...                                 func=np.log, inverse_func=np.exp)
    >>> X = np.arange(4).reshape(-1, 1)
    >>> y = np.exp(2 * X).ravel()
    >>> tt.fit(X, y) # doctest: +ELLIPSIS
    TransformedTargetClassifier(...)
    >>> tt.score(X, y)
    1.0
    >>> tt.classifier_.coef_
    array([2.])
    Notes
    -----
    Internally, the target ``y`` is always converted into a 2-dimensional array
    to be used by scikit-learn transformers. At the time of prediction, the
    output will be reshaped to a have the same number of dimensions as ``y``.
    See :ref:`examples/compose/plot_transformed_target.py
    <sphx_glr_auto_examples_compose_plot_transformed_target.py>`.
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
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
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

        if self.classifier is None:
            from sklearn.linear_model import LinearRegression
            self.classifier_ = LinearRegression()
        else:
            self.classifier_ = self.classifier # TO DO clone(self.classifier)

        print(X.shape, y_transformed.shape)
        if sample_weight is None:
            self.classifier_.fit(X, y_transformed)
        else:
            self.classifier_.fit(X, y_transformed, sample_weight=sample_weight)

        return self

    def predict(self, X):
        """Predict using the base classifier, applying inverse.
        The classifier is used to predict and the ``inverse_func`` or
        ``inverse_transform`` is applied before returning the prediction.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.
        Returns
        -------
        y_hat : array, shape = (n_samples,)
            Predicted values.
        """
        check_is_fitted(self, "classifier_")
        print(X.shape)
        y = self.classifier_.predict(X)
        print('predict: ', y.shape)

    def predict_proba(self, X, **kwargs):
        """Returns class probability estimates for the given test data.

        Parameters
        ----------
            X: array-like, shape `(n_samples, n_features)`
                Test samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
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
    """Meta-estimator to regress on a transformed target.
    Useful for applying a non-linear transformation in regression
    problems. This transformation can be given as a Transformer such as the
    QuantileTransformer or as a function and its inverse such as ``log`` and
    ``exp``.
    The computation during ``fit`` is::
        regressor.fit(X, func(y))
    or::
        regressor.fit(X, transformer.transform(y))
    The computation during ``predict`` is::
        inverse_func(regressor.predict(X))
    or::
        transformer.inverse_transform(regressor.predict(X))
    Read more in the :ref:`User Guide <preprocessing_targets>`.
    Parameters
    ----------
    regressor : object, default=LinearRegression()
        Regressor object such as derived from ``RegressorMixin``. This
        regressor will automatically be cloned each time prior to fitting.
    transformer : object, default=None
        Estimator object such as derived from ``TransformerMixin``. Cannot be
        set at the same time as ``func`` and ``inverse_func``. If
        ``transformer`` is ``None`` as well as ``func`` and ``inverse_func``,
        the transformer will be an identity transformer. Note that the
        transformer will be cloned during fitting. Also, the transformer is
        restricting ``y`` to be a numpy array.
    func : function, optional
        Function to apply to ``y`` before passing to ``fit``. Cannot be set at
        the same time as ``transformer``. The function needs to return a
        2-dimensional array. If ``func`` is ``None``, the function used will be
        the identity function.
    inverse_func : function, optional
        Function to apply to the prediction of the regressor. Cannot be set at
        the same time as ``transformer`` as well. The function needs to return
        a 2-dimensional array. The inverse function is used to return
        predictions to the same space of the original training labels.
    check_inverse : bool, default=True
        Whether to check that ``transform`` followed by ``inverse_transform``
        or ``func`` followed by ``inverse_func`` leads to the original targets.
    Attributes
    ----------
    regressor_ : object
        Fitted regressor.
    transformer_ : object
        Transformer used in ``fit`` and ``predict``.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.compose import TransformedTargetRegressor
    >>> tt = TransformedTargetRegressor(regressor=LinearRegression(),
    ...                                 func=np.log, inverse_func=np.exp)
    >>> X = np.arange(4).reshape(-1, 1)
    >>> y = np.exp(2 * X).ravel()
    >>> tt.fit(X, y) # doctest: +ELLIPSIS
    TransformedTargetRegressor(...)
    >>> tt.score(X, y)
    1.0
    >>> tt.regressor_.coef_
    array([2.])
    Notes
    -----
    Internally, the target ``y`` is always converted into a 2-dimensional array
    to be used by scikit-learn transformers. At the time of prediction, the
    output will be reshaped to a have the same number of dimensions as ``y``.
    See :ref:`examples/compose/plot_transformed_target.py
    <sphx_glr_auto_examples_compose_plot_transformed_target.py>`.
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
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
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
        """Predict using the base regressor, applying inverse.
        The regressor is used to predict and the ``inverse_func`` or
        ``inverse_transform`` is applied before returning the prediction.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.
        Returns
        -------
        y_hat : array, shape = (n_samples,)
            Predicted values.
        """
        check_is_fitted(self, "regressor_")
        return self.regressor_.predict(X)


class TargetResampler(BaseEstimator, TransformerMixin):
    """ Simple transformer subsampling data at regular steps, up to a maximum
    number of samples.
    """
    def __init__(self, step_size=None, from_right=True):
        self.step_size = step_size
        self.from_right = from_right

    def get_params(self, deep=True):
        return {'step_size': self.step_size, 'from_right': self.from_right}

    @staticmethod
    def _validate_params():
        pass

    def _get_indices(self, y, X):
        if self.from_right:
            all_indices = list(range(len(y) - 1, -1, -self.step_size))
        else:
            all_indices = list(range(0, len(y), self.step_size))
        self.max_num = len(X)
        return all_indices[:self.max_num]

    def fit(self, y, X=None):
        self._validate_params()

        self.is_fitted = True
        return self

    def transform(self, y, X):
        check_is_fitted(self, "is_fitted")

        indices = self._get_indices(y, X)
        return y[indices]
