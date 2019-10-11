"""
The module :mod:`giotto.pipeline` is an extension of scikit-learn's
and implements utilities to build a composite transformer by applying
a transformer pipeline to sliding subwindows of the input data in the
spirit of a pooling layer in a convolutional neural network.
"""
# Adapted from scikit-learn and imbalanced-learn
# License: Apache 2.0

import numpy as np

from sklearn import pipeline
from sklearn.base import clone
from sklearn.base import BaseEstimator
from .base import TransformerResamplerMixin
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import check_memory

__all__ = ['Pipeline', 'make_pipeline', 'SlidingWindowFeatureUnion']


class Pipeline(pipeline.Pipeline):
    """Pipeline of transforms and resamples with a final estimator.

    Sequentially apply a list of transforms, sampling, and a final estimator.
    Intermediate steps of the pipeline must be transformers or resamplers,
    that is, they must implement fit, transform and sample methods.
    The samplers are only applied during fit.
    The final estimator only needs to implement fit.
    The transformers and samplers in the pipeline can be cached using
    ``memory`` argument.

    The purpose of the pipeline is to assemble several steps that can be
    cross-validated together while setting different parameters.
    For this, it enables setting parameters of the various steps using their
    names and the parameter name separated by a '__', as in the example below.
    A step's estimator may be replaced entirely by setting the parameter
    with its name to another estimator, or a transformer removed by setting
    it to 'passthrough' or ``None``.

    Parameters
    ----------
    steps : list
        List of (name, transform) tuples (implementing
        fit/transform) that are chained, in the order in which
        they are chained, with the last object an estimator.

    memory : Instance of joblib.Memory or string, optional (default=None)
        Used to cache the fitted transformers of the pipeline. By default,
        no caching is performed. If a string is given, it is the path to
        the caching directory. Enabling caching triggers a clone of
        the transformers before fitting. Therefore, the transformer
        instance given to the pipeline cannot be inspected
        directly. Use the attribute ``named_steps`` or ``steps`` to
        inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.


    Attributes
    ----------
    named_steps : dict
        Read-only attribute to access any step parameter by user given name.
        Keys are step names and values are steps parameters.

    See also
    --------
    make_pipeline : helper function to make pipeline.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split as tts
    >>> from sklearn.decomposition import PCA
    >>> from sklearn.neighbors import KNeighborsClassifier as KNN
    >>> from sklearn.metrics import classification_report
    >>> from imblearn.over_sampling import SMOTE
    >>> from imblearn.pipeline import Pipeline # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape {}'.format(Counter(y)))
    Original dataset shape Counter({1: 900, 0: 100})
    >>> pca = PCA()
    >>> smt = SMOTE(random_state=42)
    >>> knn = KNN()
    >>> pipeline = Pipeline([('smt', smt), ('pca', pca), ('knn', knn)])
    >>> X_train, X_test, y_train, y_test = tts(X, y, random_state=42)
    >>> pipeline.fit(X_train, y_train) # doctest: +ELLIPSIS
    Pipeline(...)
    >>> y_hat = pipeline.predict(X_test)
    >>> print(classification_report(y_test, y_hat))
                  precision    recall  f1-score   support
    <BLANKLINE>
               0       0.87      1.00      0.93        26
               1       1.00      0.98      0.99       224
    <BLANKLINE>
        accuracy                           0.98       250
       macro avg       0.93      0.99      0.96       250
    weighted avg       0.99      0.98      0.98       250
    <BLANKLINE>

    """

    def _fit(self, X, y=None, **fit_params):
        self.steps = list(self.steps)
        self._validate_steps()
        # Setup the memory
        memory = check_memory(self.memory)

        fit_transform_one_cached = memory.cache(_fit_transform_one)
        fit_transform_resample_one_cached = memory.cache(
            _fit_transform_resample_one)

        fit_params_steps = {name: {} for name, step in self.steps
                            if step is not None}
        for pname, pval in fit_params.items():
            step, param = pname.split('__', 1)
            fit_params_steps[step][param] = pval
        for step_idx, name, transformer in self._iter(with_final=False):
            if hasattr(memory, 'location'):
                # joblib >= 0.12
                if memory.location is None:
                    # we do not clone when caching is disabled to
                    # preserve backward compatibility
                    cloned_transformer = transformer
                else:
                    cloned_transformer = clone(transformer)
            elif hasattr(memory, 'cachedir'):
                # joblib < 0.11
                if memory.cachedir is None:
                    # we do not clone when caching is disabled to
                    # preserve backward compatibility
                    cloned_transformer = transformer
            else:
                cloned_transformer = clone(transformer)
            # Fit or load from cache the current transfomer
            if (hasattr(cloned_transformer, "resample") or
                hasattr(cloned_transformer, "fit_transform_resample")):
                if y is None:
                    X, fitted_transformer = fit_transform_one_cached(
                        cloned_transformer, None, X, y,
                        **fit_params_steps[name])
                else:
                    X, y, fitted_transformer = \
                        fit_transform_resample_one_cached(
                            cloned_transformer, None, X, y,
                            **fit_params_steps[name])
            else:
                X, fitted_transformer = fit_transform_one_cached(
                    cloned_transformer, None, X, y,
                    **fit_params_steps[name])

            # Replace the transformer of the step with the fitted
            # transformer. This is necessary when loading the transformer
            # from the cache.
            self.steps[step_idx] = (name, fitted_transformer)
        if self._final_estimator == 'passthrough':
            return X, y, {}
        return X, y, fit_params_steps[self.steps[-1][0]]

    def fit(self, X, y=None, **fit_params):
        """Fit the model

        Fit all the transforms/samplers one after the other and
        transform/sample the data, then fit the transformed/sampled
        data using the final estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the :meth:`fit` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        self : Pipeline
            This estimator

        """
        Xt, yt, fit_params = self._fit(X, y, **fit_params)
        if self._final_estimator != 'passthrough':
            self._final_estimator.fit(Xt, yt, **fit_params)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit the model and transform with the final estimator

        Fits all the transformers/samplers one after the other and
        transform/sample the data, then uses fit_transform on
        transformed data with the final estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the :meth:`fit` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        Xt : array-like, shape = [n_samples, n_transformed_features]
            Transformed samples

        """
        last_step = self._final_estimator
        Xt, yt, fit_params = self._fit(X, y, **fit_params)
        if last_step == 'passthrough':
            return Xt
        elif hasattr(last_step, 'fit_transform'):
            return last_step.fit_transform(Xt, yt, **fit_params)
        else:
            return last_step.fit(Xt, yt, **fit_params).transform(Xt)

    def fit_transform_resample(self, X, y=None, **fit_params):
        """Fit the model and sample with the final estimator
        Fits all the transformers/samplers one after the other and
        transform/sample the data, then uses fit_resample on transformed
        data with the final estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.
        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.
        **fit_params : dict of string -> object
            Parameters passed to the :meth:`fit` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        Xt : array-like, shape = [n_samples, n_transformed_features]
            Transformed samples
        yt : array-like, shape = [n_samples, n_transformed_features]
            Transformed target
        """
        last_step = self._final_estimator
        Xt, yt, fit_params = self._fit(X, y, **fit_params)
        if last_step == 'passthrough':
            return Xt, yt
        elif hasattr(last_step, 'fit_transform_resample'):
            return last_step.fit_transform_resample(Xt, yt, **fit_params)
        elif hasattr(last_step, 'fit_transform'):
            return last_step.fit_transform(Xt, yt, **fit_params), yt

    @if_delegate_has_method(delegate='_final_estimator')
    def fit_predict(self, X, y=None, **fit_params):
        """Applies fit_predict of last step in pipeline after transforms.

        Applies fit_transforms of a pipeline to the data, followed by the
        fit_predict method of the final estimator in the pipeline. Valid
        only if the final estimator implements fit_predict.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of
            the pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps
            of the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the :meth:`fit` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        y_pred : array-like
        """
        Xt, yt, fit_params = self._fit(X, y, **fit_params)
        return self.steps[-1][-1].fit_predict(Xt, yt, **fit_params)

    @property
    def resample(self):
        """Apply transformers/transformer_resamplers, and transform with the
        final estimator.

        This also works where final estimator is ``None``: all prior
        transformations are applied.

        Parameters
        ----------
        X : iterable
            Data to transform. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        Xt : array-like, shape = [n_samples, n_transformed_features]
        """
        # _final_estimator is None or has transform, otherwise attribute error
        if self._final_estimator != 'passthrough':
            self._final_estimator.resample
        return self._resample

    def _resample(self, X, y=None):
        Xt, yt = X, y
        for _, _, transform in self._iter():
            yt =  transform.resample(yt)
        return yt


    @property
    def transform_resample(self):
        """Apply transformers/transformer_resamplers, and transform with the
        final estimator.

        This also works where final estimator is ``None``: all prior
        transformations are applied.

        Parameters
        ----------
        X : iterable
            Data to transform. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        Xt : array-like, shape = [n_samples, n_transformed_features]
        """
        # _final_estimator is None or has transform, otherwise attribute error
        final_estimator = self._final_estimator
        if final_estimator != 'passthrough':
            if hasattr(final_estimator, 'transform_resample'):
                final_estimator.transform_resample
            else:
                final_estimator.transform
        return self._transform_resample

    def _transform_resample(self, X, y):
        Xt, yt = X, y
        for _, _, transform in self._iter():
            if hasattr(transform, 'transform_resample'):
                Xt, yt = transform.transform_resample(Xt, yt)
            else:
                Xt = transform.transform(Xt)
        return Xt, yt

    @property
    def transform(self):
        """Apply transformers/transformer_resamplers, and transform with the
        final estimator.

        This also works where final estimator is ``None``: all prior
        transformations are applied.

        Parameters
        ----------
        X : iterable
            Data to transform. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        Xt : array-like, shape = [n_samples, n_transformed_features]
        """
        # _final_estimator is None or has transform, otherwise attribute error
        if self._final_estimator != 'passthrough':
            self._final_estimator.transform
        return self._transform

    def _transform(self, X, y=None):
        Xt, yt = X, y
        for _, _, transform in self._iter():
            Xt =  transform.transform(Xt)
        return Xt

    @property
    def inverse_transform(self):
        """Apply inverse transformations in reverse order

        All estimators in the pipeline must support ``inverse_transform``.

        Parameters
        ----------
        Xt : array-like, shape = [n_samples, n_transformed_features]
            Data samples, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features. Must fulfill
            input requirements of last step of pipeline's
            ``inverse_transform`` method.

        Returns
        -------
        Xt : array-like, shape = [n_samples, n_features]
        """
        # raise AttributeError if necessary for hasattr behaviour
        for _, _, transform in self._iter():
            transform.inverse_transform
        return self._inverse_transform

    def _inverse_transform(self, X, y=None):
        Xt, yt = X, y
        reverse_iter = reversed(list(self._iter()))
        for _, _, transform in self._iter():
            Xt = transform.inverse_transform(Xt, yt)
        return Xt

    @if_delegate_has_method(delegate='_final_estimator')
    def score(self, X, y=None, sample_weight=None):
        """Apply transformers/samplers, and score with the final estimator

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        y : iterable, default=None
            Targets used for scoring. Must fulfill label requirements for all
            steps of the pipeline.

        sample_weight : array-like, default=None
            If not None, this argument is passed as ``sample_weight`` keyword
            argument to the ``score`` method of the final estimator.

        Returns
        -------
        score : float
        """
        Xt, yt = X, y
        for _, _, transform in self._iter(with_final=False):
            if (hasattr(transform, "transform_resample")):
                Xt, yt = transform.transform_resample(Xt, yt)
            else:
                Xt = transform.transform(Xt)

        score_params = {}
        if sample_weight is not None:
            score_params['sample_weight'] = sample_weight
        return self.steps[-1][-1].score(Xt, yt, **score_params)


def _fit_transform_one(transformer, weight, X, y, **fit_params):
    if hasattr(transformer, 'fit_transform'):
        X_res = transformer.fit_transform(X, y, **fit_params)
    else:
        X_res = transformer.fit(X, y, **fit_params).transform(X)
    # if we have a weight for this transformer, multiply output
    if weight is None:
        return X_res, transformer
    return X_res * weight, transformer


def _fit_transform_resample_one(transformer_resampler, weight,
                                X, y, **fit_params):
    if hasattr(transformer_resampler, 'fit_transform_resample'):
        X_res, y_res = transformer_resampler.fit_transform_resample(
        X, y, **fit_params)
    else:
        X_res, y_res = transformer_resampler.fit(
            X, y,**fit_params).transform_resample(
            X, y)
    if weight is None:
        return X_res, y_res, transformer_resampler
    return X_res * weight, y_res, transformer_resampler


def make_pipeline(*steps, **kwargs):
    """Construct a Pipeline from the given estimators.

    This is a shorthand for the Pipeline constructor; it does not require, and
    does not permit, naming the estimators. Instead, their names will be set
    to the lowercase of their types automatically.

    Parameters
    ----------
    *steps : list of estimators.

    memory : None, str or object with the joblib.Memory interface, optional
        Used to cache the fitted transformers of the pipeline. By default,
        no caching is performed. If a string is given, it is the path to
        the caching directory. Enabling caching triggers a clone of
        the transformers before fitting. Therefore, the transformer
        instance given to the pipeline cannot be inspected
        directly. Use the attribute ``named_steps`` or ``steps`` to
        inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.

    Returns
    -------
    p : Pipeline

    See also
    --------
    imblearn.pipeline.Pipeline : Class for creating a pipeline of
        transforms with a final estimator.

    Examples
    --------
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.preprocessing import StandardScaler
    >>> make_pipeline(StandardScaler(), GaussianNB(priors=None))
    ... # doctest: +NORMALIZE_WHITESPACE
    Pipeline(memory=None,
             steps=[('standardscaler',
                     StandardScaler(copy=True, with_mean=True, with_std=True)),
                    ('gaussiannb',
                     GaussianNB(priors=None, var_smoothing=1e-09))],
             verbose=False)
    """
    memory = kwargs.pop('memory', None)
    if kwargs:
        raise TypeError('Unknown keyword arguments: "{}"'
                        .format(list(kwargs.keys())[0]))
    return Pipeline(pipeline._name_estimators(steps), memory=memory)


class SlidingWindowFeatureUnion(BaseEstimator, TransformerResamplerMixin):
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

    axes: list of int, optional (default=None)
        Axes on which to slide the window.

    width: list of int, optional (default=None)
        Width of the sliding window.

    stride: list of int, default: None
        Stride of the sliding window.

    padding: list of int, optional (default=None)
        Padding applied to the input before sliding the window.

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
    def __init__(self, transformer, axes=[0], width=None, stride=None,
                 padding=None, n_jobs=None):
        self.transformer = transformer
        self.axes = axes
        self.width = width
        self.stride = stride
        self.padding = padding
        self.n_jobs = n_jobs

    def _validate_params(self):
        """A class method that checks whether the hyperparameters and the input
        parameters of the :meth:`fit` are valid.

        """
        try:
            assert self._dimension == len(self.width_)
            assert self._dimension == len(self.stride_)
            assert self._dimension == len(self.padding_)
        except AssertionError:
            raise ValueError("axes, width, stride, and padding do not have the same"
                             " length.")

        if len(self.axes) != 1 or self.axes[0] != 0:
            raise NotImplementedError("This transformer has only been"
                                      " implemented for time series"
                                      " for which axes = [0]")

    def _pad(self, X, axis):
        return X

    def _view_X(X, begin, end, axis):
        return np.roll(X, shift=-begin, axis=axis)[:end]

    def _view_y(y, begin, end, axis):
        return np.roll(y, shift=-begin, axis=axis)[:end]

    def _parallel_func(self, X, y, fit_params, func, window_slices):
        """Runs func in parallel on X and y"""
        return Parallel(n_jobs=self.n_jobs)(
            delayed(func)(transformer, self._view_X(_X, begin, end, axis),
                          self._view_y(y, axis, begin, end, axis))
            for begin, end in window_slices)

    def _slice_windows(self):
        n_windows = [
            (X.shape[self.axes[dimension]] - 2*self.width_[dimension]+1) \
            // self._stride[dimension] + 1
            for dimension in range(self._dimension)
            ]

        window_slices = [
            tuple(i*self.stride_[dimension],
                  2*self.width_[dimension]+1 + i*self.stride_[dimension])
            for i in range(n_windows[dimension])
        ]
        return window_slices

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
        self._dimension = len(axes)

        if self.width is None:
            self.width_ = self._dimension * [1]
        else:
            self.width_ = self.width

        if self.stride is None:
            self.stride_ = self._dimension * [1]
        else:
            self.stride_ = self.stride

        if self.padding is None:
            self.padding_ =self._dimension * [0]
        else:
            self.width_ = self.width

        self._validate_params()
        _X = self._pad(X)

        window_slices = self._slice_windows()
        window_transformers = [clone(self.transformer) for _ in range(2)]
        fit = self._parallel_func(_X, y, {}, fit, window_slices)
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
        _X = self._pad(X)

        fit = self._parallel_func(_X, y, {}, fit)

        Xs, transformers = zip(*results)
        self._update_transformer_list(transformers)

        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = np.hstack(Xs)
        return Xs

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
