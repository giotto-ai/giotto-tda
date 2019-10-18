"""The module :mod:`giotto.pipeline` extends scikit-learn's module by defining
Pipelines that include TransformerResamplers.
"""
# License: Apache 2.0

import numpy as np

from sklearn import pipeline
from sklearn.base import clone
from sklearn.base import BaseEstimator
from .base import TransformerResamplerMixin
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import check_memory

__all__ = ['Pipeline', 'make_pipeline']


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
