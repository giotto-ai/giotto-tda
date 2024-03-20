"""The module :mod:`gtda.pipeline` extends scikit-learn's module by defining
Pipelines that include TransformerResamplers."""
# License: GNU AGPLv3

from sklearn import pipeline
from sklearn.base import clone
from sklearn.utils.metaestimators import available_if
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

    memory : Instance of joblib.Memory or string, optional (default: ``None``)
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
    >>> import numpy as np
    >>> import gtda.time_series as ts
    >>> import gtda.homology as hl
    >>> import gtda.diagrams as diag
    >>> from gtda.pipeline import Pipeline
    >>> import sklearn.preprocessing as skprep
    >>>
    >>> X = np.random.rand(600, 1)
    >>> n_train, n_test = 400, 200
    >>>
    >>> labeller = ts.Labeller(size=6, percentiles=[80],
    >>>                        n_steps_future=1)
    >>> X_train = X[:n_train]
    >>> y_train = X_train
    >>> X_train, y_train = labeller.fit_transform_resample(X_train, y_train)
    >>>
    >>> print(X_train.shape, y_train.shape)
    (395, 1) (395,)
    >>> steps = [
    >>>     ('embedding', ts.SingleTakensEmbedding()),
    >>>     ('window', ts.SlidingWindow(size=6, stride=1)),
    >>>     ('diagram', hl.VietorisRipsPersistence()),
    >>>     ('rescaler', diag.Scaler()),
    >>>     ('filter', diag.Filtering(epsilon=0.1)),
    >>>     ('entropy', diag.PersistenceEntropy()),
    >>>     ('scaling', skprep.MinMaxScaler(copy=True)),
    >>> ]
    >>> pipeline = Pipeline(steps)
    >>>
    >>> Xt_train, yr_train = pipeline.\\
    >>>     fit_transform_resample(X_train, y_train)
    >>>
    >>> print(X_train_final.shape, y_train_final.shape)
    (389, 2) (389,)
    """

    def _final_estimator_has(attr):
        def check(self):
            return hasattr(self._final_estimator, attr)

        return check

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
            if hasattr(memory, 'location') and (memory.location is None):
                # joblib >= 0.12. We do not clone when caching is disabled to
                # preserve backward compatibility
                cloned_transformer = transformer
            else:
                cloned_transformer = clone(transformer)
            # Fit or load from cache the current transfomer
            if hasattr(cloned_transformer, "resample") or \
               hasattr(cloned_transformer, "fit_transform_resample"):
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
        """Fit the model.

        Fit all the transforms/samplers one after the other and
        transform/sample the data, then fit the transformed/sampled
        data using the final estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable or None, default: ``None``
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
        Xt, yr, fit_params = self._fit(X, y, **fit_params)
        if self._final_estimator != 'passthrough':
            self._final_estimator.fit(Xt, yr, **fit_params)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit the model and transform with the final estimator.

        Fits all the transformers/samplers one after the other and
        transform/sample the data, then uses fit_transform on
        transformed data with the final estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default: ``None``
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the :meth:`fit` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        Xt : array-like, shape (n_samples, n_transformed_features)
            Transformed samples

        """
        last_step = self._final_estimator
        Xt, yr, fit_params = self._fit(X, y, **fit_params)
        if last_step == 'passthrough':
            return Xt
        elif hasattr(last_step, 'fit_transform'):
            return last_step.fit_transform(Xt, yr, **fit_params)
        else:
            return last_step.fit(Xt, yr, **fit_params).transform(Xt)

    def fit_transform_resample(self, X, y=None, **fit_params):
        """Fit the model and sample with the final estimator.

        Fits all the transformers/samplers one after the other and
        transform/sample the data, then uses fit_resample on transformed
        data with the final estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default: ``None``
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the :meth:`fit` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        Xt : array-like, shape (n_samples, n_transformed_features)
            Transformed samples.

        yr : array-like, shape (n_samples, n_transformed_features)
            Transformed target.
        """
        last_step = self._final_estimator
        Xt, yr, fit_params = self._fit(X, y, **fit_params)
        if last_step == 'passthrough':
            return Xt, yr
        elif hasattr(last_step, 'fit_transform_resample'):
            return last_step.fit_transform_resample(Xt, yr, **fit_params)
        elif hasattr(last_step, 'fit_transform'):
            return last_step.fit_transform(Xt, yr, **fit_params), yr

    @available_if(_final_estimator_has('fit_predict'))
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

        y : iterable or None, default: ``None``
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
        Xt, yr, fit_params = self._fit(X, y, **fit_params)
        return self.steps[-1][-1].fit_predict(Xt, yr, **fit_params)

    @property
    def resample(self):
        """Apply transformers/transformer_resamplers, and transform with the
        final estimator.

        This also works where final estimator is ``None``: all prior
        transformations are applied.

        Parameters
        ----------
        y : array-like, shape = (n_samples,)
            Data to resample. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        yr : array-like, shape = (n_samples_new,)
        """
        # _final_estimator is None or has transform, otherwise attribute error
        if self._final_estimator != 'passthrough':
            self._final_estimator.resample
        return self._resample

    def _resample(self, X, y=None):
        yr = y
        for _, _, transform in self._iter():
            yr = transform.resample(yr)
        return yr

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
        Xt : array-like, shape = (n_samples_new, n_transformed_features)
        yr : array-like, shape = (n_samples_new,)
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
        Xt, yr = X, y
        for _, _, transform in self._iter():
            if hasattr(transform, 'transform_resample'):
                Xt, yr = transform.transform_resample(Xt, yr)
            else:
                Xt = transform.transform(Xt)
        return Xt, yr

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
        Xt : array-like, shape (n_samples, n_transformed_features)
        """
        # _final_estimator is None or has transform, otherwise attribute error
        if self._final_estimator != 'passthrough':
            self._final_estimator.transform
        return self._transform

    def _transform(self, X, y=None):
        Xt = X
        for _, _, transform in self._iter():
            Xt = transform.transform(Xt)
        return Xt

    @property
    def inverse_transform(self):
        """Apply inverse transformations in reverse order

        All estimators in the pipeline must support ``inverse_transform``.

        Parameters
        ----------

        Xt : array-like, shape (n_samples, n_transformed_features)
            Data samples, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features. Must fulfill
            input requirements of last step of pipeline's
            ``inverse_transform`` method.

        Returns
        -------
        Xt : array-like, shape (n_samples, n_features)
        """
        # raise AttributeError if necessary for hasattr behaviour
        for _, _, transform in self._iter():
            transform.inverse_transform
        return self._inverse_transform

    def _inverse_transform(self, X, y=None):
        Xt, yr = X, y
        reverse_iter = reversed(list(self._iter()))
        for _, _, transform in reverse_iter:
            Xt = transform.inverse_transform(Xt, yr)
        return Xt

    @available_if(_final_estimator_has('score'))
    def score(self, X, y=None, sample_weight=None):
        """Apply transformers/samplers, and score with the final estimator

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        y : iterable or None, default: ``None``
            Targets used for scoring. Must fulfill label requirements for all
            steps of the pipeline.

        sample_weight : array-like or None, default: ``None``
            If not None, this argument is passed as ``sample_weight`` keyword
            argument to the ``score`` method of the final estimator.

        Returns
        -------
        score : float
        """
        Xt, yr = X, y
        for _, _, transform in self._iter(with_final=False):
            if (hasattr(transform, "transform_resample")):
                Xt, yr = transform.transform_resample(Xt, yr)
            else:
                Xt = transform.transform(Xt)

        score_params = {}
        if sample_weight is not None:
            score_params['sample_weight'] = sample_weight
        return self.steps[-1][-1].score(Xt, yr, **score_params)


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
            X, y, **fit_params).transform_resample(
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
        raise TypeError(
            f'Unknown keyword arguments: "{list(kwargs.keys())[0]}"')
    return Pipeline(pipeline._name_estimators(steps), memory=memory)
