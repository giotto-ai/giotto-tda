"""
The module :mod:`giotto.pipeline` is an extension of scikit-learn's
and implements utilities to build a composite transformer by applying
a transformer pipeline to sliding subwindows of the input data in the
spirit of a pooling layer in a convolutional neural network.
"""

# Adapted from scikit-learn
# Author: (Pipeline)
#         Edouard Duchesnay
#         Gael Varoquaux
#         Virgile Fritsch
#         Alexandre Gramfort
#         Lars Buitinck
#         Christos Aridas
#         Guillaume Lemaitre <g.lemaitre58@gmail.com>
#         (SlidingWindowFeatureUnion)
#         Guillaume Tauzin <guillaume.tauzin@epfl.ch>
# License: Apache 2.0


import functools
from collections import defaultdict
from itertools import islice

from sklearn import pipeline
from sklearn.base import clone
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
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
        fit/transform/fit_resample) that are chained, in the order in which
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

    Notes
    -----
    See :ref:`sphx_glr_auto_examples_pipeline_plot_pipeline_classification.py`

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

    # BaseEstimator interface

    def _validate_steps(self):
        names, estimators = zip(*self.steps)

        # validate names
        self._validate_names(names)

        # validate estimators
        transformers = estimators[:-1]
        estimator = estimators[-1]

        for t in transformers:
            if t is None or t == 'passthrough':
                continue
            if (not (hasattr(t, "fit") or
                     hasattr(t, "fit_transform") or
                     hasattr(t, "fit_resample")) or
                    not (hasattr(t, "transform") or
                         hasattr(t, "fit_resample"))):
                raise TypeError(
                    "All intermediate steps of the chain should "
                    "be estimators that implement fit and transform or "
                    "fit_resample (but not both) or be a string 'passthrough' "
                    "'%s' (type %s) doesn't)" % (t, type(t)))

            if (hasattr(t, "fit_resample") and (hasattr(t, "fit_transform") or
                                                hasattr(t, "transform"))):
                raise TypeError(
                    "All intermediate steps of the chain should "
                    "be estimators that implement fit and transform or sample."
                    " '%s' implements both)" % (t))

            if isinstance(t, pipeline.Pipeline):
                raise TypeError(
                    "All intermediate steps of the chain should not be"
                    " Pipelines")

        # We allow last estimator to be None as an identity transformation
        if (estimator is not None and estimator != 'passthrough'
                and not hasattr(estimator, "fit")):
            raise TypeError("Last step of Pipeline should implement fit or be "
                            "the string 'passthrough'. '%s' (type %s) doesn't"
                            % (estimator, type(estimator)))

    # Estimator interface

    def _fit(self, X, y=None, **fit_params):
        self.steps = list(self.steps)
        self._validate_steps()
        # Setup the memory
        memory = check_memory(self.memory)

        fit_transform_one_cached = memory.cache(_fit_transform_one)
        fit_resample_one_cached = memory.cache(_fit_resample_one)

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
            if (hasattr(cloned_transformer, "transform") or
                    hasattr(cloned_transformer, "fit_transform")):
                X, fitted_transformer = fit_transform_one_cached(
                    cloned_transformer, None, X, y,
                    **fit_params_steps[name])
            elif hasattr(cloned_transformer, "fit_resample"):
                X, y, fitted_transformer = fit_resample_one_cached(
                    cloned_transformer, X, y, **fit_params_steps[name])
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
            Parameters passed to the ``fit`` method of each step, where
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
            Parameters passed to the ``fit`` method of each step, where
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

    def fit_resample(self, X, y=None, **fit_params):
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
            Parameters passed to the ``fit`` method of each step, where
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
            return Xt
        elif hasattr(last_step, 'fit_resample'):
            return last_step.fit_resample(Xt, yt, **fit_params)

    @if_delegate_has_method(delegate='_final_estimator')
    def predict(self, X, **predict_params):
        """Apply transformers/samplers to the data, and predict with the final
        estimator

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        **predict_params : dict of string -> object
            Parameters to the ``predict`` called at the end of all
            transformations in the pipeline. Note that while this may be
            used to return uncertainties from some models with return_std
            or return_cov, uncertainties that are generated by the
            transformations in the pipeline are not propagated to the
            final estimator.

        Returns
        -------
        y_pred : array-like

        """
        Xt = X
        for _, _, transform in self._iter(with_final=False):
            if hasattr(transform, "fit_resample"):
                pass
            else:
                Xt = transform.transform(Xt)
        return self.steps[-1][-1].predict(Xt, **predict_params)

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
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        y_pred : array-like
        """
        Xt, yt, fit_params = self._fit(X, y, **fit_params)
        return self.steps[-1][-1].fit_predict(Xt, yt, **fit_params)

    @if_delegate_has_method(delegate='_final_estimator')
    def predict_proba(self, X):
        """Apply transformers/samplers, and predict_proba of the final
        estimator

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        y_proba : array-like, shape = [n_samples, n_classes]

        """
        Xt = X
        for _, _, transform in self._iter(with_final=False):
            if hasattr(transform, "fit_resample"):
                pass
            else:
                Xt = transform.transform(Xt)
        return self.steps[-1][-1].predict_proba(Xt)

    @if_delegate_has_method(delegate='_final_estimator')
    def score_samples(self, X):
        """Apply transforms, and score_samples of the final estimator.
        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.
        Returns
        -------
        y_score : ndarray, shape (n_samples,)
        """
        Xt = X
        for _, _, transformer in self._iter(with_final=False):
            if hasattr(transformer, "fit_resample"):
                pass
            else:
                Xt = transformer.transform(Xt)
        return self.steps[-1][-1].score_samples(Xt)

    @if_delegate_has_method(delegate='_final_estimator')
    def decision_function(self, X):
        """Apply transformers/samplers, and decision_function of the final
        estimator

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        y_score : array-like, shape = [n_samples, n_classes]

        """
        Xt = X
        for _, _, transform in self._iter(with_final=False):
            if hasattr(transform, "fit_resample"):
                pass
            else:
                Xt = transform.transform(Xt)
        return self.steps[-1][-1].decision_function(Xt)

    @if_delegate_has_method(delegate='_final_estimator')
    def predict_log_proba(self, X):
        """Apply transformers/samplers, and predict_log_proba of the final
        estimator

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        y_score : array-like, shape = [n_samples, n_classes]

        """
        Xt = X
        for _, _, transform in self._iter(with_final=False):
            if hasattr(transform, "fit_resample"):
                pass
            else:
                Xt = transform.transform(Xt)
        return self.steps[-1][-1].predict_log_proba(Xt)

    @property
    def transform(self):
        """Apply transformers/samplers, and transform with the final estimator

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

    def _transform(self, X):
        Xt = X
        for _, _, transform in self._iter():
            if hasattr(transform, "fit_resample"):
                pass
            else:
                Xt = transform.transform(Xt)
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

    def _inverse_transform(self, X):
        Xt = X
        reverse_iter = reversed(list(self._iter()))
        for _, _, transform in reverse_iter:
            if hasattr(transform, "fit_resample"):
                pass
            else:
                Xt = transform.inverse_transform(Xt)
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
        Xt = X
        for _, _, transform in self._iter(with_final=False):
            if hasattr(transform, "fit_resample"):
                pass
            else:
                Xt = transform.transform(Xt)
        score_params = {}
        if sample_weight is not None:
            score_params['sample_weight'] = sample_weight
        return self.steps[-1][-1].score(Xt, y, **score_params)


def _fit_transform_one(transformer, weight, X, y, **fit_params):
    if hasattr(transformer, 'fit_transform'):
        res = transformer.fit_transform(X, y, **fit_params)
    else:
        res = transformer.fit(X, y, **fit_params).transform(X)
    # if we have a weight for this transformer, multiply output
    if weight is None:
        return res, transformer
    return res * weight, transformer


def _fit_resample_one(sampler, X, y, **fit_params):
    X_res, y_res = sampler.fit_resample(X, y, **fit_params)

    return X_res, y_res, sampler


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
