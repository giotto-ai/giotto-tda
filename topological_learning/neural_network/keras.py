import sklearn as sk
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import sklearn.ensemble
from keras.models import Sequential
import keras.layers as klayers
import keras.optimizers as koptimizers
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

import numpy as np
import pandas as pd


class KerasClassifierWrapper(KerasClassifier):
    """ A wrapper for Keras classifiers.

    Parameters
    ----------

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """
    def __call__(self, steps_kwargs=[ {'layer': klayers.LSTM, 'units': 4, 'activation': 'tanh'} ],
                 optimizer_kwargs={'optimizer': koptimizers.RMSprop, 'lr': 0.01},
                 loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy']):
        # Create model
        model = Sequential()
        temp_step_kwargs = steps_kwargs[0]
        step_kwargs = temp_step_kwargs.copy()
        model.add(step_kwargs.pop('layer')(input_shape=self.input_shape, **step_kwargs))
        for temp_step_kwargs in steps_kwargs[1:-1]:
            step_kwargs = temp_step_kwargs.copy()
            model.add(step_kwargs.pop('layer')(**step_kwargs))
        temp_step_kwargs = steps_kwargs[-1]
        step_kwargs = temp_step_kwargs.copy()
        model.add(step_kwargs.pop('layer')(units=self.output_units, **step_kwargs))

        # Compile model
        temp_optimizer_kwargs = optimizer_kwargs.copy()
        optimizer = temp_optimizer_kwargs.pop('optimizer')(**temp_optimizer_kwargs)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model

    def fit(self, X, y=None, **kwargs):
        """A reference implementation of a fitting function for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        #  X, y = check_X_y(X, y)
        # Store the classes seen during fit
        #  self.classes_ = unique_labels(y)

        self.input_shape = X.shape[1:]
        self.output_units = np.max(y) + 1
        return KerasClassifier.fit(self, X, y, verbose=0, **kwargs)

    def score(self, X, y=None, **kwargs):
        """
        Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X: array-like, shape `(n_samples, n_features)
        Test samples where `n_samples` is the number of samples
        and `n_features` is the number of features.
        y: array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
        True labels for `x`.
        **kwargs: dictionary arguments
        Legal arguments are the arguments of `Sequential.evaluate.

        Returns
        -------
        score: float
        Mean accuracy of predictions on `X` wrt. `y`.
        """
        kwargs = self.filter_sk_params(Sequential.evaluate, kwargs)

        loss = self.model.evaluate(X, y, **kwargs)

        if isinstance(loss, list):
            return -loss[0]
        return -loss


class KerasRegressorWrapper(KerasRegressor):
    """ A wrapper for Keras regressors.

    Parameters
    ----------

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """
    def __call__(self, steps_kwargs = [ {'layer': klayers.LSTM, 'units': 4, 'activation': 'tanh'} ],
                 optimizer_kwargs = {'optimizer': koptimizers.SGD, 'lr': 0.01},
                 loss = 'mean_squared_error', metrics = ['accuracy']):
        # Create model
        model = Sequential()
        temp_step_kwargs = steps_kwargs[-1]
        step_kwargs = temp_step_kwargs.copy()
        model.add(step_kwargs.pop('layer')(input_shape = self.input_shape, **step_kwargs))
        for temp_step_kwargs in steps_kwargs[1:]:
            step_kwargs = temp_step_kwargs.copy()
            model.add(step_kwargs.pop('layer')(**step_kwargs))
        temp_step_kwargs = steps_kwargs[-1]
        step_kwargs = temp_step_kwargs.copy()
        model.add(step_kwargs.pop('layer')(**step_kwargs))

        # Compile model
        temp_optimizer_kwargs = optimizer_kwargs.copy()
        optimizer = temp_optimizer_kwargs.pop('optimizer')(**temp_optimizer_kwargs)
        model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
        return model

    def fit(self, X, y=None, **kwargs):
        """A reference implementation of a fitting function for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        #  X, y = check_X_y(X, y)
        # Store the classes seen during fit
        #  self.classes_ = unique_labels(y)

        self.input_shape = X.shape[1:]
        return KerasRegressor.fit(self, X, y, verbose=0, **kwargs)

    def score(self, X, y=None, **kwargs):
        """
        Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X: array-like, shape `(n_samples, n_features)
        Test samples where `n_samples` is the number of samples
        and `n_features` is the number of features.
        y: array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
        True labels for `x`.
        **kwargs: dictionary arguments
        Legal arguments are the arguments of `Sequential.evaluate.

        Returns
        -------
        score: float
        Mean accuracy of predictions on `X` wrt. `y`.
        """
        kwargs = self.filter_sk_params(Sequential.evaluate, kwargs)

        loss = self.model.evaluate(X, y, **kwargs)

        if isinstance(loss, list):
            return -loss[0]
        return -loss
