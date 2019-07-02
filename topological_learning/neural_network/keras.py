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
    """
    A wrapper for using Keras classifiers in sk-learn's pipelines.

    Parameters
    ----------
    layers_kwargs : list of dict
        List of dictionaries, each of them describing a Keras layer of the neural network.
        The 'layer' key corresponds to the Keras class of the current layer, while other
        arguments are its parameters.

    optimizer_kwargs : dict
        Dictionary describing the Keras optimizer.
        The 'optimizer' key corresponds to the Keras class of the optimizer, while other
        arguments are its parameters.

    loss : str, default 'sparse_categorical_crossentropy'
        Keras loss function name.

    metrics : list of str, default ['sparse_categorical_accuracy']
        List of Keras metrics.

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from topological_learning.neural_network import KerasClassifier
    >>> import keras.layers as klayers
    >>> import keras.optimizers as koptimizers
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = load_digits(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.33, random_state=42)
    >>> layers_kwargs = [
    ...     {'layer': klayers.normalization.BatchNormalization},
    ...     {'layer': layer, 'units': 4, 'activation': 'tanh'},
    ...     {'layer': klayers.Dense}
    ... ]
    >>> optimizer_kwargs = {'optimizer': koptimizers.RMSprop, 'lr': 0.01}
    >>> classifier = KerasClassifier(layers_kwargs=layers_kwargs, optimizer_kwargs=optimizer_kwargs)
    >>> classifier.fit(X_train, y_train)
    >>> classifier.score(X_test)

    """
    def __call__(self, layers_kwargs=[ {'layer': klayers.LSTM, 'units': 4, 'activation': 'tanh'} ],
                 optimizer_kwargs={'optimizer': koptimizers.RMSprop, 'lr': 0.01},
                 loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy']):
        # Create model
        model = Sequential()
        temp_layer_kwargs = layers_kwargs[0]
        layer_kwargs = temp_layer_kwargs.copy()
        model.add(layer_kwargs.pop('layer')(input_shape=self._input_shape, **layer_kwargs))
        for temp_layer_kwargs in layers_kwargs[1:-1]:
            layer_kwargs = temp_layer_kwargs.copy()
            model.add(layer_kwargs.pop('layer')(**layer_kwargs))
        temp_layer_kwargs = layers_kwargs[-1]
        layer_kwargs = temp_layer_kwargs.copy()
        model.add(layer_kwargs.pop('layer')(units=self._output_units, **layer_kwargs))

        # Compile model
        temp_optimizer_kwargs = optimizer_kwargs.copy()
        optimizer = temp_optimizer_kwargs.pop('optimizer')(**temp_optimizer_kwargs)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model

    def clone(self):
        params = self.get_params()
        return KerasClassifierWrapper(**params)

    def fit(self, X, y=None, **kwargs):
        """
        Fit the Keras neural network on the training set (X, y).

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data.

        y : None
            Ignored.

        **kwargs : dictionary arguments
            Legal arguments are the arguments of `Sequential.fit.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        #  X, y = check_X_y(X, y)
        # Store the classes seen during fit
        #  self.classes_ = unique_labels(y)

        self._input_shape = X.shape[1:]
        self._output_units = np.max(y) + 1
        return KerasClassifier.fit(self, X, y, verbose=0, **kwargs)

    def score(self, X, y=None, **kwargs):
        """
        Returns the mean metric on the given test data and labels.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Test samples where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            True labels for `X`.

        **kwargs : dictionary arguments
            Legal arguments are the arguments of `Sequential.evaluate.

        Returns
        -------
        score : float
            Mean metric of predictions on `X` wrt. `y`.
        """
        kwargs = self.filter_sk_params(Sequential.evaluate, kwargs)

        loss = self.model.evaluate(X, y, **kwargs)

        if isinstance(loss, list):
            return -loss[0]
        return -loss


class KerasRegressorWrapper(KerasRegressor):
    """
    A wrapper for using Keras regressors in sk-learn's pipelines.

    Parameters
    ----------
    layers_kwargs : list of dict
        List of dictionaries, each of them describing a Keras layer of the neural network.
        The 'layer' key corresponds to the Keras class of the current layer, while other
        arguments are its parameters.

    optimizer_kwargs : dict
        Dictionary describing the Keras optimizer.
        The 'optimizer' key corresponds to the Keras class of the optimizer, while other
        arguments are its parameters.

    loss : str, default 'mean_squared_error'
        Keras loss function name.

    metrics : list of str, default ['accuracy']
        List of Keras metrics.

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from topological_learning.neural_network import KerasRegressor
    >>> import keras.layers as klayers
    >>> import keras.optimizers as koptimizers
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = load_digits(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.33, random_state=42)
    >>> layers_kwargs = [
    ...     {'layer': klayers.normalization.BatchNormalization},
    ...     {'layer': layer, 'units': 4, 'activation': 'tanh'},
    ...     {'layer': klayers.Dense}
    ... ]
    >>> optimizer_kwargs = {'optimizer': koptimizers.RMSprop, 'lr': 0.01}
    >>> regressor = KerasRegressor(layers_kwargs=layers_kwargs, optimizer_kwargs=optimizer_kwargs)
    >>> regressor.fit(X_train, y_train)
    >>> regressor.score(X_test)

    """
    def __call__(self, layers_kwargs = [ {'layer': klayers.LSTM, 'units': 4, 'activation': 'tanh'} ],
                 optimizer_kwargs = {'optimizer': koptimizers.SGD, 'lr': 0.01},
                 loss = 'mean_squared_error', metrics = ['accuracy']):
        # Create model
        model = Sequential()
        temp_layer_kwargs = layers_kwargs[-1]
        layer_kwargs = temp_layer_kwargs.copy()
        model.add(layer_kwargs.pop('layer')(input_shape = self._input_shape, **layer_kwargs))
        for temp_layer_kwargs in layers_kwargs[1:]:
            layer_kwargs = temp_layer_kwargs.copy()
            model.add(layer_kwargs.pop('layer')(**layer_kwargs))
        temp_layer_kwargs = layers_kwargs[-1]
        layer_kwargs = temp_layer_kwargs.copy()
        model.add(layer_kwargs.pop('layer')(**layer_kwargs))

        # Compile model
        temp_optimizer_kwargs = optimizer_kwargs.copy()
        optimizer = temp_optimizer_kwargs.pop('optimizer')(**temp_optimizer_kwargs)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model

    def clone(self):
        params = self.get_params()
        return KerasRegressorWrapper(**params)

    def fit(self, X, y=None, **kwargs):
        """
        Fit the Keras neural network on the training set (X, y).

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data.

        y : None
            Ignored.

        **kwargs : dictionary arguments
            Legal arguments are the arguments of `Sequential.fit.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        #  X, y = check_X_y(X, y)
        # Store the classes seen during fit
        #  self.classes_ = unique_labels(y)
        self._input_shape = X.shape[1:]
        return KerasRegressor.fit(self, X, y, verbose=0, **kwargs)

    def score(self, X, y=None, **kwargs):
        """
        Returns the mean metric on the given test data and labels.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Test samples where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : ndarray, shape (n_samples,) or (n_samples, n_outputs)
            True labels for `X`.

        **kwargs : dictionary arguments
            Legal arguments are the arguments of `Sequential.evaluate.

        Returns
        -------
        score : float
            Mean metric of predictions on `X` wrt. `y`.
        """
        kwargs = self.filter_sk_params(Sequential.evaluate, kwargs)

        loss = self.model.evaluate(X, y, **kwargs)

        if isinstance(loss, list):
            return -loss[0]
        return -loss
