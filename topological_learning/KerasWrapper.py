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

from sklearn.model_selection import GridSearchCV


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
    def __call__(self, modelSteps_kwargs = [ {'layerClass': klayers.LSTM, 'units': 4, 'activation': 'tanh'} ],
                 optimizer_kwargs = {'optimizerClass': koptimizers.SGD, 'lr': 0.01},
                 loss = 'binary_crossentropy', metrics = ['accuracy']):
        # Create model
        model = Sequential()
        tempStep_kwargs = modelSteps_kwargs[0]
        modelStep_kwargs = tempStep_kwargs.copy()
        model.add(modelStep_kwargs.pop('layerClass')(input_shape = self.input_shape, **modelStep_kwargs))
        for tempStep_kwargs in modelSteps_kwargs[1:]:
            modelStep_kwargs = tempStep_kwargs.copy()
            model.add(modelStep_kwargs.pop('layerClass')(**modelStep_kwargs))

        # Compile model
        tempOptimizer_kwargs = optimizer_kwargs.copy()
        optimizer = tempOptimizer_kwargs.pop('optimizerClass')(**tempOptimizer_kwargs)
        model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
        return model

    def fit(self, XList, y = None, **kwargs):
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

        # Return the classifier
        if type(XList) is list:
            X = XList[0]
            y = XList[1]
        else:
            X = XList

        self.input_shape = X.shape[1:]
        return KerasRegressor.fit(self, X, y, **kwargs)

    def predict(self, XList, **kwargs):
        """ A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check is fit had been called
        #  check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        #  X = check_array(X)
        if type(XList) is list:
            return KerasClassifier.predict(self, XList[0], **kwargs), XList[1]
        else:
            return KerasClassifier.predict(self, XList, **kwargs)

    def predict_proba(self, XList, **kwargs):
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
        kwargs = self.filter_sk_params(Sequential.predict_proba, kwargs)
        if type(XList) is list:
            probs, _ = KerasClassifier.predict(self, XList[0], **kwargs)
        else:
            probs = KerasClassifier.predict(self, XList, **kwargs)
        # check if binary classification
        if probs.shape[1] == 1:
            # first column is probability of class 0 and second is of class 1
            probs = np.hstack([1 - probs, probs])
        return probs

    def score(self, XList, y = None, **kwargs):
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

        if type(XList) is list:
            X = XList[0]
            y = XList[1]
        else:
            X = XList

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
    def __call__(self, modelSteps_kwargs = [ {'layerClass': klayers.LSTM, 'units': 4, 'activation': 'tanh'} ],
                 optimizer_kwargs = {'optimizerClass': koptimizers.SGD, 'lr': 0.01},
                 loss = 'mean_squared_error', metrics = ['accuracy']):
        # Create model
        model = Sequential()
        tempStep_kwargs = modelSteps_kwargs[0]
        modelStep_kwargs = tempStep_kwargs.copy()
        model.add(modelStep_kwargs.pop('layerClass')(input_shape = self.input_shape, **modelStep_kwargs))
        for tempStep_kwargs in modelSteps_kwargs[1:]:
            modelStep_kwargs = tempStep_kwargs.copy()
            model.add(modelStep_kwargs.pop('layerClass')(**modelStep_kwargs))

        # Compile model
        tempOptimizer_kwargs = optimizer_kwargs.copy()
        optimizer = tempOptimizer_kwargs.pop('optimizerClass')(**tempOptimizer_kwargs)
        model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
        return model

    def fit(self, XList, y = None, **kwargs):
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

        # Return the classifier
        if type(XList) is list:
            X = XList[0]
            y = XList[1]
        else:
            X = XList

        self.input_shape = X.shape[1:]
        return KerasRegressor.fit(self, X, y, **kwargs)

    def predict(self, XList, **kwargs):
        """ A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check is fit had been called
        #  check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        #  X = check_array(X)
        if type(XList) is list:
            return KerasRegressor.predict(self, XList[0], **kwargs), XList[1]
        else:
            return KerasRegressor.predict(self, XList, **kwargs)

    def score(self, XList, y = None, **kwargs):
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

        if type(XList) is list:
            X = XList[0]
            y = XList[1]
        else:
            X = XList

        loss = self.model.evaluate(X, y, **kwargs)

        if isinstance(loss, list):
            return -loss[0]
        return -loss
