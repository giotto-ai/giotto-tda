import sklearn as sk
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
import sklearn.preprocessing as skprep
import sklearn.ensemble as skens

import numpy as np

class ScalerWrapper(BaseEstimator, TransformerMixin):
    """
    data sampling transformer that returns a sampled Pandas dataframe with a datetime index

    Parameters
    ----------
    samplingType : str
        The type of sampling

        - data_type: string, must equal either 'points' or 'distance_matrix'.
        - data_iter: an iterator. If data_iter is 'points' then each object in the iterator
          should be a numpy array of dimension (number of points, number of coordinates),
          or equivalent nested list structure. If data_iter is 'distance_matrix' then each
          object in the iterator should be a full (symmetric) square matrix (numpy array)
          of shape (number of points, number of points), __or a sparse distance matrix

    Attributes
    ----------
    isFitted : boolean
        Whether the transformer has been fitted
    """

    def __init__(self, scaler = skprep.MinMaxScaler, **scaler_kwargs):
        self.scaler = scaler(**scaler_kwargs)

    def get_params(self, deep=True):
        return self.scaler.get_params(deep)

    def set_params(self, **parameters):
        return self.scaler.set_params(**parameters)

    def fit(self, XList, y = None):
        """A reference implementation of a fitting function for a transformer.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        self.scaler.fit(XList[0])

        return self

    #@jit
    def transform(self, XList, y = None, copy = None):
        """ Implementation of the sk-learn transform function that samples the input.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        X_transformed : array of int of shape = [n_samples, n_features]
            The array containing the element-wise square roots of the values
            in `X`
        """
        # Check is fit had been called
        #check_is_fitted(self, ['isFitted'])

        XListTransformed = []

        XTransformed = self.scaler.transform(XList[0])

        XListTransformed.append(XTransformed)

        if len(XList) == 2:
            XListTransformed.append(XList[1])

        return XListTransformed

    def inverse_transform(self, XList, copy = None):
        """Scale back the data to the original representation

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to scale along the features axis.
        copy : bool, optional (default: None)
            Copy the input X or not.

        Returns
        -------
        X_tr : array-like, shape [n_samples, n_features]
            Transformed array.
        """
        # Check is fit had been called
        #check_is_fitted(self, ['isFitted'])

        XListTransformed = []

        XTransformed = self.scaler.inverse_transform(XList[0])

        XListTransformed.append(XTransformed)

        if len(XList) == 2:
            XListTransformed.append(XList[1])

        return X_transformed


class TransformerWrapper(BaseEstimator, TransformerMixin):
    """
    data sampling transformer that returns a sampled Pandas dataframe with a datetime index

    Parameters
    ----------
    samplingType : str
        The type of sampling

        - data_type: string, must equal either 'points' or 'distance_matrix'.
        - data_iter: an iterator. If data_iter is 'points' then each object in the iterator
          should be a numpy array of dimension (number of points, number of coordinates),
          or equivalent nested list structure. If data_iter is 'distance_matrix' then each
          object in the iterator should be a full (symmetric) square matrix (numpy array)
          of shape (number of points, number of points), __or a sparse distance matrix

    Attributes
    ----------
    isFitted : boolean
        Whether the transformer has been fitted
    """

    def __init__(self, transformer = skprep.MinMaxScaler, **transformer_kwargs):
        self.scaler = transformer(**transformer_kwargs)

    def get_params(self, deep=True):
        return self.transformer.get_params(deep)

    def set_params(self, **parameters):
        return self.transformer.set_params(parameters)

    def fit(self, XList, y = None):
        """A reference implementation of a fitting function for a transformer.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        return self.transformer.fit(XList[0])

    #@jit
    def transform(self, XList, y = None, copy = None):
        """ Implementation of the sk-learn transform function that samples the input.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        X_transformed : array of int of shape = [n_samples, n_features]
            The array containing the element-wise square roots of the values
            in `X`
        """
        # Check is fit had been called
        #check_is_fitted(self, ['isFitted'])

        XListTransformed = []

        XTransformed = self.transformer.transform(XList[0])

        XListTransformed.append(XTransformed)

        if len(XList) == 2:
            XListTransformed.append(XList[1])

        return XListTransformed

class EstimatorWrapper(BaseEstimator):
    """ A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    """
    def __init__(self, estimator = skens.RandomForestRegressor, **estimator_kwargs):
        self.estimator = estimator(**estimator_kwargs)

    def get_params(self, deep=True):
        return self.estimator.get_params(deep)

    def set_params(self, **parameters):
        return self.estimator.set_params(parameters)

    def fit(self, X, y):
        """A reference implementation of a fitting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        self.isFitted = True

        return self.estimator.fit(XList[0], XList[1])

    def predict(self, X):
        """ A reference implementation of a predicting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        return self.estimator.predict(XList[0]), XList[1]
