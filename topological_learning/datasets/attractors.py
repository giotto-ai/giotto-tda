# Authors: Guillaume Tauzin <guillaume.tauzin@epfl.ch>
# License: TBD

import numpy as np
from random import randint


class Dataset(object):
    def __init__(self, time_step=0.01, max_duration=20000, mean_noise=0, std_deviation_noise=0):
        self.time_step = time_step
        self.max_duration = max_duration
        self.mean_noise = mean_noise
        self.std_deviation_noise = std_deviation_noise

    def add_noise(self):
        return np.random.normal(self.mean_noise, self.std_deviation_noise, size=self.max_duration)

class LorenzDataset(Dataset):
    """

    Parameters
    ----------
    samplingType : str
        The type of sampling

    samplingPeriod : str
        Time anchors giving the period of the sampling. Used only if samplingType is 'periodic'

    samplingTimeList : list of datetime
        Datetime at which the samples should be taken. Used only if samplingType is 'fixed'

    Attributes
    ----------
    rho_ : ndarray

    """

    def __init__(self, initial_conditions=(1, -10, 10), sigma=10., beta=8./3.,
                 rho_min=5, rho_max=20, transition_list=None, number_transitions=25,
                 transition_duration=100, time_step=0.01, max_duration=20000,
                 mean_noise = 0, std_deviation_noise=0):
        super(LorenzDataset, self).__init__(time_step, max_duration, mean_noise, std_deviation_noise)
        self.initial_conditions = initial_conditions
        self.sigma = sigma
        self.beta = beta
        self.transition_list = transition_list
        self.generate_rho(rho_min, rho_max, transition_list, number_transitions, transition_duration)

    def generate_rho(self, rho_min, rho_max, transition_list, number_transitions, transition_duration):
        self.rho_ = rho_min * np.ones(self.max_duration)
        self.regime_ = np.zeros((self.max_duration, 2), dtype=np.int8)

        if self.transition_list == None:
            self.transition_list = [i/(2*number_transitions) + randint(0,100)/2000. for i in range(2*number_transitions)]
            self.transition_list.sort()
            # print(len(self.transition_list), self.transition_list)

        for i in range(0, len(self.transition_list), 2):
            peak_begin = int(self.transition_list[i] * self.max_duration)
            peak_end = int(self.transition_list[i+1] * self.max_duration)

            for n in range(peak_begin - transition_duration, min(peak_begin, self.max_duration)):
                self.rho_[n] = rho_min + (n - (peak_begin - transition_duration)) * (rho_max - rho_min) / transition_duration
                if self.rho_[n] > 10:
                    self.regime_[n, :] = np.array([1, 1])
                else:
                    self.regime_[n, :] = np.array([0, 1])

            for n in range(peak_begin, min(peak_end, self.max_duration)):
                self.rho_[n] = rho_max
                self.regime_[n, :] = np.array([1, 1])

            for n in range(peak_end, min(peak_end + transition_duration, self.max_duration)):
                self.rho_[n] = rho_max + (n - peak_end)*(rho_min - rho_max) / transition_duration
                if self.rho_[n] > 10:
                    self.regime_[n, :] = np.array([1, 1])
                else:
                    self.regime_[n, :] = np.array([1, 0])

    def run(self):
        self.x_, self.y_, self.z_ = (np.zeros((self.max_duration)) for _ in range(3))
        self.x_[0], self.y_[0], self.z_[0] = self.initial_conditions

        for n in range(1, self.max_duration):
            self.x_[n] = self.x_[n-1] + (self.sigma * (self.y_[n-1] - self.x_[n-1])) * self.time_step
            self.y_[n] = self.y_[n-1] + (self.rho_[n-1] * self.x_[n-1] - self.x_[n-1] * self.z_[n-1] - self.y_[n-1]) * self.time_step
            self.z_[n] = self.z_[n-1] + (self.x_[n-1] * self.y_[n-1] - self.beta * self.z_[n-1]) * self.time_step

        self.x_ += self.add_noise()
        self.y_ += self.add_noise()
        self.z_ += self.add_noise()

class _LorenzLabeller():
    """
    Target transformer for the Lorenz attractor.

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

    def __init__(self):
        pass

    @staticmethod
    def _validate_params():
        """A class method that checks whether the hyperparameters and the input parameters
           of the :meth:'fit' are valid.
        """
        pass

    def label(self, XList):
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
        check_is_fitted(self, ['isFitted'])

        y = XList[2]

        yTransformed = y.reshape((-1, y.shape[2] // 2, 2))
        yTransformed = np.mean(yTransformed, axis=1)
        XListTransformed = [XList[0], yTransformed]

        return XListTransformed
