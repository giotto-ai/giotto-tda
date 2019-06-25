import numpy as np
from random import randint

class Dataset(object):
    def __init__(self, timeStep = 0.01, maxDuration = 20000, meanNoise = 0, stdDeviationNoise = 0):
        self.timeStep = timeStep
        self.maxDuration = maxDuration
        self.meanNoise = meanNoise
        self.stdDeviationNoise = stdDeviationNoise

    def add_noise(self):
        return np.random.normal(self.meanNoise, self.stdDeviationNoise, size=self.maxDuration)

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
    isFitted : boolean
        Whether the transformer has been fitted
    """

    def __init__(self, initialConditions = (1, -10, 10), sigma = 10., beta = 8./3.,
                 rhoMin = 5, rhoMax = 20, transitionList = None, numberTransitions = 25,
                 transitionDuration = 100, timeStep = 0.01, maxDuration = 20000,
                 meanNoise = 0, stdDeviationNoise = 0):
        super(LorenzDataset, self).__init__(timeStep, maxDuration, meanNoise, stdDeviationNoise)
        self.initialConditions = initialConditions
        self.sigma = sigma
        self.beta  = beta
        self.transitionList = transitionList
        self.generate_rho(rhoMin, rhoMax, transitionList, numberTransitions, transitionDuration)

    def generate_rho(self, rhoMin, rhoMax, transitionList, numberTransitions, transitionDuration):
        self.rho = rhoMin * np.ones(self.maxDuration)
        self.regime = np.zeros((self.maxDuration, 2), dtype=np.int8)

        if self.transitionList == None:
            self.transitionList = [i/(2*numberTransitions) + randint(0,100)/2000. for i in range(2*numberTransitions)]
            self.transitionList.sort()
            # print(len(self.transitionList), self.transitionList)

        for i in range(0, len(self.transitionList), 2):
            peakBegin = int(self.transitionList[i] * self.maxDuration)
            peakEnd = int(self.transitionList[i+1] * self.maxDuration)

            for n in range(peakBegin - transitionDuration, min(peakBegin, self.maxDuration)):
                self.rho[n] = rhoMin + (n - (peakBegin - transitionDuration)) * (rhoMax - rhoMin) / transitionDuration
                if self.rho[n] > 10:
                    self.regime[n, :] = np.array([1, 1])
                else:
                    self.regime[n, :] = np.array([0, 1])

            for n in range(peakBegin, min(peakEnd, self.maxDuration)):
                self.rho[n] = rhoMax
                self.regime[n, :] = np.array([1, 1])

            for n in range(peakEnd, min(peakEnd + transitionDuration, self.maxDuration)):
                self.rho[n] = rhoMax + (n - peakEnd)*(rhoMin - rhoMax) / transitionDuration
                if self.rho[n] > 10:
                    self.regime[n, :] = np.array([1, 1])
                else:
                    self.regime[n, :] = np.array([1, 0])

    def run(self):
        self.x, self.y, self.z = (np.zeros((self.maxDuration)) for _ in range(3))
        self.x[0], self.y[0], self.z[0] = self.initialConditions

        for n in range(1, self.maxDuration):
            self.x[n] = self.x[n-1] + (self.sigma * (self.y[n-1] - self.x[n-1])) * self.timeStep
            self.y[n] = self.y[n-1] + (self.rho[n-1] * self.x[n-1] - self.x[n-1] * self.z[n-1] - self.y[n-1]) * self.timeStep
            self.z[n] = self.z[n-1] + (self.x[n-1] * self.y[n-1] - self.beta * self.z[n-1]) * self.timeStep

        self.x += self.add_noise()
        self.y += self.add_noise()
        self.z += self.add_noise()

class LorenzLabeller():
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
