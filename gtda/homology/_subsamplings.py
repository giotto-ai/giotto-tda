# License: GNU AGPLv3

import numpy as np

import random


def random_subsampling(X, n_landmarks, **kwargs):
    return X[random.sample(xrange(len(X)), self.n_landmarks), :]


SUBSAMPLING_FUNCTIONS = {
    'random': random_subsampling
}
