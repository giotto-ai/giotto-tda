"""Helper functions for image processing."""
# License: GNU AGPLv3

import numpy as np
from scipy import ndimage as ndi


def _dilate(X, min_iteration, max_iteration, min_value, max_value):
    X = X * 1.
    for iteration in range(min_iteration, min(max_iteration, max_value) + 1):
        Xtemp = np.asarray([ndi.binary_dilation(x) for x in X])
        Xnew = (X + Xtemp) == 1
        if np.any(Xnew):
            X[Xnew] = iteration + min_value
        else:
            break

    return X


def _erode(X, min_iteration, max_iteration, min_value, max_value):
    return _dilate(np.logical_not(X), min_iteration, max_iteration,
                   min_value, max_value)
