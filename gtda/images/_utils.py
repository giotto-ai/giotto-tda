"""Helper functions for image processing."""
# License: GNU AGPLv3

import numpy as np
from scipy import ndimage as ndi


def _dilate(Xd, min_iteration, max_iteration, min_value, max_value):
    Xd = Xd.astype(float, copy=False)
    for iteration in range(min_iteration, min(max_iteration, max_value) + 1):
        Xtemp = np.asarray([ndi.binary_dilation(Xd[i])
                            for i in range(Xd.shape[0])])
        Xnew = (Xd + Xtemp) == 1
        if np.any(Xnew):
            Xd[Xnew] = iteration + min_value
        else:
            break

    return Xd


def _erode(Xd, min_iteration, max_iteration, min_value, max_value):
    return _dilate(np.logical_not(Xd), min_iteration, max_iteration,
                   min_value, max_value)
