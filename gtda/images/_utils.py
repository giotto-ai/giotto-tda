"""Helper functions for image processing."""
# License: GNU AGPLv3

import numpy as np
from scipy import ndimage as ndi


def _dilate(Xd, n_iterations, max_value):
    for iteration in range(1, min(n_iterations, max_value) + 1):
        Xtemp = np.asarray([ndi.binary_dilation(Xd[i])
                            for i in range(Xd.shape[0])])
        Xnew = (Xd + Xtemp) == 1
        if np.any(Xnew):
            Xd[Xnew] = iteration + 1
        else:
            break

    mask_filtered = Xd == 0
    Xd -= 1
    Xd[mask_filtered] = max_value
    return Xd
