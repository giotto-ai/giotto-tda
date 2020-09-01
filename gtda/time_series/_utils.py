"""Utility function for time series processing."""
# License: GNU AGPLv3

import numpy as np
from functools import partial


def _time_delay_embedding(X, time_delay=1, dimension=2, stride=1,
                          flatten=False, ensure_last_value=True):
    if hasattr(X, 'shape') and hasattr(X, 'ndim'):  # ndarray input
        n_timestamps = X.shape[-1]
        n_points, offset = \
            divmod(n_timestamps - time_delay * (dimension - 1) - 1, stride)
        n_points += 1
        if n_points <= 0:
            raise ValueError(
                f"Not enough time stamps ({n_timestamps}) to produce at least "
                f"one {dimension}-dimensional vector under the current choice "
                f"of time delay ({time_delay})."
                )
        indices = np.tile(np.arange(0, time_delay * dimension, time_delay),
                          (n_points, 1))
        indices += np.arange(n_points)[:, None] * stride
        if ensure_last_value:
            indices += offset

        X_embedded = X[..., indices]
        if flatten:
            transpose_axes = (0, *range(1, X.ndim)[::-1], X.ndim)
            X_embedded = np.transpose(X_embedded, axes=transpose_axes).\
                reshape(len(X), -1, dimension * np.prod(X.shape[1:-1]))
    else:  # list of ndarray input
        func = partial(_time_delay_embedding, time_delay=time_delay,
                       dimension=dimension, stride=stride, flatten=flatten,
                       ensure_last_value=ensure_last_value)
        X_embedded = []
        for x in X:
            x_embedded = func(x[None, ...])[0]
            X_embedded.append(x_embedded)

    return X_embedded