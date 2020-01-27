import numpy as np


def _check_has_one_column(X):
    if X.shape[1] > 1:
        raise ValueError("X cannot have more than one column.")


def _remove_empty_and_duplicate_intervals(X_masks):
    # Remove any mask which contains only False
    X_masks = X_masks[:, np.any(X_masks, axis=0)]
    # Avoid repeating the same boolean masks (columns)
    X_masks_unique, indices = np.unique(X_masks, axis=1, return_index=True)
    # Respect the original relative column ordering
    X_masks_unique = X_masks_unique[:, np.argsort(indices)]
    return X_masks_unique
