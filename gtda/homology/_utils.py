""" Utilities function for persistent homology."""
# License: GNU AGPLv3

import numpy as np


def _pad_diagram(Xd, homology_dimensions, max_n_points, min_values):
    for dim in homology_dimensions:
        n_points = len(Xd[dim])
        n_points_to_pad = max_n_points[dim] - n_points
        if n_points == 0 and n_points_to_pad == 0:
            n_points_to_pad = 1

        if n_points_to_pad > 0:
            padding = ((0, n_points_to_pad), (0, 0))
            Xd[dim] = np.pad(Xd[dim], padding, 'constant')
            Xd[dim][-n_points_to_pad:, :] = \
                [min_values[dim], min_values[dim], dim]
    Xd = np.vstack([Xd[dim] for dim in homology_dimensions])
    return Xd
