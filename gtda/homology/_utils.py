""" Utilities function for persistent homology."""
# License: GNU AGPLv3

import numpy as np
from joblib import Parallel, delayed


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
                [min_values[dim], min_values[dim]]

    # Add dimension as the third elements of each (b, d) tuple
    Xd = [np.insert(Xd[dim], 2, dim, axis=1)
          for dim in homology_dimensions]

    Xd = np.vstack(Xd)
    return Xd


def _postprocess_diagrams(Xt, homology_dimensions, infinity_values, n_jobs):
    # Replacing np.inf with infinity_values and turning list of list of
    # subdiagrams into list of dictionaries whose keys are the dimensions
    Xt = [{dim: np.nan_to_num(diagram[dim], posinf=infinity_values)
          for dim in homology_dimensions}
          for diagram in Xt]

    # Removing points whose birth is higher than their death
    Xt = [{dim: subdiagram[subdiagram[:, 0] < subdiagram[:, 1]]
          for dim, subdiagram in diagram.items()}
          for diagram in Xt]

    max_n_points = {dim: np.max([len(diagram[dim]) for diagram in Xt] + [1])
                    for dim in homology_dimensions}
    min_values = {dim: min([np.min(diagram[dim][:, 0]) if diagram[dim].size
                            else np.inf for diagram in Xt])
                  for dim in homology_dimensions}
    min_values = {dim: min_value if min_value != np.inf else 0
                  for dim, min_value in min_values.items()}
    Xt = Parallel(n_jobs=n_jobs)(delayed(_pad_diagram)(
        diagram, homology_dimensions, max_n_points, min_values)
                                 for diagram in Xt)
    Xt = np.stack(Xt)
    return Xt
