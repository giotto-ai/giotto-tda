""" Utilities function for persistent homology."""
# License: GNU AGPLv3

import numpy as np
from joblib import Parallel, delayed


def _postprocess_diagrams(
        Xt, format, homology_dimensions, infinity_values, n_jobs
        ):
    def replace_infinity_values(subdiagram):
        np.nan_to_num(subdiagram, posinf=infinity_values, copy=False)
        return subdiagram[subdiagram[:, 0] < subdiagram[:, 1]]

    def pad_diagram(Xd):
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

    # Replace np.inf with infinity_values and turn into list of dictionaries
    # whose keys are the dimensions
    if format in ["ripser", "flagser"]:  # Input is list of list of subdiagrams
        # In H0, remove one infinite bar placed at the end by ripser
        slices = {dim: slice(None) if dim else slice(None, -1)
                  for dim in homology_dimensions}
        Xt = [{dim: replace_infinity_values(diagram[dim][slices[dim]])
               for dim in homology_dimensions} for diagram in Xt]
    elif format == "gudhi":  # Input is list of list of [dim, (birth, death)]
        # In H0, remove one infinite bar placed at the beginning by GUDHI
        slices = {dim: slice(None) if dim else slice(1, None)
                  for dim in homology_dimensions}
        Xt = [{dim: replace_infinity_values(
            np.array([pers_info[1] for pers_info in diagram
                      if pers_info[0] == dim]).reshape(-1, 2)[slices[dim]]
            )
            for dim in homology_dimensions} for diagram in Xt]
    else:
        raise ValueError(
            f"Unknown input format {format} for collection of diagrams."
            )

    max_n_points = {dim: np.max([len(diagram[dim]) for diagram in Xt] + [1])
                    for dim in homology_dimensions}
    min_values = {dim: min([np.min(diagram[dim][:, 0]) if diagram[dim].size
                            else np.inf for diagram in Xt])
                  for dim in homology_dimensions}
    min_values = {dim: min_value if min_value != np.inf else 0
                  for dim, min_value in min_values.items()}

    Xt = Parallel(n_jobs=n_jobs)(delayed(pad_diagram)(diagram)
                                 for diagram in Xt)
    Xt = np.stack(Xt)
    return Xt
