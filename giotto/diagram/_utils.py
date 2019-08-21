# Authors: Guillaume Tauzin <guillaume.tauzin@epfl.ch>
# License: TBD

import math as m
import numpy as np


def _rotate_clockwise(X):
    rotationMatrix = m.sqrt(2) / 2. * np.array([[1, -1], [1, 1]])
    return np.dot(X, rotationMatrix)


def _rotate_anticlockwise(X):
    rotationMatrix = m.sqrt(2) / 2. * np.array([[1, 1], [-1, 1]])
    return np.dot(X, rotationMatrix)


def _pad(X, max_betti_numbers):
    X_padded = {dimension: np.pad(X[dimension], ((0, 0), (0, max_betti_numbers[dimension] - X[dimension].shape[1]), (0, 0)), 'constant')
                for dimension in X.keys()}
    return X_padded


def _sort(XScaled, homology_dimensions):
    indices = {dimension: np.argsort(XScaled[dimension][:, :, 1] - XScaled[dimension][:, :, 0], axis=1)
               for dimension in homology_dimensions}
    indices = {dimension: np.stack([indices[dimension], indices[dimension]], axis=2) for dimension in homology_dimensions}
    XSorted = {dimension: np.flip(np.take_along_axis(XScaled[dimension], indices[dimension], axis=1), axis=1)
               for dimension in homology_dimensions}
    return {**XSorted, **{dimension: XScaled[dimension] for dimension in set(XScaled.keys()) - homology_dimensions}}


def _filter(XScaled, homology_dimensions, cutoff):
    XFiltered = {dimension: XScaled[dimension].copy() for dimension in homology_dimensions}
    mask = {dimension: m.sqrt(2) / 2. * (X[:, :, 1] - X[:, :, 0]) <= cutoff for dimension, X in XFiltered.items()}

    for dimension, X in XFiltered.items():
        X[mask[dimension], :] = [0, 0]

    maxPoints = {dimension: np.max(np.sum(X[:, :, 1] != 0, axis=1)) for dimension, X in XFiltered.items()}
    XFiltered = {dimension: X[:, :maxPoints[dimension], :] for dimension, X in XFiltered.items()}
    return {**XFiltered, **{dimension: XScaled[dimension] for dimension in set(XScaled.keys()) - homology_dimensions}}


def _sample(X, n_samples):
    maximum_persistences = {dimension: np.max(X[dimension][:, :, 1]) * m.sqrt(2) if X[dimension][:, :, 0].size != 0
                            else -np.inf for dimension in X.keys()}
    maximum_persistence = max(list(maximum_persistences.values()))
    maximum_persistences = {dimension: maximum_persistences[dimension] if maximum_persistences[dimension] != -np.inf else maximum_persistence
                            for dimension in X.keys()}

    minimum_persistences = {dimension: np.min(X[dimension][:, :, 0]) * m.sqrt(2) if X[dimension][:, :, 0].size != 0
                            else np.inf for dimension in X.keys()}
    minimum_persistence = min(list(minimum_persistences.values()))
    minimum_persistences = {dimension: minimum_persistences[dimension] if minimum_persistences[dimension] != np.inf else minimum_persistence
                            for dimension in X.keys()}

    step_persistences = {dimension: (maximum_persistences[dimension] - minimum_persistences[dimension]) / n_samples for dimension in X.keys()}
    sampling = {dimension: np.arange(minimum_persistences[dimension], maximum_persistences[dimension],
                                     step_persistences[dimension]).reshape((-1, 1))
                for dimension in X.keys()}
    return sampling
