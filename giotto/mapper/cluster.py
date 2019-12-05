from inspect import signature

import numpy as np

from joblib import Parallel, delayed

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted

from sklearn.cluster import DBSCAN


class ParallelClustering(BaseEstimator, ClusterMixin, TransformerMixin):
    def __init__(self, clusterer=None,
                 n_jobs_outer=None,
                 prefer='threads'):
        self.clusterer = clusterer
        self.n_jobs_outer = n_jobs_outer
        self.prefer = prefer

    def _validate_clusterer(self, default=DBSCAN()):
        """Check the clusterer parameter, set the
        `clusterer_` attribute."""
        if self.clusterer is not None:
            self.clusterer_ = self.clusterer
        else:
            self.clusterer_ = default

    def fit(self, X, y=None, sample_weight=None):
        self._validate_clusterer()
        X_orig, masks = X
        if sample_weight is not None:
            sample_weights = [sample_weight[masks[:, i]]
                              for i in range(masks.shape[1])]
        else:
            sample_weights = [None] * masks.shape[1]

        if self.clusterer_.metric == 'precomputed':
            single_fitter = self._fit_single_abs_labels_precomputed
        else:
            single_fitter = self._fit_single_abs_labels

        self.clusterers_ = Parallel(
            n_jobs=self.n_jobs_outer, prefer=self.prefer)(
            delayed(single_fitter)(
                X_orig, np.flatnonzero(mask),
                mask_num, sample_weight=sample_weights[mask_num])
            for mask_num, mask in enumerate(masks.T))
        return self

    def _fit_single_abs_labels(self, X, relative_indices, mask_num,
                               sample_weight=None):
        cloned_clusterer, unique_labels, unique_labels_inverse = \
            self._fit_single(X, relative_indices, sample_weight)
        self._create_abs_labels(cloned_clusterer, relative_indices, mask_num,
                                unique_labels, unique_labels_inverse)
        return cloned_clusterer

    def _fit_single_abs_labels_precomputed(self, X, relative_indices, mask_num,
                                           sample_weight=None):
        relative_2d_indices = np.ix_(relative_indices, relative_indices)
        cloned_clusterer, unique_labels, unique_labels_inverse = \
            self._fit_single(X, relative_2d_indices, sample_weight)
        self._create_abs_labels(cloned_clusterer, relative_indices, mask_num,
                                unique_labels, unique_labels_inverse)
        return cloned_clusterer

    def _fit_single(self, X, relative_indices, sample_weight):
        cloned_clusterer = clone(self.clusterer_)
        X_sub = X[relative_indices]

        fit_params = signature(cloned_clusterer.fit).parameters
        if 'sample_weight' in fit_params:
            cloned_clusterer.fit(X_sub, sample_weight=sample_weight)
        else:
            cloned_clusterer.fit(X_sub)

        unique_labels, unique_labels_inverse = np.unique(
            cloned_clusterer.labels_, return_inverse=True)
        return cloned_clusterer, unique_labels, unique_labels_inverse

    @staticmethod
    def _create_abs_labels(cloned_clusterer, relative_indices, mask_num,
                           unique_labels, inv):
        cloned_clusterer.abs_labels_ = [
            (mask_num, label, relative_indices[inv == i])
            for i, label in enumerate(unique_labels)]

    def transform(self, X, y=None, sample_weight=None):
        check_is_fitted(self, ['clusterers_'])
        Xt = [clusterer.abs_labels_ for clusterer in self.clusterers_]
        return Xt

    def fit_predict(self, X, y=None, sample_weight=None):
        self.fit(X, sample_weight=sample_weight)
        Xt = [clusterer.abs_labels_ for clusterer in self.clusterers_]
        return Xt

