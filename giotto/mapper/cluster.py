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
        X_tot, masks = X
        if sample_weight is not None:
            sample_weights = [sample_weight[masks[:, i]]
                              for i in range(masks.shape[1])]
        else:
            sample_weights = [None] * masks.shape[1]

        self.clusterers_ = Parallel(n_jobs=self.n_jobs_outer,
                                    prefer=self.prefer)(
            delayed(self._fit_single)(X_tot[mask],
                                      np.argwhere(mask).squeeze(axis=1),
                                      k, sample_weight=sample_weights[k])
            for k, mask in enumerate(masks.T))
        self._is_fitted = True
        return self

    def _fit_single(self, X, relative_indices, mask_num,
                    sample_weight=None):
        cloned_clusterer = clone(self.clusterer_)
        fit_params = signature(cloned_clusterer.fit).parameters
        if 'sample_weight' in fit_params:
            cloned_clusterer.fit(X, sample_weight=sample_weight)
        else:
            cloned_clusterer.fit(X)
        unique_cluster_labels, inv = np.unique(cloned_clusterer.labels_,
                                               return_inverse=True)
        cloned_clusterer.abs_labels_ = [
            (mask_num, label, relative_indices[inv == i])
            for i, label in enumerate(unique_cluster_labels)]
        return cloned_clusterer

    def transform(self, X, y=None, sample_weight=None):
        check_is_fitted(self, ['_is_fitted'])
        Xt = [clusterer.abs_labels_ for clusterer in self.clusterers_]
        return Xt

    def fit_predict(self, X, y=None, sample_weight=None):
        self.fit(X, sample_weight=sample_weight)
        Xt = [clusterer.abs_labels_ for clusterer in self.clusterers_]
        return Xt
