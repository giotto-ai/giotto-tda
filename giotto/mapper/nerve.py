"""Construct the nerve of a Mapper cover."""
# License: Apache 2.0

from functools import reduce
from itertools import combinations
from operator import iconcat

import igraph as ig
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class Nerve(BaseEstimator, TransformerMixin):
    """TODO

    """

    def __init__(self, min_intersection=1):
        self.min_intersection = min_intersection

    def fit(self, X, y=None):
        """TODO

        """
        # X is a list of lists where each sublist corresponds to a filter
        # interval. The sublists contain tuples, one for each cluster found
        # within the interval.
        self.X_ = reduce(iconcat, X, [])
        # preprocesses X by 1) flattening and 2) extending each tuple
        self.X_ = [(node_info[0], *node_info[1])
                   for node_info in zip(range(len(self.X_)), self.X_)]
        self.edges_ = list(self._generate_edges(self.X_))
        return self

    def transform(self, X, y=None):
        """TODO

        """
        check_is_fitted(self)
        graph = ig.Graph()
        graph.add_vertices([vertex[0] for vertex in self.X_])
        graph.add_edges([
            (edge['node_indices'][0][0], edge['node_indices'][1][0])
            for edge in self.edges_
        ])
        graph['node_metadata'] = dict(
            zip(['node_id', 'interval_id', 'cluster_id', 'node_elements'],
                zip(*self.X_)))
        return graph

    @staticmethod
    def _pairwise_intersections(min_intersection, node_pair):
        data = dict()
        node_1, node_2 = node_pair
        data['node_indices'] = tuple((node_1[0:3], node_2[0:3]))
        data['intersection'] = np.intersect1d(node_1[3], node_2[3])
        if data['intersection'].size >= min_intersection:
            yield data

    def _generate_edges(self, nodes):
        node_tuples = combinations(nodes, 2)
        for pair in node_tuples:
            for intersection in\
             self._pairwise_intersections(self.min_intersection, pair):
                yield intersection
