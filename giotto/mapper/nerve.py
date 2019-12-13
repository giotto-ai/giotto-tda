import numpy as np
import igraph as ig

from functools import partial
from itertools import product, combinations

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class Nerve(BaseEstimator, TransformerMixin):
    def __init__(self, min_intersection=1):
        self.min_intersection = min_intersection

    def fit(self, X, y=None):
        self.edges_ = list(self._generate_edges(X))
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ['edges_'])
        graph = ig.Graph()
        # TODO: improve dummy variable names in list comprehension
        graph.add_vertices([x[:2] for V in X for x in V])
        graph.add_edges([
            (edge['node_indices'][0][0], edge['node_indices'][1][0])
            for edge in self.edges_
        ])
        # add cluster member indices to each node
        graph.vs['elements'] = [x[2] for V in X for x in V]
        return graph

    @staticmethod
    def _unpack_product(tup):
        return product(*tup)

    @staticmethod
    def _pairwise_intersections(min_intersection, element):
        for tup in element:
            data = dict()
            tuple_1, tuple_2 = tup
            data['node_indices'] = tuple((tuple_1[:2], tuple_2[:2]))
            data['intersection'] = np.intersect1d(tuple_1[2], tuple_2[2])
            if data['intersection'].size >= min_intersection:
                yield data

    def _generate_edges(self, nodes):
        valid_intersections = partial(self._pairwise_intersections,
                                      self.min_intersection)
        for pair in map(self._unpack_product, combinations(nodes, 2)):
            for intersection in valid_intersections(pair):
                yield intersection
