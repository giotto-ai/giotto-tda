"""Construct the nerve of a refined Mapper cover."""
# License: GNU AGPLv3

from collections import defaultdict
from itertools import combinations, filterfalse

import numpy as np
from igraph import Graph
from sklearn.base import BaseEstimator, TransformerMixin


def _limit_mapping(mapping):
    """Given a 1D array interpreted as a function
    :math:`f : \\{0, \\ldots, n - 1\\}} \to \\{0, \\ldots, n - 1\\}}`, such
    that :math:`f^{(k)} = f^{(k + 1)}` for some :math:`k`, find the 1D array
    corresponding to :math:`f^{(k)}`."""
    terminal_states = np.empty_like(mapping)
    for i, initial_target_idx in enumerate(mapping):
        temp_target_idx = i
        next_target_idx = initial_target_idx
        while temp_target_idx != next_target_idx:
            temp_target_idx = mapping[temp_target_idx]
            next_target_idx = mapping[mapping[temp_target_idx]]
        terminal_states[i] = temp_target_idx

    return terminal_states


class Nerve(BaseEstimator, TransformerMixin):
    """1-skeleton of the nerve of a refined Mapper cover, i.e. the Mapper
    graph.

    This transformer is the final step in the
    :class:`gtda.mapper.pipeline.MapperPipeline` objects created
    by :func:`gtda.mapper.make_mapper_pipeline`. It corresponds the last two
    arrows in `this diagram <../../../../_images/mapper_pipeline.svg>`_.

    Parameters
    ----------
    min_intersection : int, optional, default: ``1``
        Minimum size of the intersection, between data subsets associated to
        any two Mapper nodes, required to create an edge between the nodes in
        the Mapper graph. Must be positive.

    store_edge_elements : bool, optional, default: ``False``
        Whether the indices of data elements associated to Mapper edges (i.e.
        in the intersections allowed by `min_intersection`) should be stored in
        the :class:`igraph.Graph` object output by :meth:`fit_transform`. When
        ``True``, might lead to a large :class:`igraph.Graph` object.

    contract_nodes : bool, optional, default: ``False``
        If ``True``, any node representing a cluster which is a strict subset
        of the cluster corresponding to another node is eliminated, and only
        one maximal node is kept.

    Attributes
    ----------
    graph_ : :class:`igraph.Graph` object
        Mapper graph obtained from the input data. Created when :meth:`fit` is
        called.

    """

    def __init__(self, min_intersection=1, store_edge_elements=False,
                 contract_nodes=False):
        self.min_intersection = min_intersection
        self.store_edge_elements = store_edge_elements
        self.contract_nodes = contract_nodes

    def fit(self, X, y=None):
        """Compute the Mapper graph as in :meth:`fit_transform`, but store the
        graph as :attr:`graph_` and return the estimator.

        Parameters
        ----------
        X : list of list of tuple
            See :meth:`fit_transform`.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        self.graph_ = self.fit_transform(X, y=y)
        return self

    def fit_transform(self, X, y=None):
        """Construct a Mapper graph from a refined Mapper cover.

        Parameters
        ----------
        X : ndarray of shape (n_samples,)
            Cluster labels describing a refined cover of a dataset produced by
            the clustering step of a :class:`gtda.mapper.MapperPipeline`,
            as depicted in
            `this diagram <../../../../_images/mapper_pipeline.svg>`_. Each
            entry in `X` is a tuple of pairs of the form
            ``(pullback_set_label, partial_cluster_label)`` where
            ``partial_cluster_label`` is a cluster label within the pullback
            cover set identified by ``pullback_set_label``. The unique pairs
            correspond to nodes in the output graph.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        graph : :class:`igraph.Graph` object
            Undirected Mapper graph according to `X` and `min_intersection`.
            Each node is an :class:`igraph.Vertex` object with attributes
            ``"pullback_set_label"``, ``"partial_cluster_label"`` and
            ``"node_elements"``. Each edge is an :class:`igraph.Edge` object
            with a ``"weight"`` attribute which is equal to the size of the
            intersection between the data subsets associated to its two nodes.
            If `store_edge_elements` is ``True`` each edge also has an
            additional attribute ``"edge_elements"``.

        """
        # TODO: Include a validation step for X
        # Graph construction -- vertices with their metadata
        labels_to_indices = defaultdict(list)
        for i, sample in enumerate(X):
            for node_id_pair in sample:
                labels_to_indices[node_id_pair].append(i)
        labels_to_indices = {key: np.array(value)
                             for key, value in labels_to_indices.items()}
        n_nodes = len(labels_to_indices)
        graph = Graph(n_nodes)

        # labels_to_indices is a dictionary of, say, N key-value pairs of the
        # form (pullback_set_label, partial_cluster_label): node_elements.
        # Hence, zip(*labels_to_indices) generates two tuples of length N, each
        # corresponding to a type of node attribute in the final graph.
        node_attributes = zip(*labels_to_indices)
        graph.vs["pullback_set_label"] = next(node_attributes)
        graph.vs["partial_cluster_label"] = next(node_attributes)
        graph.vs["node_elements"] = [*labels_to_indices.values()]

        # Graph construction -- edges with weights given by intersection sizes.
        # In general, we need all information in `nodes` to narrow down the set
        # of combinations to check, especially when `contract_nodes` is True.
        nodes = zip(*zip(*labels_to_indices), labels_to_indices.values())
        node_index_pairs, weights, intersections, mapping = \
            self._generate_edge_data(nodes, n_nodes)
        graph.es["weight"] = 1
        graph.add_edges(node_index_pairs)
        graph.es["weight"] = weights
        if self.store_edge_elements:
            graph.es["edge_elements"] = intersections
        if self.contract_nodes:
            # Due to the order in which itertools.combinations produces pairs,
            # and to the preference given to node 1 in the if-elif-else clause
            # in `_subset_check_metadata_append`, `mapping` is guaranteed to
            # send everything to one of its fixed points after sufficiently
            # many repeated applications and, by construction, no two pairs of
            # indices in `_limit_mapping(mapping)` can correspond to data
            # subsets which are in a subset relation. Thus the nodes are
            # correctly contracted by `_limit_mapping(mapping)`.
            limit_mapping = _limit_mapping(mapping)
            graph.contract_vertices(limit_mapping,
                                    combine_attrs="first")
            graph.delete_vertices([i for i in graph.vs.indices
                                   if i != limit_mapping[i]])

        return graph

    def _generate_edge_data(self, nodes, n_nodes):
        def _in_same_pullback_set(_node_tuple):
            return _node_tuple[0][1][0] == _node_tuple[1][1][0]

        def _do_nothing(*args):
            pass

        def _intersections_append(_intersection):
            return intersections.append(_intersection)

        def _metadata_append(
                _node_1_idx, _node_2_idx, _intersection_size, _intersection,
                *args
                ):
            if _intersection_size >= self.min_intersection:
                # Add edge (as a node tuple) to list of node index pairs
                node_index_pairs.append((_node_1_idx, _node_2_idx))
                weights.append(_intersection_size)
                intersection_behavior(_intersection)

        def _subset_check_metadata_append(
                _node_1_idx, _node_2_idx, _intersection_size, _intersection,
                _node_1_elements, _node_2_elements
                ):
            if _intersection_size == len(_node_2_elements):
                # Node 2 is contained in node 1 and we remove it in favour of
                # node 1.
                mapping[_node_2_idx] = _node_1_idx
            elif _intersection_size == len(_node_1_elements):
                # Node 1 is strictly contained in node 2 and we remove it in
                # favour of node 2.
                mapping[_node_1_idx] = _node_2_idx
            else:
                # Edge exists provided `_intersection_size` is large enough
                _metadata_append(_node_1_idx, _node_2_idx, _intersection_size,
                                 _intersection)

        node_tuples = combinations(enumerate(nodes), 2)

        node_index_pairs = []
        weights = []
        intersections = []

        # Choose whether intersections are stored or not.
        # `intersection_behavior` is in scope for `_metadata_append` and
        # `_subset_check_metadata_append`.
        if self.store_edge_elements:
            intersection_behavior = _intersections_append
        else:
            intersection_behavior = _do_nothing

        if self.contract_nodes:
            mapping = np.arange(n_nodes)
            behavior = _subset_check_metadata_append
        else:
            mapping = None
            behavior = _metadata_append

        # No need to check for intersections within each pullback set as the
        # input is assumed to be a refined Mapper cover
        for node_tuple in filterfalse(_in_same_pullback_set, node_tuples):
            ((node_1_idx, (_, _, node_1_elements)),
             (node_2_idx, (_, _, node_2_elements))) = node_tuple
            intersection = np.intersect1d(node_1_elements, node_2_elements)
            intersection_size = len(intersection)

            if intersection_size:
                behavior(node_1_idx, node_2_idx, intersection_size,
                         intersection, node_1_elements, node_2_elements)
            else:
                continue

        return node_index_pairs, weights, intersections, mapping
