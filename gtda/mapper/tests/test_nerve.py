"""Testing for Nerve (Mapper graph construction)."""
# License: GNU AGPLv3

import numpy as np
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis.extra.numpy import arrays, array_shapes
from hypothesis.strategies import floats
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_circles

from gtda.mapper import Projection, OneDimensionalCover, make_mapper_pipeline


mapper_input = arrays(
    dtype=float, unique=True, elements=floats(
        allow_nan=False, allow_infinity=False,
        min_value=-1e6, max_value=1e6
    ),
    shape=array_shapes(min_dims=2, max_dims=2, min_side=8, max_side=12)
)


hypothesis_settings = dict(
    deadline=5000, suppress_health_check=(HealthCheck.data_too_large,)
)


@settings(**hypothesis_settings)
@given(X=mapper_input)
def test_node_intersection(X):
    # TODO: Replace pipe and graph by Nerve transformer
    pipe = make_mapper_pipeline()
    graph = pipe.fit_transform(X)

    # Check if the elements of nodes defining an edge are disjoint or not:
    # If True, they are disjoint, i.e. the created edge is incorrect.
    # If all are False, all edges are correct.
    disjoint_nodes = [set(graph.vs['node_elements'][node_1])
                      .isdisjoint(graph.vs['node_elements'][node_2])
                      for node_1, node_2 in graph.get_edgelist()]

    # Check if there is a disjoint node pair given by an edge.
    assert not any(disjoint_nodes)


@settings(**hypothesis_settings)
@given(X=mapper_input)
def test_edge_elements(X):
    # TODO: Replace pipe and graph by Nerve transformer
    pipe = make_mapper_pipeline()
    pipe_edge_elems = make_mapper_pipeline(store_edge_elements=True)

    graph = pipe.fit_transform(X)
    graph_edge_elems = pipe_edge_elems.fit_transform(X)

    # Check that when store_edge_elements=False (default) there is no
    # "edge_elements" attribute.
    with pytest.raises(KeyError):
        _ = graph.es["edge_elements"]

    # Check that graph and graph_ee agree otherwise
    # Vertices
    assert graph.vs.indices == graph_edge_elems.vs.indices
    for attr_name in ["pullback_set_label", "partial_cluster_label"]:
        assert graph.vs[attr_name] == graph_edge_elems.vs[attr_name]
    node_elements = graph.vs["node_elements"]
    node_elements_ee = graph_edge_elems.vs["node_elements"]
    assert all([np.array_equal(node, node_ee)
                for node, node_ee in zip(node_elements, node_elements_ee)])
    assert graph.vs.indices == graph_edge_elems.vs.indices
    # Edges
    assert graph.es.indices == graph_edge_elems.es.indices
    assert graph.es["weight"] == graph_edge_elems.es["weight"]
    assert all([edge.tuple == edge_ee.tuple
                for edge, edge_ee in zip(graph.es, graph_edge_elems.es)])

    # Check that the arrays edge_elements contain precisely those indices which
    # are in the element sets associated to both the first and second vertex,
    # and that the edge weight equals the size of edge_elements.
    flag = True
    for edge in graph_edge_elems.es:
        v1, v2 = edge.vertex_tuple
        flag *= np.array_equal(
            edge["edge_elements"],
            np.intersect1d(v1["node_elements"], v2["node_elements"])
            )
        flag *= len(edge["edge_elements"]) == edge["weight"]
    assert flag


@settings(**hypothesis_settings)
@pytest.mark.parametrize("min_intersection", [1, 3, 5])
@given(X=mapper_input)
def test_min_intersection(X, min_intersection):
    # TODO: Replace pipe and graph by Nerve transformer
    pipe = make_mapper_pipeline(min_intersection=min_intersection)
    graph = pipe.fit_transform(X)

    # Check that there are no edges with weight less than min_intersection
    assert all([x >= min_intersection for x in graph.es["weight"]])


def test_contract_nodes():
    """Test that, on a pathological dataset, we generate a graph without edges
    when `contract_nodes` is set to False and with edges when it is set to
    True."""
    X = make_circles(n_samples=2000)[0]

    filter_func = Projection()
    cover = OneDimensionalCover(n_intervals=5, overlap_frac=0.4)
    p = filter_func.fit_transform(X)
    m = cover.fit_transform(p)

    gap = 0.1
    idx_to_remove = []
    for i in range(m.shape[1] - 1):
        inters = np.logical_and(m[:, i], m[:, i + 1])
        inters_idx = np.flatnonzero(inters)
        p_inters = p[inters_idx]
        min_p, max_p = np.min(p_inters), np.max(p_inters)
        idx_to_remove += list(
            np.flatnonzero((min_p <= p) & (p <= min_p + gap)))
        idx_to_remove += list(
            np.flatnonzero((max_p - gap <= p) & (p <= max_p)))

    X_f = X[[x for x in range(len(X)) if x not in idx_to_remove]]

    clusterer = DBSCAN(eps=0.05)
    pipe = make_mapper_pipeline(filter_func=filter_func,
                                cover=cover,
                                clusterer=clusterer,
                                contract_nodes=True)
    graph = pipe.fit_transform(X_f)
    assert not len(graph.es)

    pipe.set_params(contract_nodes=False)
    graph = pipe.fit_transform(X_f)
    assert len(graph.es)
