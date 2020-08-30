"""Testing for Nerve (Mapper graph construction)."""
# License: GNU AGPLv3

import numpy as np
import pytest
from hypothesis import given
from hypothesis.extra.numpy import arrays, array_shapes
from hypothesis.strategies import floats

from gtda.mapper.pipeline import make_mapper_pipeline


@given(X=arrays(dtype=np.float, unique=True,
                elements=floats(allow_nan=False,
                                allow_infinity=False,
                                min_value=-1e10,
                                max_value=1e10),
                shape=array_shapes(min_dims=2, max_dims=2, min_side=11)))
def test_node_intersection(X):
    # TODO: Replace pipe and graph by Nerve transformer
    # TODO: Improve the Hypothesis strategy to avoid needing to hardcode the
    # min_side to be greater than n_intervals (10 by default).
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


@given(X=arrays(dtype=np.float, unique=True,
                elements=floats(allow_nan=False,
                                allow_infinity=False,
                                min_value=-1e10,
                                max_value=1e10),
                shape=array_shapes(min_dims=2, max_dims=2, min_side=11)))
def test_edge_elements(X):
    # TODO: Replace pipe and graph by Nerve transformer
    # TODO: Improve the Hypothesis strategy to avoid needing to hardcode the
    # min_side to be greater than n_intervals (10 by default).
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
    assert all([
        np.array_equal(node, node_ee)
        for node, node_ee in zip(node_elements, node_elements_ee)
    ])
    assert graph.vs.indices == graph_edge_elems.vs.indices
    # Edges
    assert graph.es.indices == graph_edge_elems.es.indices
    assert graph.es["weight"] == graph_edge_elems.es["weight"]
    assert all([
        edge.tuple == edge_ee.tuple
        for edge, edge_ee in zip(graph.es, graph_edge_elems.es)
    ])

    # Check that the arrays edge_elements contain precisely those indices which
    # are in the element sets associated to both the first and second vertex,
    # and that the edge weight equals the size of edge_elements.
    flag = True
    for edge in graph_edge_elems.es:
        v1, v2 = edge.vertex_tuple
        flag *= np.array_equal(
            edge["edge_elements"],
            np.intersect1d(v1["node_elements"], v2["node_elements"]),
        )
        flag *= len(edge["edge_elements"]) == edge["weight"]
    assert flag


@pytest.mark.parametrize("min_intersection", [1, 2, 3, 10])
@given(X=arrays(dtype=np.float, unique=True,
                elements=floats(allow_nan=False,
                                allow_infinity=False,
                                min_value=-1e10,
                                max_value=1e10),
                shape=array_shapes(min_dims=2, max_dims=2, min_side=11)))
def test_min_intersection(X, min_intersection):
    # TODO: Replace pipe and graph by Nerve transformer
    # TODO: Improve the Hypothesis strategy to avoid needing to hardcode the
    # min_side to be greater than n_intervals (10 by default).
    pipe = make_mapper_pipeline(min_intersection=min_intersection)
    graph = pipe.fit_transform(X)

    # Check that there are no edges with weight less than min_intersection
    assert all([x >= min_intersection for x in graph.es["weight"]])
