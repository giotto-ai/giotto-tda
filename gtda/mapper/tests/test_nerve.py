import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import arrays, array_shapes
from hypothesis.strategies import floats
from gtda.mapper.pipeline import make_mapper_pipeline


@given(X=arrays(dtype=np.float, unique=True,
                elements=floats(allow_nan=False,
                                allow_infinity=False,
                                min_value=-1e10,
                                max_value=1e10
                                ),
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
    disjoint_nodes = [set(graph['node_metadata']['node_elements'][node_1])
                      .isdisjoint(graph['node_metadata']['node_elements']
                                  [node_2])
                      for node_1, node_2 in graph.get_edgelist()]

    # Check if there is a disjoint node pair given by an edge.
    assert not any(disjoint_nodes)
