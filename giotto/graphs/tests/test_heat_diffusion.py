import numpy as np
import pytest

from sklearn.exceptions import NotFittedError
from giotto.graphs.create_clique_complex import (CreateCliqueComplex,
                                                 CreateLaplacianMatrices)
from giotto.graphs.heat_diffusion import HeatDiffusion

taus = np.linspace(0, 5, 20)

X = np.random.random((20, 2))
cc = CreateCliqueComplex(data=X, alpha=0.4, data_type='cloud')
cd = cc.create_complex_from_graph()
lap_node, lap_edge = CreateLaplacianMatrices().fit(cd, (0, 1)).transform(cd)


def test_heat_diffusion_not_fitted():
    diffusor = HeatDiffusion()

    with pytest.raises(NotFittedError):
        diffusor.transform(lap_node)


def test_heat_vectors_shape():
    heat_node = HeatDiffusion().fit(lap_node, taus).transform(lap_node)
    heat_edge = HeatDiffusion().fit(lap_edge, taus).transform(lap_edge)

    assert heat_node.shape == (lap_node.shape[0], lap_node.shape[0], len(taus))
    assert heat_edge.shape == (lap_edge.shape[0], lap_edge.shape[0], len(taus))


def test_heat_diffusion_initial_condition_shape():
    # Check with 5 random initial conditions
    ic = np.random.random((20, 5))

    heat_node = HeatDiffusion().fit(
        lap_node, taus, initial_condition=ic).transform(lap_node)

    assert heat_node.shape == (lap_node.shape[0], ic.shape[1], len(taus))
