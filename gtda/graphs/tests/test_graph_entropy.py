import numpy as np
import pytest

from sklearn.exceptions import NotFittedError
from gtda.graphs.create_clique_complex import (CreateCliqueComplex,
                                               CreateLaplacianMatrices)
from gtda.graphs.graph_entropy import GraphEntropy
from gtda.graphs.heat_diffusion import HeatDiffusion

taus = np.linspace(0, 5, 20)

X = np.random.random((10, 2))
alpha = 0.4
cc = CreateCliqueComplex(data=X, alpha=alpha, data_type='cloud')
cd = cc.create_complex_from_graph()


def test_graph_entropy_fitted():
    l_n = CreateLaplacianMatrices().fit(cd, (0,)).transform(cd)[0]
    heat_vectors = HeatDiffusion().fit(l_n, taus=taus).transform(l_n)
    entropy = GraphEntropy()

    with pytest.raises(NotFittedError):
        entropy.transform(heat_vectors)
