import numpy as np
import pytest
import networkx as nx


from sklearn.exceptions import NotFittedError
from giotto.graphs.create_clique_complex import CreateCliqueComplex,\
    CreateLaplacianMatrices, CreateBoundaryMatrices

from numpy.testing import assert_almost_equal
from scipy.spatial import distance_matrix

X = np.random.random((10, 2))
alpha = 0.4
cc = CreateCliqueComplex(data=X, alpha=alpha, data_type='cloud')
cd = cc.create_complex_from_graph()


def test_graph_input_with_data_type():

    with pytest.raises(ValueError):
        CreateCliqueComplex(
            graph=nx.barbell_graph(10, 10), data_type='cloud')


def test_data_input_with__graph_data_type():

    with pytest.raises(ValueError):
        CreateCliqueComplex(data=X, alpha=alpha)


def test_input_distance_matrix():
    dist = distance_matrix(X, X, p=2)

    cc_ = CreateCliqueComplex(data=dist, alpha=alpha, data_type='matrix')
    cd_ = cc_.create_complex_from_graph()

    lap_cc = CreateLaplacianMatrices().fit(cd, (0,)).transform(cd)[0].todense()
    lap_cc_ = CreateLaplacianMatrices().fit(
        cd_, (0,)).transform(cd_)[0].todense()

    assert_almost_equal(lap_cc, lap_cc_)


def test_adjacency_matrix():
    g = nx.barbell_graph(8, 5)
    cc_ = CreateCliqueComplex(g)
    adjacency = nx.adjacency_matrix(g)

    assert_almost_equal(adjacency.todense(),
                        cc_.get_adjacent_matrix().todense())


def test_create_boundary_matrices_not_fitted():
    bound = CreateBoundaryMatrices()

    with pytest.raises(NotFittedError):
        bound.transform(cd)


def test_create_laplacian_matrices_not_fitted():
    lap = CreateLaplacianMatrices()

    with pytest.raises(NotFittedError):
        lap.transform(cd)


def test_create_laplacian_matrices_symmetry():
    lap = CreateLaplacianMatrices().fit(cd, (0,)).transform(cd)

    assert_almost_equal(lap[0].todense(), lap[0].todense().T)
