# License: Apache 2.0

import numpy as np
import networkx as nx

from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix, lil_matrix
from itertools import combinations
from sklearn.utils.validation import check_symmetric, check_is_fitted,\
    check_array
from sklearn.base import BaseEstimator, TransformerMixin


class CreateCliqueComplex:
    """Pre-processing step from unstructured data to graph and complex.

    Useful in graph analysis to convert general point clouds or distance
    matrices (numpy objects) into graphs (networkx objects) and clique
    complex (described as a dictionary).

    Parameters
    ----------
    graph : nx.graph, optional, default: ``None``
        Graph from which to compute the clique complex

    data : ndarray, shape (n_points, n_dimensions) or \
        (n_points, n_points)
        If ``data_type == 'cloud'``, the input should be an
        ndarray, where each entry along axis 0 is a vector with
        the features of the relative point. Otherwise when
        'matrix' is passed as data_type the object should be a symmetric
        square matrix with distances between points.

    alpha : float, optional, default: ``None``
        Real value to be used as a threshold while constructing
        the alpha-neighboring graph. For a point 'x' there will be an edge
        to each point 'y' if dist(x,y) < alpha, where here the
        distance is the euclidean.

    data_type : string, optional, default: ``'graph'``
        Type of raw data to be preprocessed. If set to 'graph'
        the Input 'graph' has to be a netowrkx graph object.
        If set to 'cloud' the data is to be interpreted as a collection
        of points where row index represents sample ID and columns the features
        If set to 'matrix' the data is to be interpreted as a
        distance matrix (symmetric) collecting element-wise distances
        between the elements.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import networkx as nx
    >>> X = np.random.random((20,2))
    >>> plt.scatter(X[:,0], X[:,1])
    >>> cc = CreateCliqueComplex(X, 0.5)
    >>> cd = cc.create_complex_from_graph()
    >>> nx.draw(cc.get_graph())

    """

    def __init__(self, graph=None, data=None, alpha=None, data_type='graph'):
        self.data = data
        self.alpha = alpha
        self.data_type = data_type
        self.complex_dict = dict()

        if data_type == 'graph':
            if not isinstance(graph, nx.Graph):
                raise ValueError("The parameter 'graph' should be "
                                 "a networkx Graph object")
            self.graph = graph
            self.adjacent_matrix = nx.adjacency_matrix(self.graph)
        else:
            check_array(data)
            if self.alpha is None:
                raise ValueError("If 'data_type' is not 'graph' the parameter"
                                 "alpha must be a float.")
            self.graph = self._create_graph()
            self.adjacent_matrix = csr_matrix(self.adjacent_matrix)

    def create_complex_from_graph(self):
        """
        Function that computes the Clique complex from the associated
        networkx Graph object.

        Returns
        -------
        self.complex_dict : dict
            Dictionary containing all simplices of the clique
            complex identified with arbitrary ID. Each entry 'K' of the
            dictionary is a dictionary containing all K-simplexes.

        """

        self._create_cliques_dict()

        return self.complex_dict

    def _create_cliques_dict(self):

        """
        Collect all cliques of a graph and create dictionary with all simplices
        """

        # Index for edges and nodes, arbitrary id to edges
        sim_complex = dict()

        sim_complex[1] = dict(zip(np.arange(
            nx.number_of_edges(self.graph)), self.graph.edges))
        sim_complex[0] = dict(zip(self.graph.nodes, self.graph.nodes))

        # Dictionary containing simplexes orders as indexes,
        # list of tuplas with node id forming the simplexes
        cliques = list(nx.enumerate_all_cliques(self.graph))

        for x in range(nx.number_of_nodes(self.graph) +
                       nx.number_of_edges(self.graph), len(cliques)):
            if sim_complex.get(len(cliques[x]) - 1) is None:
                i = 0
                sim_complex[(len(cliques[x]) - 1)] = dict()
                sim_complex[len(cliques[x]) - 1][i] = tuple(cliques[x])
                i += 1
            else:
                sim_complex[len(cliques[x]) - 1][i] = tuple(cliques[x])
                i += 1

        self.complex_dict = sim_complex

    def _create_graph(self):
        distance_to_adjacent = np.vectorize(lambda x:
                                            1 if x < self.alpha else 0)

        if self.data_type == 'cloud':
            self.adjacent_matrix = distance_to_adjacent(
                distance_matrix(self.data, self.data, p=2))
        else:
            check_symmetric(self.data)
            self.adjacent_matrix = distance_to_adjacent(self.data)

        self.adjacent_matrix = self.adjacent_matrix - np.identity(
            self.data.shape[0])
        return nx.from_numpy_matrix(self.adjacent_matrix)

    def get_adjacent_matrix(self):
        return self.adjacent_matrix

    def get_graph(self):
        return self.graph

    def get_complex_dict(self):
        return self.complex_dict


class CreateBoundaryMatrices(BaseEstimator, TransformerMixin):
    """Construction of Boundary Matrices from dictionary complex.

    This step computes the boundary matrices that can be used both to
    analyse the complex and compute the laplacian matrices.

    """

    def fit(self, X, orders, y=None):
        """Do nothing and return the estimator unchanged.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : dictionary
            Dictionary containing information on the Clique Complex of which
            computing the boundary matrices.

        orders : tuple
            Order of boundary matrixces

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        self.sizes_ = dict()
        for ele in X:
            self.sizes_[ele] = len(X[ele])
        self.orders_ = sorted(orders)

        return self

    def transform(self, X, y=None):
        """Compute boundary matrices.

        It computes the boundary matrices of specified orders. The
        orders are taken as element of the tuple 'orders'.

        Parameters
        ----------
        X : dictionary
            Dictionary containing information on the Clique Complex of which
            computing the boundary matrices.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        boundaries : list
            List containing the boundary matrices for all orders specified
            in tuple 'orders'.
        """
        check_is_fitted(self, ['sizes_', 'orders_'])
        boundaries = list()
        incidence = self._create_incidence_from_dict(X)

        for order, order_inc in incidence.items():
            if isinstance(self.orders_, int):
                orders = [self.orders_]
            else:
                orders = self.orders_
            if order in orders:
                # This is the boundary matrix from order to order-1 simplexes
                temp_mat = lil_matrix(
                    (self.sizes_[order - 1], self.sizes_[order]))
                for k, v in order_inc.items():
                    for x in v:
                        temp_mat[k, x[0]] = np.sign(x[1])
                boundaries.append(csr_matrix(temp_mat))

        return boundaries

    def _create_incidence_from_dict(self, complex_dict):

        # store incidence values
        incidence = dict()

        # edges
        incidence[1] = dict()
        # ID of edges (1-simplexes)
        idx = 0
        # Create incidence nodes to edges
        for x, y in complex_dict[1].items():
            i = 1
            for c in np.sort(y):

                if incidence[1].get(c) is not None:
                    incidence[1][c].append((idx, (-1) ** i))
                else:
                    incidence[1][c] = [(idx, (-1) ** i)]
                i -= 1
            idx += 1

        # Find incidence matrix for higher structures
        for order, order_list in complex_dict.items():
            if order > 1:

                # To check if this simplex has been already
                # put into the dictionary
                compare = dict((tuple(np.sort(y)), x) for x, y in
                               complex_dict[order - 1].items())
                # Create new dict for incidence from order-1 to order simplexes
                incidence[order] = dict()
                # Start from zero to label simplexes
                idx = 0

                for y, c in order_list.items():
                    # Order the set of vertices to impose the orientations
                    c = np.sort(c)
                    complex_dict[order][idx] = tuple(c)
                    # Keep track of the (-1) exponent
                    i = 0
                    # Find all faces of c (c contains order+1 vertices)
                    for x in combinations(c, order):
                        if incidence[order].get(compare[x]) is not None:
                            incidence[order][compare[x]].append(
                                (idx, (-1) ** i))
                        else:
                            incidence[order][compare[x]] = [(idx, (-1) ** i)]
                        i += 1
                    idx += 1

        return incidence


class CreateLaplacianMatrices(BaseEstimator, TransformerMixin):
    """Compute Laplacian matrices.

    This step computes the Laplacian matrices that can be used both to
    analyse the complex and compute the heat diffusion.


    """

    def fit(self, X, orders, y=None):
        """Set the orders parameters.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : dictionary
            Dictionary containing information on the Clique Complex of which
            computing the boundary matrices.

        orders : tuple
            Tuple containing the orders of Laplacian matrices to be computed

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        # Check if there is only one index
        if isinstance(orders, int):
            self.orders_ = [orders]
        else:
            self.orders_ = sorted(orders)

        self.bound_orders_ = set(self.orders_).union(
            set([x + 1 for x in self.orders_]))
        self.bound_orders_ = tuple(sorted(self.bound_orders_ - {0}))
        self.order_id_ = dict()

        for x, y in enumerate(self.bound_orders_):
            self.order_id_[y] = x

        return self

    def transform(self, X, y=None):
        """Compute Laplacians of complex.

        Compute Laplacians starting from a Graph Object up to certain
        order applying the formula from the boundary matrices

        Parameters
        ----------
         X : dictionary
            Dictionary containing information on the Clique Complex of which
            computing the boundary matrices.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        laplacians : list
            List containing the Laplacian matrices for all orders specified
            in tuple 'orders'.

        """
        check_is_fitted(self, ['orders_', 'bound_orders_', 'order_id_'])
        laplacians = list()

        cb = CreateBoundaryMatrices().fit(X, self.bound_orders_)
        boundaries = cb.transform(X)

        for x in self.orders_:
            # if maximal order, don't use the x+1 boundary,
            # if minimal don't use 0 boundaries
            if x > 0:
                lap = csr_matrix.transpose(boundaries[self.order_id_[x]]) *\
                      boundaries[self.order_id_[x]]
                if x < len(X)-1:
                    lap += boundaries[self.order_id_[x+1]] *\
                        csr_matrix.transpose(boundaries[self.order_id_[x+1]])
            else:
                lap = boundaries[self.order_id_[x+1]] *\
                      csr_matrix.transpose(boundaries[self.order_id_[x+1]])

            laplacians.append(lap)

        return laplacians
