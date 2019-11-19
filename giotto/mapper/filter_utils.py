from sklearn.base import BaseEstimator, TransformerMixin
from scipy.spatial.distance import pdist, squareform
import numpy as np

class Eccentricity(BaseEstimator, TransformerMixin):
    def __init__(self, exp=2, metric='euclidean', metric_params={}):
        self.exp = exp
        self.metric = metric
        self.metric_params = metric_params

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        dm = squareform(pdist(X, metric=self.metric, **self.metric_params))
        Xt = np.linalg.norm(dm, axis=1, ord=self.exp)
        return Xt


# def project(
#         self,
#         X,
#         projection="sum",
#         scaler="default:MinMaxScaler",
#         distance_matrix=None,
# ):
#     """Creates the projection/lens from a dataset. Input the data set. Specify a projection/lens type. Output the projected data/lens.
#     Parameters
#     ----------
#     X : Numpy Array
#         The data to fit a projection/lens to.
#     projection :
#         Projection parameter is either a string, a Scikit-learn class with fit_transform, like manifold.TSNE(), or a list of dimension indices. A string from ["sum", "mean", "median", "max", "min", "std", "dist_mean", "l2norm", "knn_distance_n"]. If using knn_distance_n write the number of desired neighbors in place of n: knn_distance_5 for summed distances to 5 nearest neighbors. Default = "sum".
#     scaler : Scikit-Learn API compatible scaler.
#         Scaler of the data applied after mapping. Use None for no scaling. Default = preprocessing.MinMaxScaler() if None, do no scaling, else apply scaling to the projection. Default: Min-Max scaling
#     distance_matrix : Either str or None
#         If not None, then any of ["braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "dice", "euclidean", "hamming", "jaccard", "kulsinski", "mahalanobis", "matching", "minkowski", "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath", "sqeuclidean", "yule"].
#         If False do nothing, else create a squared distance matrix with the chosen metric, before applying the projection.
#     Returns
#     -------
#     lens : Numpy Array
#         projected data.
#     Examples
#     --------
#     >>> # Project by taking the first dimension and third dimension
#     >>> X_projected = mapper.project(
#     >>>     X_inverse,
#     >>>     projection=[0,2]
#     >>> )
#     >>> # Project by taking the sum of row values
#     >>> X_projected = mapper.project(
#     >>>     X_inverse,
#     >>>     projection="sum"
#     >>> )
#     >>> # Do not scale the projection (default is minmax-scaling)
#     >>> X_projected = mapper.project(
#     >>>     X_inverse,
#     >>>     scaler=None
#     >>> )
#     >>> # Project by standard-scaled summed distance to 5 nearest neighbors
#     >>> X_projected = mapper.project(
#     >>>     X_inverse,
#     >>>     projection="knn_distance_5",
#     >>>     scaler=sklearn.preprocessing.StandardScaler()
#     >>> )
#     >>> # Project by first two PCA components
#     >>> X_projected = mapper.project(
#     >>>     X_inverse,
#     >>>     projection=sklearn.decomposition.PCA()
#     >>> )
#     >>> # Project by first three UMAP components
#     >>> X_projected = mapper.project(
#     >>>     X_inverse,
#     >>>     projection=umap.UMAP(n_components=3)
#     >>> )
#     >>> # Project by L2-norm on squared Pearson distance matrix
#     >>> X_projected = mapper.project(
#     >>>     X_inverse,
#     >>>     projection="l2norm",
#     >>>     distance_matrix="pearson"
#     >>> )
#     >>> # Mix and match different projections
#     >>> X_projected = np.c_[
#     >>>     mapper.project(X_inverse, projection=sklearn.decomposition.PCA()),
#     >>>     mapper.project(X_inverse, projection="knn_distance_5")
#     >>> ]
#     """
#
#     # Sae original values off so they can be referenced by later functions in the pipeline
#     self.inverse = X
#     scaler = preprocessing.MinMaxScaler() if scaler == "default:MinMaxScaler" else scaler
#     self.scaler = scaler
#     self.projection = str(projection)
#     self.distance_matrix = distance_matrix
#
#     if self.verbose > 0:
#         print("..Projecting on data shaped %s" % (str(X.shape)))
#
#     # If distance_matrix is a scipy.spatial.pdist string, we create a square distance matrix
#     # from the vectors, before applying a projection.
#     if self.distance_matrix in [
#         "braycurtis",
#         "canberra",
#         "chebyshev",
#         "cityblock",
#         "correlation",
#         "cosine",
#         "dice",
#         "euclidean",
#         "hamming",
#         "jaccard",
#         "kulsinski",
#         "mahalanobis",
#         "matching",
#         "minkowski",
#         "rogerstanimoto",
#         "russellrao",
#         "seuclidean",
#         "sokalmichener",
#         "sokalsneath",
#         "sqeuclidean",
#         "yule",
#     ]:
#         X = distance.squareform(distance.pdist(X, metric=distance_matrix))
#         if self.verbose > 0:
#             print(
#                 "Created distance matrix, shape: %s, with distance metric `%s`"
#                 % (X.shape, distance_matrix)
#             )
#
#     # Detect if projection is a class (for scikit-learn)
#     try:
#         p = projection.get_params()  # fail quickly
#         reducer = projection
#         if self.verbose > 0:
#             try:
#                 projection.set_params(**{"verbose": self.verbose})
#             except:
#                 pass
#             print("\n..Projecting data using: \n\t%s\n" % str(projection))
#         X = reducer.fit_transform(X)
#     except:
#         pass
#
#     # What is this used for?
#     if isinstance(projection, tuple):
#         X = self._process_projection_tuple(projection)
#
#     # Detect if projection is a string (for standard functions)
#     # TODO: test each one of these projections
#     if isinstance(projection, str):
#         if self.verbose > 0:
#             print("\n..Projecting data using: %s" % (projection))
#
#         def dist_mean(X, axis=1):
#             X_mean = np.mean(X, axis=0)
#             X = np.sum(np.sqrt((X - X_mean) ** 2), axis=1)
#             return X
#
#         projection_funcs = {
#             "sum": np.sum,
#             "mean": np.mean,
#             "median": np.median,
#             "max": np.max,
#             "min": np.min,
#             "std": np.std,
#             "l2norm": np.linalg.norm,
#             "dist_mean": dist_mean,
#         }
#
#         if projection in projection_funcs.keys():
#             X = projection_funcs[projection](X, axis=1).reshape(
#                 (X.shape[0], 1))
#
#         if "knn_distance_" in projection:
#             n_neighbors = int(projection.split("_")[2])
#             if (
#                     self.distance_matrix
#             ):  # We use the distance matrix for finding neighbors
#                 X = np.sum(np.sort(X, axis=1)[:, :n_neighbors],
#                            axis=1).reshape(
#                     (X.shape[0], 1)
#                 )
#             else:
#                 from sklearn import neighbors
#
#                 nn = neighbors.NearestNeighbors(n_neighbors=n_neighbors)
#                 nn.fit(X)
#                 X = np.sum(
#                     nn.kneighbors(X, n_neighbors=n_neighbors,
#                                   return_distance=True)[
#                         0
#                     ],
#                     axis=1,
#                 ).reshape((X.shape[0], 1))
#
#     # Detect if projection is a list (with dimension indices)
#     if isinstance(projection, list):
#         if self.verbose > 0:
#             print("\n..Projecting data using: %s" % (str(projection)))
#         X = X[:, np.array(projection)]
#
#     # If projection produced sparse output, turn into a dense array
#     if issparse(X):
#         X = X.toarray()
#         if self.verbose > 0:
#             print("\n..Created projection shaped %s" % (str(X.shape)))
#
#     # Scaling
#     if scaler is not None:
#         if self.verbose > 0:
#             print("\n..Scaling with: %s\n" % str(scaler))
#         X = scaler.fit_transform(X)
#
#     return X
