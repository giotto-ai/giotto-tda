import numpy as np
from scipy import spatial as spatial
from sklearn.preprocessing import scale

def get_density_filtered_point_cloud(weight_array,
                                     k_th_nearest=200,
                                     percentage=0.3,
                                     user_axis=1):
    """
    Returns a density filtered point cloud of 9-vectors
    Parameters
    ----------
    weight_array: array
        An array of 9-dimensional points i.e the weights of 3x3 patches
    k_th_nearest : integer, optional, default: 200
        Indicates the k-th neighbour used as a density estimator
    percentage : integer, optional, default: 0.3
        The percentage of the point cloud cardinality to be left after filtration
    user_axis :
        integer 0 or 1 for scaling axis, 0 is column normalization, 1 is row norm.
    """
    normalized_weight_array = scale(weight_array, axis=user_axis)
    m_dimension, n_dimension = normalized_weight_array.shape
    number_of_kth_densest_points = np.int(percentage*m_dimension)
    condensed_weight_distance_matrix = spatial.distance.pdist(normalized_weight_array)
    redundant_weight_distance_matrix = spatial.distance.squareform(condensed_weight_distance_matrix)
    kth_nearest_index_matrix = np.argsort(redundant_weight_distance_matrix, axis=1)
    kth_nearest_index_vector = np.zeros((m_dimension, 1), dtype=int)
    for i in range(m_dimension):
        kth_nearest_index = kth_nearest_index_matrix[i][k_th_nearest]
        kth_nearest_index_vector[i] = kth_nearest_index
    # kth_nearest_distances = redundant_weight_distance_matrix[kth_nearest_index_vector]
    kth_nearest_distances = np.take_along_axis(redundant_weight_distance_matrix,
                                               kth_nearest_index_vector, axis=1)
    kth_nearest_indices_sorted = np.argsort(kth_nearest_distances, axis=0)
    p_of_kth_nearest_indices = kth_nearest_indices_sorted[0:number_of_kth_densest_points]
    pth_densest_points = normalized_weight_array[p_of_kth_nearest_indices]
    pth_densest_points = np.reshape(pth_densest_points, (-1, n_dimension))
    return pth_densest_points