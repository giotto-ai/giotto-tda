from hypothesis import given, settings
from hypothesis.strategies import floats, integers, composite, lists

from gtda.local_homology.simplicial import KNeighborsLocalVietorisRips, \
      RadiusLocalVietorisRips


@composite
def gen_3d_point_cloud(draw):
    """ Generates point clouds as lists of floats in the unit cube of
        size between 2 and 20."""
    three_floats = lists(floats(min_value=0, max_value=1), min_size=3,
                         max_size=3, unique=True)
    return draw(lists(three_floats, min_size=2, max_size=20))


@composite
def gen_dimensions(draw):
    """ Generates dimension as tuples of integers."""
    return tuple(draw(lists(integers(min_value=1, max_value=10),
                      min_size=1, max_size=5, unique=True)))


@composite
def gen_epsilon(draw):
    """ Generates radii as floats. """
    epsilon1 = draw(floats(min_value=0, max_value=1))
    # max to in min_value to avoid warning
    epsilon2 = draw(floats(min_value=max(epsilon1 + 1e-3, 1e-3),
                           max_value=1 + 2e-3))
    return (epsilon1, epsilon2)


@composite
def gen_n_neighbors(draw):
    """Generates number of neighbors as integers. """
    n_neighbor1 = draw(integers(min_value=1, max_value=20))
    n_neighbor2 = draw(integers(min_value=n_neighbor1+1, max_value=30))
    return (n_neighbor1, n_neighbor2)


@settings(deadline=None)
@given(point_cloud=gen_3d_point_cloud(),
       point_cloud2=gen_3d_point_cloud(),
       dims=gen_dimensions(),
       n_neighbors=gen_n_neighbors())
def test_KNeighborsLocalVietoris(point_cloud, point_cloud2, dims,
                                 n_neighbors):
    # fit transform on same point cloud:
    X = point_cloud

    # 'min' below to avoid warnings
    n_neighbors = (min(len(X)-1, n_neighbors[0]),
                   min(len(X), n_neighbors[1]))

    lh_kn = KNeighborsLocalVietorisRips(metric='euclidean',
                                        n_neighbors=n_neighbors,
                                        homology_dimensions=dims,
                                        n_jobs=-1)
    lh_kn.fit(X)
    lh_kn.transform(X)
    lh_kn = KNeighborsLocalVietorisRips(metric='euclidean',
                                        n_neighbors=n_neighbors,
                                        homology_dimensions=dims,
                                        collapse_edges=True,
                                        n_jobs=-1)
    lh_kn.fit_transform(X)

    # fit and transform on different point clouds:
    Y = point_cloud2
    lh_kn = KNeighborsLocalVietorisRips(metric='euclidean',
                                        n_neighbors=n_neighbors,
                                        homology_dimensions=dims,
                                        n_jobs=-1)
    lh_kn.fit(X)
    lh_kn.transform(Y)


@settings(deadline=None)
@given(point_cloud=gen_3d_point_cloud(),
       point_cloud2=gen_3d_point_cloud(),
       dims=gen_dimensions(),
       radii=gen_epsilon())
def test_RadiusLocalVietoris(point_cloud, point_cloud2, dims, radii):
    # fit transform on same point cloud:
    X = point_cloud

    lh_rad = RadiusLocalVietorisRips(metric='euclidean',
                                     radii=radii,
                                     homology_dimensions=dims,
                                     collapse_edges=True,
                                     n_jobs=-1)
    lh_rad.fit(X)
    lh_rad.transform(X)
    lh_rad = RadiusLocalVietorisRips(metric='euclidean',
                                     radii=radii,
                                     homology_dimensions=dims,
                                     n_jobs=-1)
    lh_rad.fit_transform(X)

    # fit and transform on different point clouds:
    Y = point_cloud2
    lh_rad = RadiusLocalVietorisRips(metric='euclidean',
                                     radii=radii,
                                     homology_dimensions=dims,
                                     collapse_edges=True,
                                     n_jobs=-1)
    lh_rad.fit(X)
    lh_rad.transform(Y)
