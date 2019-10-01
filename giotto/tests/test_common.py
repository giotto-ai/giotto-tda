from sklearn.utils.estimator_checks import parametrize_with_checks
from giotto.graphs import GraphGeodesicDistance


@parametrize_with_checks([GraphGeodesicDistance])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
