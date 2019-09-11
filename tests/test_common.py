from sklearn.utils.estimator_checks import parametrize_with_checks
from giotto.images import HeightFiltration


@parametrize_with_checks([HeightFiltration])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
