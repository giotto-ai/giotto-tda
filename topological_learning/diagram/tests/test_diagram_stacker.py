import pytest

import numpy as np

from topological_learning.diagram import DiagramStacker, DiagramScaler


@pytest.mark.parametrize("num_samples", [0, 1, 10])
def test_diagram_stacker(num_samples):
    stacker = DiagramStacker()

    X = {
        0: np.ones((num_samples, 8, 2)),
        1: np.zeros((num_samples, 3, 2))
    }

    result = stacker.fit_transform(X, None)
    assert list(result.keys()) == [None]
    assert result[None].shape == (num_samples, 11, 2)


@pytest.mark.parametrize("num_samples", [1, 10])
@pytest.mark.parametrize("norm", ["bottleneck", "landscape", "betti"])
def test_diagram_scaler(num_samples, norm):
    rng = np.random.mtrand.RandomState(42)

    X = {
        0: rng.randn(num_samples, 8, 2),
        1: rng.randn(num_samples, 3, 2)
    }

    scaler = DiagramScaler(norm=norm, function=np.max, n_jobs=1)
    Xtr = scaler.fit(X, None).transform(X, None)

    print(scaler._scale)

    Xinv = scaler.inverse_transform(Xtr)

    assert list(Xinv.keys()) == list(X.keys())
    for k in Xinv.keys():
        np.testing.assert_array_almost_equal(Xinv[k], X[k], decimal=10)
