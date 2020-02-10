# License: GNU AGPLv3

import gtda.time_series as ts
import gtda.homology as hl
import gtda.diagrams as diag
from gtda.pipeline import Pipeline
import numpy as np
from numpy.testing import assert_almost_equal

from sklearn.model_selection import TimeSeriesSplit
import sklearn.preprocessing as skprep
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

data = np.random.rand(600, 1)


def split_train_test(data):
    n_train = int(0.7 * data.shape[0])
    n_test = data.shape[0] - n_train
    labeller = ts.Labeller(width=5, percentiles=[80],
                           n_steps_future=1)
    X_train = data[:n_train]
    y_train = X_train
    X_train, y_train = labeller.fit_transform_resample(X_train, y_train)

    X_test = data[n_train:n_train + n_test]
    y_test = X_test
    X_test, y_test = labeller.fit_transform_resample(X_test, y_test)

    return X_train, y_train, X_test, y_test


def get_steps():
    steps = [
        ('embedding', ts.TakensEmbedding()),
        ('window', ts.SlidingWindow(width=5, stride=1)),
        ('diagram', hl.VietorisRipsPersistence()),
        ('rescaler', diag.Scaler()),
        ('filter', diag.Filtering(epsilon=0.1)),
        ('entropy', diag.PersistenceEntropy()),
        ('scaling', skprep.MinMaxScaler(copy=True))
    ]
    return steps


def get_param_grid():
    embedding_param = {}
    window_param = {}
    diagram_param = {}
    classification_param = {}

    window_param['width'] = [2, 3]
    diagram_param['homology_dimensions'] = [[0, 1]]
    classification_param['n_estimators'] = [10, 100]

    embedding_param_grid = {'embedding__' + k: v
                            for k, v in embedding_param.items()}
    diagram_param_grid = {'diagram__' + k: v
                          for k, v in diagram_param.items()}
    classification_param_grid = {'classification__' + k: v
                                 for k, v in classification_param.items()}

    param_grid = {**embedding_param_grid, **diagram_param_grid,
                  **classification_param_grid}

    return param_grid


def test_pipeline_time_series():
    X_train, y_train, X_test, y_test = split_train_test(data)

    steps = get_steps()
    pipeline = Pipeline(steps)
    X_train_final, y_train_final = pipeline.\
        fit_transform_resample(X_train, y_train)

    # Running the pipeline step by step
    X_train_temp, y_train_temp = X_train, y_train
    for _, transformer in steps:
        if hasattr(transformer, 'fit_transform_resample'):
            X_train_temp, y_train_temp = transformer.\
                fit_transform_resample(X_train_temp, y_train_temp)
        else:
            X_train_temp = transformer.\
                fit_transform(X_train_temp, y_train_temp)

    assert_almost_equal(X_train_final, X_train_temp)
    assert_almost_equal(y_train_final, y_train_temp)

    pipeline.fit(X_train, y_train)
    X_test_final, y_test_final = pipeline.transform_resample(X_test, y_test)

    X_test_temp, y_test_temp = X_test, y_test
    for _, transformer in steps:
        if hasattr(transformer, 'transform_resample'):
            X_test_temp, y_test_temp = transformer.\
                transform_resample(X_test_temp, y_test_temp)
        else:
            X_test_temp = transformer.transform(X_test_temp)

    assert_almost_equal(X_test_final, X_test_temp)
    assert_almost_equal(y_test_final, y_test_temp)


def test_grid_search_time_series():
    X_train, y_train, X_test, y_test = split_train_test(data)

    steps = get_steps() + [('classification', RandomForestClassifier())]
    pipeline = Pipeline(steps)
    param_grid = get_param_grid()
    cv = TimeSeriesSplit(n_splits=2)
    grid = GridSearchCV(
        estimator=pipeline, param_grid=param_grid, cv=cv, verbose=0)
    grid.fit(X_train, y_train)
