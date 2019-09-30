# Authors: Guillaume Tauzin <guillaume.tauzin@epfl.ch>
# License: Apache 2.0

import giotto as go
import giotto.time_series as ts
import giotto.homology as hl
import giotto.diagram as diag
from giotto.pipeline import Pipeline

import numpy as np
from numpy.testing import assert_almost_equal

import sklearn.utils as skutils
from sklearn.model_selection import TimeSeriesSplit
import sklearn.preprocessing as skprep
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

data = np.random.rand(600, 1)


def split_train_test(data):
    n_train = int(0.7 * data.shape[0])
    n_test = data.shape[0] - n_train
    labeller = ts.Labeller(labelling_kwargs={'type': 'derivation'},
                           window_size=5, percentiles=[80], n_steps_future=1)
    X_train = data[:n_train]
    y_train = X_train
    labeller.fit(y_train)
    y_train = labeller.transform(y_train)
    X_train = labeller.cut(X_train)

    X_test = data[n_train:n_train + n_test]
    y_test = X_test
    y_test = labeller.transform(y_test)
    X_test = labeller.cut(X_test)

    return X_train, y_train, X_test, y_test



def make_pipeline():
    steps = [
        ('embedding', ts.TakensEmbedder()),
        ('window', ts.SlidingWindow(width=5, stride=1)),
        ('diagram', hl.VietorisRipsPersistence()),
        ('rescaler', diag.DiagramScaler()),
        ('filter', diag.DiagramFilter(delta=0.1)),
        ('entropy', diag.PersistentEntropy()),
        ('scaling', skprep.MinMaxScaler(copy=True)),
        ('classification', RandomForestClassifier())
   ]
    return Pipeline(steps)

def get_param_grid():
    embedding_param = {}
    window_param = {}
    distance_param = {}
    diagram_param = {}
    classification_param = {}

    window_param['width'] = [2, 3]
    diagram_param['homology_dimensions'] = [[0, 1]]
    classification_param['n_estimators'] = [10, 100]

    embedding_param_grid = {'embedding__' + k: v
                            for k, v in embedding_param.items()}
    window_param_grid = {'window__' + k: v
                         for k, v in window_param.items()}
    diagram_param_grid = {'diagram__' + k: v
                          for k, v in diagram_param.items()}
    classification_param_grid = {'classification__classifier__' + k: v
                                 for k, v in classification_param.items()}

    param_grid = {**embedding_param_grid, **diagram_param_grid,
                  **classification_param_grid}

    return param_grid


def test_pipeline_time_series():
    X_train, y_train, X_test, y_test = split_train_test(data)

    pipeline = make_pipeline()
    X_train_predict, y_train_predict = pipeline.fit(X_train, y_train).\
        transform_resample(X_train)

    X_test_predict, y_test_predict = pipeline.transform_resample(X_test,
                                                                 y_test)

    # Running the pipeline step by step
    X_test_temp, y_test_temp = X_test, y_test
    for _, transformer in steps[:-1]:
        X_test_temp, y_test_temp = transformer.fit(X_temp, y_temp).\
            transform_resample(X_temp, y_temp)
    assert_almost_equal(X_test_predict, X_test_temp)
    assert_almost_equal(y_test_predict, y_test_temp)

def test_grid_search_time_series():
    X_train, y_train, X_test, y_test = split_train_test(data)

    pipeline = make_pipeline()
    param_grid = get_param_grid()
    cv = TimeSeriesSplit(n_splits=2)
    grid = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, verbose=0)
    grid_result = grid.fit(X_train, y_train)
