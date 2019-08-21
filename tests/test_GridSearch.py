# Authors: Guillaume Tauzin <guillaume.tauzin@epfl.ch>
# License: TBD

import giotto as go
import giotto.time_series as ts
import giotto.diagram as diag
import giotto.homology as hl
import giotto.neural_network as nn
import giotto.manifold as ma
import giotto.compose as cp

import numpy as np
import pandas as pd
import datetime as dt
import sklearn as sk
import pickle as pkl
import argparse

import sklearn.utils as skutils
from sklearn.model_selection import TimeSeriesSplit, KFold
import sklearn.preprocessing as skprep
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from keras.models import Sequential
import keras.layers as klayers
import keras.optimizers as koptimizers

import keras
import tensorflow as tf

# If I don't do this, the GPU is automatically used and gets out of memory
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def init_keras_session(n_cpus_k, n_gpus_k):
    config = tf.ConfigProto(device_count={'CPU': n_cpus_k, 'GPU': n_gpus_k})

    sess = tf.Session(config=config)
    keras.backend.set_session(sess)


def get_data(n_train, n_test):
    data = np.random.rand(n_train + n_test, 1)
    stationarizing = ts.Stationarizer(stationarization_type='return')
    data = stationarizing.fit(data).transform(data)

    return data


def split_train_test(data, n_train, n_test):
    labeller = ts.Labeller(labelling_kwargs={'type': 'derivation', 'delta_t': 3}, window_size=5, percentiles=[80], n_steps_future=1)
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
        ('diagram', hl.VietorisRipsPersistence()),
        ('distance', diag.DiagramDistance()),
        ('physical', ma.StatefulMDS()),
        ('kinematics', ma.Kinematics()),
        ('scaling', skprep.MinMaxScaler(copy=True)),
        ('aggregator', cp.FeatureAggregator(is_keras=True)),
        ('classification', cp.TargetResamplingClassifier(classifier=nn.KerasClassifierWrapper(),
                                                         resampler=cp.TargetResampler()))
    ]
    return Pipeline(steps)


def get_param_grid():
    embedding_param = {}
    distance_param = {}
    diagram_param = {}
    physical_param = {}
    kinematics_param = {}
    scaling_param = {}
    aggregator_param = {}
    classification_param = {}

    embedding_param['outer_window_duration'] = [20, 30]

    diagram_param['homology_dimensions'] = [[0, 1]]
    distance_param['metric'] = ['bottleneck']
    distance_param['metric_params'] = [{'order': np.inf}]

    physical_param['n_components'] = [3]

    kinematics_param['orders'] = [[0, 1, 2]]

    aggregator_param['n_steps_in_past'] = [2]

    classification_param['layers_kwargs'] = [
        [
            {'layer': klayers.normalization.BatchNormalization},
            {'layer': layer, 'units': units, 'activation': 'tanh'},
            {'layer': klayers.Dense}
        ] for layer in [klayers.LSTM] for units in [1]]

    classification_param['optimizer_kwargs'] = [{'optimizer': optimizer, 'lr': lr} for optimizer in [koptimizers.RMSprop]
                                                for lr in [0.5]]
    classification_param['batch_size'] = [10]
    classification_param['epochs'] = [1]
    classification_param['loss'] = ['sparse_categorical_crossentropy']
    classification_param['metrics'] = [['sparse_categorical_accuracy']]

    embedding_param_grid = {'embedding__' + k: v for k, v in embedding_param.items()}
    diagram_param_grid = {'diagram__' + k: v for k, v in diagram_param.items()}
    distance_param_grid = {'distance__' + k: v for k, v in distance_param.items()}
    physical_param_grid = {'physical__' + k: v for k, v in physical_param.items()}
    kinematics_param_grid = {'kinematics__' + k: v for k, v in kinematics_param.items()}
    scaling_param_grid = {'scaling__' + k: v for k, v in scaling_param.items()}
    aggregator_param_grid = {'aggregator__' + k: v for k, v in aggregator_param.items()}
    classification_param_grid = {'classification__classifier__' + k: v for k, v in classification_param.items()}

    param_grid = [{**embedding_param_grid, **diagram_param_grid, **distance_param_grid, **physical_param_grid,
                   **kinematics_param_grid, **scaling_param_grid, **aggregator_param_grid, **classification_param_grid,
                   'embedding__outer_window_stride': [outer_window_stride], 'classification__resampler__step_size': [outer_window_stride]}
                  for outer_window_stride in [2, 5]]

    return param_grid


def run_grid_search(estimator, param_grid, X_train, y_train, number_splits, n_jobs):
    cv = TimeSeriesSplit(n_splits=number_splits)
    grid = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv, n_jobs=n_jobs, verbose=0)
    grid_result = grid.fit(X_train, y_train)

    return grid_result


def main(n_jobs):
    n_train, n_test = 400, 200
    data = get_data(n_train, n_test)
    X_train, y_train, X_test, y_test = split_train_test(data, n_train, n_test)
    pipeline = make_pipeline()

    param_grid = get_param_grid()

    grid_result = run_grid_search(pipeline, param_grid, X_train, y_train, number_splits=2, n_jobs=n_jobs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script for typical use of the giotto library")
    parser.add_argument('-n_jobs', help="Number of processes", type=int, required=True)
    parser.add_argument('-n_cpus_k', '--number_cpus_keras', help="Number of CPUs used to initialize keras session. Warning: This might lead to an explosion of thread allocations.", type=int, default=1)
    parser.add_argument('-n_gpus_k', '--number_gpus_keras', help="Number of GPUs used to initialize keras session. Warning: In the case of a grid search, this has lead to crashes as GPU memory got full.", type=int, default=0)
    args = vars(parser.parse_args())

    init_keras_session(args.pop('number_cpus_keras'), args.pop('number_gpus_keras'))
    main(**args)
