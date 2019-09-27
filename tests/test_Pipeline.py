# Authors: Guillaume Tauzin <guillaume.tauzin@epfl.ch>
# License: Apache 2.0

import giotto as go
import giotto.time_series as ts
import giotto.diagram as diag
import giotto.compose as cp
import giotto.homology as hl
import giotto.manifold as ma
from giotto.pipeline import Pipeline

import numpy as np
import pandas as pd
import datetime as dt
import sklearn as sk
import pickle as pkl
import argparse

import sklearn.utils as skutils
from sklearn.model_selection import TimeSeriesSplit, KFold
import sklearn.preprocessing as skprep
from sklearn.ensemble import RandomForestClassifier


def get_data(n_train, n_test):
    data = np.random.rand(n_train + n_test, 1)
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


def main(n_jobs):
    n_train, n_test = 400, 200
    data = get_data(n_train, n_test)
    X_train, y_train, X_test, y_test = split_train_test(data, n_train, n_test)

    steps = [
        ('stationarizing', ts.Stationarizer(operation='return')),
        ('embedding', ts.TakensEmbedder(dimension=5, time_delay=5)),
        ('window', ts.SlidingWindow(width=5, stride=1)),
        ('diagram', hl.VietorisRipsPersistence(homology_dimensions=[0, 1], n_jobs=n_jobs)),
        ('distance', diag.DiagramDistance(metric='bottleneck', metric_params={'delta':0.1}, order = np.inf, n_jobs=n_jobs)),
        ('physical', ma.StatefulMDS(n_components=3, n_jobs=n_jobs)),
        ('kinematics', ma.Kinematics(orders=[0, 1, 2])),
        ('scaling', skprep.MinMaxScaler(copy=True)),
        ('aggregator', cp.FeatureAggregator(is_keras=False)),
        ('classification', RandomForestClassifier())
    ]

    # Running the full pipeline
    pipeline = Pipeline(steps)
    pipeline.fit(X_train, y_train).transform(X_train)

    # Running the pipeline step by step
    X_temp = X_train
    for _, transformer in steps:
        X_temp = transformer.fit(X_temp, y_train).transform(X_temp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script for typical use of the giotto library")
    parser.add_argument('-n_jobs', help="Number of processes", type=int, required=True)
    args = vars(parser.parse_args())

    main(**args)
