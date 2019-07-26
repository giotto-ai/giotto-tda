import topological_learning as tl
import topological_learning.preprocessing as prep
import topological_learning.diagram as diag
import topological_learning.homology as hl
import topological_learning.neural_network as nn
import topological_learning.manifold as ma
import topological_learning.compose as cp

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

from keras.models import Sequential
import keras.layers as klayers
import keras.optimizers as koptimizers

import keras
import tensorflow as tf

# from dask_ml.model_selection import GridSearchCV

# If I don't do this, the GPU is automatically used and gets out of memory
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

def init_keras_session():
    sess = tf.Session()
    keras.backend.set_session(sess)

def get_data(n_train, n_test):
    data = np.random.rand(n_train+n_test, 1)
    return data

def split_train_test(data, n_train, n_test):
    labeller = prep.Labeller(labelling_kwargs={'type': 'derivation', 'delta_t':3}, window_size=5, percentiles=[80], n_steps_future=1)
    X_train = data[:n_train]
    y_train = X_train
    labeller.fit(y_train)
    y_train = labeller.transform(y_train)
    X_train = labeller.cut(X_train)

    X_test = data[n_train:n_train+n_test]
    y_test = X_test
    y_test = labeller.transform(y_test)
    X_test = labeller.cut(X_test)

    return X_train, y_train, X_test, y_test

def main(n_jobs):
    n_train, n_test = 400, 200
    data = get_data(n_train, n_test)
    X_train, y_train, X_test, y_test = split_train_test(data, n_train, n_test)

    steps = [
        ('stationarizing', prep.Stationarizer(stationarization_type='return')),
        ('embedding', prep.TakensEmbedder(outer_window_duration=20)),
        ('diagram', hl.VietorisRipsPersistence(homology_dimensions=[ 0, 1 ], n_jobs=n_jobs)),
        ('distance', diag.DiagramDistance(metric='bottleneck', metric_params={'order': np.inf}, n_jobs=n_jobs)),
        ('physical', ma.StatefulMDS(n_components=3, n_jobs=n_jobs)),
        ('kinematics', ma.Kinematics(orders=[0, 1, 2])),
        ('scaling', skprep.MinMaxScaler(copy=True)),
        ('aggregator', cp.FeatureAggregator(n_steps_in_past=2, is_keras=True))
    ]

    # Running the full pipeline
    pipeline = Pipeline(steps)
    pipeline.fit(X_train, y_train).transform(X_train)

    # Running the pipeline step by step
    X_temp = X_train
    for _, transformer in steps:
        temp = transformer.fit(X_temp, y_train).transform(X_temp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script for typical use of the Topological Learning library")
    parser.add_argument('-n_jobs', help="Number of processes", type=int, required=True)
    args = vars(parser.parse_args())

    init_keras_session()
    main(**args)
