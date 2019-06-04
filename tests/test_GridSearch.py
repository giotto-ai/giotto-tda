import topological_learning as tl
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

# from dask_ml.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV

authorizedDataSize = { 'small': 0.2, 'median': 0.5, 'large': 1.0 }
authorizedGridSize = ['small', 'median', 'large']

# If I don't do this, the GPU is automatically used and gets out of memory
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

def init_keras_session(n_cpus_k, n_gpus_k):
    config = tf.ConfigProto( device_count = {'CPU': n_cpus_k, 'GPU': n_gpus_k} )

    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

def get_data(input_file):
    data = pd.read_parquet(input_file)

    data.time = pd.to_datetime(data.time, utc=True)
    data.time = data.time.dt.tz_convert(tz='America/New_York')
    data.time = data.time.dt.tz_localize(None)
    data.set_index('time', drop=True, inplace=True)
    data.columns = range(1)

    samplingTimeList = [ dt.time(i, 30, 0) for i in range(9, 20, 1) ] + [ dt.time(i, 0, 0) for i in range(9, 20, 1) ]
    sampling = tl.Sampler(transformationType='return', removeWeekends=True, samplingType = 'fixed', samplingTimeList=samplingTimeList)
    data = sampling.fit(data).transform(data)
    return data[0]


def split_train_test(data, size_data):
    numberTrain = int(authorizedDataSize[size_data] * data.shape[0] * 3 // 4)
    X_train = data[:numberTrain]
    y_train = np.empty((X_train.shape[0], 1))

    numberTest = numberTrain // 3
    X_test = data[numberTrain:numberTrain+numberTest]
    y_test = np.empty((X_test.shape[0], 1))
    return X_train, y_train, X_test, y_test

def make_pipeline():
    steps = [
        ('embedding', tl.TakensEmbedder()),
        ('labelling', tl.Labeller(labellingType='variation', function = np.std, percentiles=[80,90,95])),
        ('diagram', tl.VietorisRipsDiagram()),
        ('distance', tl.DiagramDistance()),
        ('physical', tl.MDS()),
        ('derivatives', tl.Derivatives()),
        ('scaling', tl.ScalerWrapper(copy=True)),
        ('formulation', tl.FormulationTransformer()),
        ('classification', tl.KerasClassifierWrapper())
    ]
    return Pipeline(steps)

def get_param_grid(size_grid):
    embedding_param = {}
    labelling_param = {}
    distance_param = {}
    diagram_param = {}
    physical_param = {}
    derivatives_param = {}
    scaling_param = {}
    formulation_param = {}
    classification_param = {}

    embedding_param['outerWindowDuration'] = [ 200, 400 ]
    embedding_param['outerWindowStride'] = [ 5, 10 ]
    embedding_param['innerWindowDuration'] = [ 10, 40 ]
    embedding_param['innerWindowStride'] = [ 1 ]

    labelling_param['deltaT'] = [ 5, 10 ]

    diagram_param['homologyDimensions'] = [ [ 0, 1 ], [0, 1 , 2] ]


    classification_param['loss'] = [ 'sparse_categorical_crossentropy' ]
    classification_param['metrics'] = [ ['sparse_categorical_accuracy'] ]

    distance_param['metric_kwargs'] = [ {'metric':'bottleneck'} ]

    physical_param['n_components'] = [ 10]

    derivatives_param['orders'] = [ [0, 1, 2] ]

    formulation_param['numberStepsInPast'] = [ 20 ]
    formulation_param['stepInFuture'] = [ 1 ]

    classification_param['modelSteps_kwargs'] = [
        [
            {'layerClass': klayers.normalization.BatchNormalization},
            {'layerClass': layerClass, 'units': units, 'activation': 'tanh'},
            {'layerClass': klayers.Dense}
        ] for layerClass in [klayers.LSTM] for units in [4] ]

    classification_param['optimizer_kwargs'] = [ {'optimizerClass': optimizerClass, 'lr': lr} for optimizerClass in [ koptimizers.RMSprop ]
                                             for lr in [0.5] ]
    classification_param['batch_size'] =  [ 200 ]
    classification_param['epochs'] =  [ 5000 ]

    if size_grid in ['median', 'large']:
        distance_param['metric_kwargs'] += [ {'metric':'landscape', 'n_layers':1, 'n_samples':1000, 'order':2},
                                             {'metric':'betti', 'n_samples':1000},
                                             {'metric':'wasserstein', 'order':2} ]

        physical_param['n_components'] += [ 15 ]

        formulation_param['numberStepsInPast'] += [ 50 ]

    if size_grid == 'large':
        classification_param['modelSteps_kwargs'] += [
            [
                {'layerClass': klayers.normalization.BatchNormalization},
                {'layerClass': klayers.Dropout, 'rate': rateInput},
                {'layerClass': layerClass, 'units': units, 'activation': 'tanh'},
                {'layerClass': klayers.Dropout, 'rate': rateLSTM},
                {'layerClass': klayers.Dense}
            ] for layerClass in [klayers.LSTM] for units in [4, 16] for rateInput in [0., 0.2] for rateLSTM in [0., 0.2]] +\
            [ [
                {'layerClass': klayers.normalization.BatchNormalization},
                {'layerClass': klayers.Dropout, 'rate': rateInput},
                {'layerClass': layerClass, 'units': units, 'activation': 'tanh', 'return_sequences':True},
                {'layerClass': klayers.Dropout, 'rate': rateLSTM},
                {'layerClass': layerClass, 'units': units, 'activation': 'tanh'},
                {'layerClass': klayers.Dropout, 'rate': rateLSTM},
                {'layerClass': klayers.Dense}
            ] for layerClass in [klayers.LSTM] for units in [4, 8] for rateInput in [0., 0.2] for rateLSTM in [0., 0.2]]

        classification_param['optimizer_kwargs'] += [ {'optimizerClass': optimizerClass, 'lr': lr}
                                             for optimizerClass in [ koptimizers.RMSprop, koptimizers.Adam ]
                                             for lr in [0.001, 0.01] ]
        classification_param['batch_size'] +=  [ 500 ]
        classification_param['epochs'] += [ 8000 ]

    embedding_param_grid = {'embedding__' + k: v for k, v in embedding_param.items()}
    labelling_param_grid = {'labelling__' + k: v for k, v in labelling_param.items()}
    diagram_param_grid = {'diagram__' + k: v for k, v in diagram_param.items()}
    distance_param_grid = {'distance__' + k: v for k, v in distance_param.items()}
    physical_param_grid = {'physical__' + k: v for k, v in physical_param.items()}
    derivatives_param_grid = {'derivatives__' + k: v for k, v in derivatives_param.items()}
    scaling_param_grid = {'scaling__' + k: v for k, v in scaling_param.items()}
    formulation_param_grid = {'formulation__' + k: v for k, v in formulation_param.items()}
    classification_param_grid = {'classification__' + k: v for k, v in classification_param.items()}

    param_grid = {**embedding_param_grid, **labelling_param_grid, **diagram_param_grid, **distance_param_grid,  **physical_param_grid,
                  **derivatives_param_grid, **scaling_param_grid, **formulation_param_grid, **classification_param_grid}

    return param_grid

def run_grid_search(estimator, param_grid, X_train, y_train, number_splits, n_jobs):
    cv = TimeSeriesSplit(n_splits=number_splits)
    grid = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv, n_jobs=n_jobs, verbose=0)
    grid_result = grid.fit(X_train, y_train)

    return grid_result

def main(input_file, size_data, size_grid, number_splits):

    data = get_data(input_file)
    X_train, y_train, X_test, y_test = split_train_test(data, size_data)
    pipeline = make_pipeline()

    param_grid = get_param_grid(size_grid)

    grid_result = run_grid_search(pipeline, param_grid, X_train, y_train, number_splits, n_jobs=-1)

    y_predict_train, y_true_train = grid_result.predict(X_train)
    y_predict_test, y_true_test = grid_result.predict(X_test)

    # Dumping artifacts
    pkl.dump(grid_result, open('grid_result.pkl', 'wb'))
    pkl.dump(y_predict_train, open('y_predict_train.pkl', 'wb'))
    pkl.dump(y_predict_test, open('y_predict_test.pkl', 'wb'))
    pkl.dump(y_true_train, open('y_true_train.pkl', 'wb'))
    pkl.dump(y_true_test, open('y_true_test.pkl', 'wb'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script for typical use of the Topological Learning library")
    parser.add_argument('-i', '--input_file', help="Path to the input file containing the time series to process.", required=True)
    parser.add_argument('-d', '--size_data', help="Size of the data to process. Possible values are 'small', 'median', and 'large'.", required=True)
    parser.add_argument('-g', '--size_grid', help="Size of the parameter grid to process. Possible values are 'small', 'median', and 'large'.", required=True)
    parser.add_argument('-s', '--number_splits', help="Number of splits run in the cross validation step.", type=int, default=2)
    parser.add_argument('-n_cpus_k', '--number_cpus_keras', help="Number of CPUs used to initialize keras session. Warning: This might lead to an explosion of thread allocations.", type=int, required=True)
    parser.add_argument('-n_gpus_k', '--number_gpus_keras', help="Number of GPUs used to initialize keras session. Warning: In the case of a grid search, this has lead to crashes as GPU memory got full.", type=int, default=0)
    args = vars(parser.parse_args())

    if args['size_data'] not in authorizedDataSize.keys():
        raise ValueError("Possible values for size_data are 'small', 'median', and 'large'")
    if args['size_grid'] not in authorizedGridSize:
        raise ValueError("Possible values for grid_data are 'small', 'median', and 'large'")

    init_keras_session(args.pop('number_cpus_keras'), args.pop('number_gpus_keras'))
    main(**args)
