
# coding: utf-8

# In[1]:


import l2f_tda as tda
import numpy as np
import pandas as pd
import datetime as dt
import gudhi as gd
import sklearn as sk
import dask_ml as daml
import pickle as pkl
from sklearn_tda.hera_wasserstein import wasserstein
from l2f_tda.DiagramDistance import kernel_landscape_distance
import joblib

import matplotlib.pyplot as plt

import sklearn.utils as skutils
from sklearn.model_selection import TimeSeriesSplit, KFold
import sklearn.preprocessing as skprep
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from keras.models import Sequential
import keras.layers as klayers
import keras.optimizers as koptimizers
import keras.callbacks as kcallbacks
from keras.wrappers.scikit_learn import KerasRegressor


# In[2]:


# In[3]:


import keras
import tensorflow as tf


import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

config = tf.ConfigProto( device_count = {'CPU': 48} )
config = tf.ConfigProto( device_count = {'GPU': 1} )

sess = tf.Session(config=config)
keras.backend.set_session(sess)


# In[51]:


data = pd.read_csv('AAPL.csv', names=['time', 'price'], parse_dates=['time'])

data.time = pd.to_datetime(data.time, utc=True)
data.time = data.time.dt.tz_convert(tz='America/New_York')
data.time = data.time.dt.tz_localize(None)
data.set_index('time', drop=True, inplace=True)
data.columns = range(1)
plt.plot(data)


# In[52]:


samplingTimeList = [ dt.time(i, 30, 0) for i in range(9, 20, 1) ] + [ dt.time(i, 0, 0) for i in range(9, 20, 1) ]
sampling = tda.Sampler(transformationType='return', removeWeekends=True, samplingType = 'fixed', samplingTimeList=samplingTimeList)
data = sampling.fit(data).transform(data)
plt.plot(data[0])


# In[53]:


numberTrain = data[0].shape[0] * 3 // 4
X_train = data[0][:numberTrain]
y_train = np.empty((X_train.shape[0], 1))
X_train.head(10)


# In[54]:


numberTest = data[0].shape[0] // 4
X_test = data[0][numberTrain:numberTrain+numberTest]
y_test = np.empty((X_test.shape[0], 1))
X_test.head()


# In[55]:


steps = [
#     ('sampling', tda.Sampler(transformationType='return', removeWeekends=True, samplingType = 'fixed', samplingTimeList=samplingTimeList)),
    ('embedding', tda.TakensEmbedder()),
    ('labelling', tda.Labeller(labellingType='variation', function = np.std)),
    ('diagram', tda.VietorisRipsDiagram()),
    ('distance', tda.DiagramDistance()),
    ('physical', tda.MDS()),
    ('derivatives', tda.Derivatives()),
    ('scaling', tda.ScalerWrapper(copy=True)),
    ('formulation', tda.FormulationTransformer()),
    ('regression', tda.KerasRegressorWrapper())
]

pipeline_transform = Pipeline(steps[:-1])
pipeline_estimate = Pipeline([steps[-1]])

pipeline = Pipeline(steps)


# In[56]:


pipeline_transform.get_params()
pipeline_estimate.get_params()
pipeline.get_params()


# In[57]:


# Sampling
sampling_param = {}
sampling_param_grid = {'sampling__' + k: v for k, v in sampling_param.items()}

# Embedding
embedding_param = {}
embedding_param['outerWindowDuration'] = [ 100, 200 ]
embedding_param['outerWindowStride'] = [ 10, 20 ]
embedding_param['innerWindowDuration'] = [ 10, 40 ]
embedding_param['innerWindowStride'] = [ 1 ]
embedding_param_grid = {'embedding__' + k: v for k, v in embedding_param.items()}

# Labelling
labelling_param = {}
labelling_param['deltaT'] = [ 10 ]
labelling_param_grid = {'labelling__' + k: v for k, v in labelling_param.items()}

# Diagram
diagram_param = {}
diagram_param['homologyDimensions'] = [ [ 0, 1 ] ]
diagram_param_grid = {'diagram__' + k: v for k, v in diagram_param.items()}

# Distance
distance_param = {}
distance_param['metric'] = [ 'landscape', 'bottleneck', 'wasserstein' ]
distance_param_grid = {'distance__' + k: v for k, v in distance_param.items()}

# Physical
physical_param = {}
physical_param['n_components'] = [ 10 ]
physical_param_grid = {'physical__' + k: v for k, v in physical_param.items()}

# Derivatives
derivatives_param = {}
derivatives_param['orders'] = [ [0, 1, 2] ]
derivatives_param_grid = {'derivatives__' + k: v for k, v in derivatives_param.items()}

# Scaling
scaling_param = {}
# scaling_param['scaler'] = [ skprep.MinMaxScaler, skprep.StandardScaler ]
scaling_param_grid = {'scaling__' + k: v for k, v in scaling_param.items()}

# Formulation
formulation_param = {}
formulation_param['numberStepsInPast'] = [ 20 ]
formulation_param['stepInFuture'] = [ 1 ]
formulation_param_grid = {'formulation__' + k: v for k, v in formulation_param.items()}

# Regression
regression_param = {}
regression_param['numberFeatures'] =  [ physical_param['n_components'][0] *  len(derivatives_param['orders'][0]) ]
regression_param['numberStepsInPast'] =  formulation_param['numberStepsInPast']
regression_param['modelSteps'] = [
    [
        {'layerClass': klayers.normalization.BatchNormalization},
        {'layerClass': klayers.Dropout, 'rate': rateInput},
        {'layerClass': layerClass, 'units': units, 'activation': 'tanh'},
        {'layerClass': klayers.Dropout, 'rate': rateLSTM},
        {'layerClass': klayers.Dense, 'units': 1, 'use_bias': False}
] for layerClass in [klayers.LSTM] for units in [4, 16] for rateInput in [0, 0.1, 0.2] for rateLSTM in [0, 0.1, 0.2]] #+\
# [ [
#         {'layerClass': klayers.normalization.BatchNormalization},
#         {'layerClass': klayers.Dropout, 'rate': rateInput},
#         {'layerClass': layerClass, 'units': units, 'activation': 'tanh'},
#         {'layerClass': klayers.Dropout, 'rate': rateLSTM1},
#         {'layerClass': layerClass, 'units': units, 'activation': 'tanh'},
#         {'layerClass': klayers.Dense, 'units': 1}
# ] for layerClass in [klayers.LSTM] for units in [2, 8] for rateInput in [0.1, 0.2] for rateLSTM1 in [0.1, 0.2] ]

regression_param['optimizerClass'] = [ koptimizers.RMSprop ] #, koptimizers.Adam ]
regression_param['optimizer_kwargs'] = [ {'lr': lr}
                                         for lr in [0.001, 0.01] ]
# regression_param['callbacks'] = [ [kcallbacks.ModelCheckpoint('./model.sk', monitor='loss', save_best_only=True)] ]
regression_param['loss'] = [ 'mean_squared_error' ]
regression_param['batch_size'] =  [ 100, 500 ]
regression_param['epochs'] =  [ 5000, 10000 ]
regression_param_grid = {'regression__' + k: v for k, v in regression_param.items()}

param_grid_transform = {**sampling_param_grid, **embedding_param_grid, **labelling_param_grid, **diagram_param_grid, **distance_param_grid,
              **physical_param_grid, **derivatives_param_grid, **scaling_param_grid, **formulation_param_grid}
param_grid_estimate = {**regression_param_grid}
param_grid = {**param_grid_transform, ** param_grid_estimate}
print(param_grid_transform)
print(param_grid_estimate)


# In[11]:


from dask_ml.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV

# from dask.distributed import Client
# client = Client()
# skutils.parallel_backend(backend='multiprocessing')


# In[ ]:


cv = TimeSeriesSplit(n_splits=3)
gridSearch = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, n_jobs=-1, pre_dispatch='2*n_jobs', verbose=100, error_score='raise') #iid=False ???\ngrid_result_estimate = grid_estimate.fit(X_train, X_train)")
bestPipeline = gridSearch.fit(X_train, X_train)
joblib.dump(gridSearch.best_estimator_, 'best_pipeline.pkl', compress = 1)

# In[ ]:


# summarize results
print("Best: %f using %s" % (bestPipeline.best_score_, bestPipeline.best_params_))
means = bestPipeline.cv_results_['mean_test_score']
stds = bestPipeline.cv_results_['std_test_score']
params = bestPipeline.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[39]:


y_predict_train, y_true_train = pipeline_estimate.predict(X_train)

# In[40]:

figure = plt.figure(figsize=(20,10))
plt.plot(y_predict_train, marker='x')
plt.plot(y_true_train)
plt.savefig('train_prediction.png')

# In[41]:


X_test_transformed = pkl.load(open('XAAPL_test_transformed.pkl', 'rb'))
y_predict_test, y_true_test = pipeline_estimate.predict(X_test)


# In[42]:


figure = plt.figure(figsize=(20,10))
plt.plot(y_predict_test, marker='x')
plt.plot(y_true_test)
plt.savefig('test_prediction.png')
