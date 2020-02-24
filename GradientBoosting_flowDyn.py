#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing scikit-learn gradient boosting method to traffic flow data (dynamic)

@author: hannunie
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import NuSVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from time import time
from TraffficFlow_RF import random_search, set_pipeline, baseline_model, plot_scatter, plot_importance, plot_prediction_actual

# load data from file
data_dir = '/home/flow_dyn'
print(f'Data loaded from {data_dir}')
df = pd.read_csv(os.path.join(data_dir, 'Data.csv'), delimiter=';')

# Convert hour (string) to number
df['hourINT'] = df['HOUR'].str.extract('(\d+)').astype(int)
df = df.drop(['HOUR'], axis=1)

# extract names of features and assign features and targets
feature_list = list(df.drop(['COUNT_FRAC'], axis=1).columns)
df2 = df.drop(['COUNT_FRAC'], axis=1)
y = df['COUNT_FRAC'].values
x = df.loc[:, df.columns != 'COUNT_FRAC'].values
print(feature_list)

# split data to train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3, shuffle=True)
print(f'Training data, x: {x_train.shape}, y: {y_train.shape}')
print(f'Test data, x: {x_test.shape}, y: {y_test.shape}')

# Fit Gradient Boosting regressor
params = {'n_estimators': 1000, 'max_depth': 8, 'min_samples_leaf': 10, 'max_features': 20,
          'learning_rate': 0.1, 'loss': 'ls', 'subsample': 1.0}
start = time()
gbr = GradientBoostingRegressor(**params)
gbr.fit(x_train, y_train)
print(f'Fitting the GB regressor took {(time() - start)} seconds.')

# Predict target and check feature importances
prediction = gbr.predict(x_test)
importances = gbr.feature_importances_

# Check performance metrics
r2 = np.round(r2_score(y_test, prediction), 4)
mae = mean_absolute_error(y_test, prediction)
print(f'Model\'s R2 score: {r2} and MAE: {mae}')

# Plot actual and predicted values and check feature importances
plot_scatter(y_test, prediction, r2)
plot_importance(importances, feature_list)
plot_prediction_actual(y_test, prediction)

# Plot training and test set losses
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(gbr.staged_predict(x_test)):
    test_score[i] = gbr.loss_(y_test, y_pred)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Loss')
plt.plot(np.arange(params['n_estimators']) + 1, gbr.train_score_, 'b-',
          label='Training Set loss')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
          label='Test Set loss')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Loss')