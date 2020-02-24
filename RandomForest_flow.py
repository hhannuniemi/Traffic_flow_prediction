#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing scikit-learn random forest method to traffic flow data

@author: Hanna Hannuniemi
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from time import time


# function for random search
def random_search(model, param_dist, n_iter_search, scorer, x_train, y_train):
    rsearch = RandomizedSearchCV(model, param_distributions=param_dist,
                                       n_iter=n_iter_search, scoring=scorer, cv=5, random_state=1)
    start = time()
    result = rsearch.fit(x_train, y_train)
    print(f'RandomizedSearchCV took {(time() - start)} seconds for {n_iter_search} candidates')
    return rsearch, result

# function for using Pipeline to combine feature selection and hyperparameter tuning
def set_pipeline(sel, search, x_train, y_train, x_test):
    model  = Pipeline([('fs',sel), ('rs',search)])
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    return model, prediction

# function for plotting feature importances for fitted model
def plot_importance(importances, features):
    plt.figure(figsize=(11, 8))
    plt.style.use('seaborn-deep')
    indices = np.argsort(importances)
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='#8f63f4', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()

# evaluate baseline model
def baseline_model(x_train, y_train, x_test, y_test):
    dummy_regr = DummyRegressor(strategy="median")
    dummy_regr.fit(x_train, y_train)
    dummy_regr.predict(x_test)
    r2_score_dummy = dummy_regr.score(x_test, y_test)
    print(f'Baseline model\'s R2 score: {r2_score_dummy}')

# plot actual and predicted values to same figure
def plot_prediction_actual(y_test, prediction):
    plt.figure(figsize=(11, 8))
    plt.style.use('seaborn-deep')
    plt.plot(np.arange(len(y_test)), y_test, 'bo', label = 'actual')
    plt.plot(np.arange(len(prediction)), prediction, 'ro', label = 'prediction')
    plt.legend()
    plt.xlabel('index') 
    plt.ylabel('Traffic flow count') 
    plt.title('Predicted and actual values')
    plt.show()
    
# scatter plot of predicted and actual values    
def plot_scatter(y_test, prediction, r2):
    plt.figure(figsize=(11, 8))
    plt.style.use('seaborn-deep')
    plt.scatter(y_test, prediction)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k', lw=4)
    plt.text(y_test.max()*0.5, y_test.max()*0.8, f'R2 = {r2}', fontsize=14)
    plt.xlabel('actual') 
    plt.ylabel('prediction') 
    plt.title('Predicted vs. actual values')
    plt.show()

# save model
def save_model(model, filename):
    joblib.dump(model, filename)

if __name__ == "__main__":  
    # load data from file
    data_dir = '/home/flow'
    print(f'Data loaded from {data_dir}')
    df = pd.read_csv(os.path.join(data_dir, 'Data.csv'), delimiter=';')
    
    # check data and statistics
    print(df.columns)
    print(df.describe())
    
    # extract names of features and assign features and targets
    feature_list = list(df.drop(['FLOW_COUNT'], axis=1).columns)
    y = df['FLOW_COUNT'].values
    x = df.loc[:, df.columns != 'FLOW_COUNT'].values
    df2 = df.drop(['FLOW_COUNT'], axis=1)
    
    # split data to train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, shuffle=True)
    print(f'Training data, x: {x_train.shape}, y: {y_train.shape}')
    print(f'Test data, x: {x_test.shape}, y: {y_test.shape}')
    
    # run randomized search for finding best hyperparameters for random forest model
    model = RandomForestRegressor()
    param_dist={
               'max_features': ['sqrt'] + list(np.arange(6,10,1)),
               'min_samples_leaf': list(np.arange(1, 20, 4)),
               'max_depth': range(10,20),
               'n_estimators': list(np.arange(100, 800, 150)),
               'criterion': ('mse', 'mae')
           }
    n_iter_search = 20
    scorer = 'neg_mean_squared_error'
    

    # Build a pipeline with feature selection and random search
    start = time()
    sel = SelectFromModel(RandomForestRegressor(), threshold='median')
    rsearch = RandomizedSearchCV(model, param_distributions=param_dist,
                                            n_iter=n_iter_search, scoring=scorer, cv=5, random_state=1, refit=True)
    
    # Fit the pipeline object and predict target
    rfr_pipe  = Pipeline([('fs', sel), ('rs', rsearch)])
    rfr_pipe.fit(x_train, y_train)
    print(f'Feature selection and random search for the RF regressor took {(time() - start)} seconds.')
    prediction = rfr_pipe.predict(x_test)

    # Get best estimator and selected features
    feat = rfr_pipe.named_steps.fs.get_support()
    selected_features = df2.columns[feat]
    best_params = rfr_pipe.named_steps.rs.best_params_
    importances = rfr_pipe.named_steps['rs'].best_estimator_.feature_importances_
    print(f'Selected features: {selected_features}')
    print(f'Best parameters: {best_params}')
    

    # Evaluate model performance against actual test targets
    r2 = np.round(r2_score(y_test, prediction), 4)
    mae = mean_absolute_error(y_test, prediction)
    print(f'Model\'s R2 score: {r2} and MAE: {mae}')
    
    # Evaluate model against baseline model
    baseline_model(x_train, y_train, x_test, y_test)
    
    # Plot actual and predicted values and check feature importances
    plot_scatter(y_test, prediction, r2)
    plot_importance(importances, selected_features)
  

