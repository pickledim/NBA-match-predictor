#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 18:40:15 2021

@author: dimitrisglenis
"""

import numpy as np
from time import time
from os import system
from src import Data_analysis_tools as Tools, algorithms
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV


def hp_tuning(df, features, XGB=True):
    start = time()
    traduction = {
        'W': 1,
        'L': 0
    }

    df['W/L_Home'] = df['W/L_Home'].map(traduction)


    grid = False

    X = df[features]
    y = df['W/L_Home']

    assert not(X.isnull().values.any()), 'Nan Values in data'

    case = np.isinf(X).values.sum()
    assert case is not None, 'Inf Values in data'

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)

    if XGB:
        # =============================================================================
        # xgboost
        # =============================================================================

        random_grid = {
            'max_depth': [1, 2, 3, 4, 5, 10],
            'learning_rate': [0.2, 0.1, 0.01, 0.05],
            'gamma': [0, 0.25, 0.5, 1, 2, 3, 5, 10],
            'reg_lambda': [0, 0.5, 1, 2, 5, 6, 10],
            'scale_pos_weight': [1, 3, 5, 10]

        }

        param_grid = {
            'max_depth': [1, 2, 3, 4, 5, 10],
            'learning_rate': [0.2, 0.1, 0.08, 0.15],
            'gamma': [0.8, 1, 1.25, 1.5],
            'reg_lambda': [4, 5, 5.5, 6]
        }
        model = xgb.XGBClassifier(objective='binary:logistic')

        save_as = 'params_from_ht_rs_xgb_test'
    else:
        # =============================================================================
        # Random Forest
        # =============================================================================

        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        # max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        print(random_grid)

        param_grid = {
            'bootstrap': [True],
            'max_depth': [10, 20, 30, 40, 100],
            'max_features': ['sqrt'],
            'min_samples_leaf': [2, 4, 6],
            'min_samples_split': [3, 5, 7],
            'n_estimators': [100, 900, 1000, 1100]
        }
        model = RandomForestClassifier()
        save_as = 'rf_params'

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune

    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    if grid:
        Grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=2, n_jobs=-1)
    else:
        Grid = RandomizedSearchCV(estimator=model, param_distributions=random_grid,
                                  n_iter=100, cv=3, random_state=42, verbose=2, n_jobs=-1)
    Grid.fit(X_train, y_train)

    params = Grid.best_params_


    save_as = './ml_models/' + save_as
    algorithms.check_dir(save_as)
    Tools.save_pickle(params, save_as)
    print(f'Train Accuracy - : {Grid.score(X_train, y_train):.3f}')
    print(f'Test Accuracy - : {Grid.score(X_test, y_test):.3f}')

    end = time()
    et = end - start
    if et > 60:
        print(f'Elapsed time {et / 60} mins')
    else:
        print(f'Elapsed time {et} sec')
    system('say Dude the program has finished ')

    return params
