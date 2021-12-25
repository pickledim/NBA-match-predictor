#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 12:49:31 2020

@author: dimitrisglenis
"""
import pandas as pd
from sklearn import preprocessing
import pickle
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import xgboost as xgb


def f1_eval(y_pred, y_val):
    f1 = f1_score(y_val, y_pred)
    return f1


def regularizer(vector):
    # Create a minimum and maximum processor object
    min_max_scaler = preprocessing.StandardScaler()

    # Create an object to transform the data to fit minmax processor
    x_scaled = min_max_scaler.fit_transform(vector)

    return x_scaled


def stoiximan_fp_calculator(df):
    # TO MAP
    df['Stoiximan FP'] = 3.5 * df['3PM'] + 2 * df['2PM'] + 1 * df['FTM'] + 1.2 * df['REB'] + 1.5 * df['AST'] + 2 * df[
        'STL'] + 2 * df['BLK'] \
                         - 0.5 * df['TO'] + 1.5 * df['DD2'] + 3 * df['TD3']
    return df


def average_by_game(df):
    to_average = ['MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TO', 'FPTS', '2PM', '3PM', 'FTM', 'DD2', 'TD3']

    for attribute in to_average:
        df[attribute] = df[attribute] / df['GMS']
    df = df.fillna(0)

    return df


def mapping_team():
    teams_abbr = pd.read_csv('teams_abbreviation.csv', sep=';')
    mapping_team_names = {}
    for i, row in teams_abbr.iterrows():
        mapping_team_names[row['TEAM']] = row['ABBREVIATION']

    return mapping_team_names


def save_pickle(obj, name):
    with open(f'{name}.pickle', 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(name):
    with open(f'{name}.pickle', 'rb') as handle:
        obj = pickle.load(handle)
    return obj


def ridge_regression(X, y, lamda, n_kfold):
    # =============================================================================
    # Ridge Regression
    # =============================================================================
    # X=np.array(df[features])
    # y=np.array(df[obj_f]) #has a normal distribution
    print('\nRidge Regression\n')
    kf = KFold(n_splits=n_kfold)
    kf.get_n_splits(X)
    sc = MinMaxScaler()
    rmse = 0
    mae = 0
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train = sc.fit_transform(x_train)
        X_test = sc.transform(x_test)
        clf = Ridge(alpha=0.1)
        clf.fit(X_train, y_train)
        clf.score(X_test, y_test)
        print('Score: ', clf.score(x_test, y_test))
        y_pred = clf.predict(X_test)
        rmse += (mean_squared_error(y_test, y_pred, squared=False))
        mae += (mean_absolute_error(y_test, y_pred))

    print('RMSE: ', rmse / kf.get_n_splits())
    print('MAE: ', mae / kf.get_n_splits())
    return clf, sc


def lasso_regression(X, y, lamda, n_kfold):
    # =============================================================================
    # Ridge Regression
    # =============================================================================
    # X=np.array(df[features])
    # y=np.array(df[obj_f]) #has a normal distribution
    print('\nLasso Regression\n')
    kf = KFold(n_splits=n_kfold)
    kf.get_n_splits(X)
    sc = MinMaxScaler()
    rmse = 0
    mae = 0
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train = sc.fit_transform(x_train)
        X_test = sc.transform(x_test)
        clf = Lasso(alpha=lamda)
        clf.fit(X_train, y_train)
        clf.score(X_test, y_test)
        print('Score: ', clf.score(x_test, y_test))
        y_pred = clf.predict(X_test)
        rmse += (mean_squared_error(y_test, y_pred, squared=False))
        mae += (mean_absolute_error(y_test, y_pred))
    '''
    
    Research more the way it works
    '''

    print('RMSE: ', rmse / kf.get_n_splits())
    print('MAE: ', mae / kf.get_n_splits())
    return clf, sc


def random_forest(df, features, obj_f, lamda, n_kfold):
    # =============================================================================
    # Random Forest
    # =============================================================================
    X = np.array(df[features])
    y = np.array(df[obj_f])  # has a normal distribution

    kf = KFold(n_splits=n_kfold)
    kf.get_n_splits(X)
    sc = MinMaxScaler()
    rmse = 0
    mae = 0
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train = sc.fit_transform(x_train)
        X_test = sc.transform(x_test)
        rf = RandomForestRegressor(n_estimators=lamda, n_jobs=-1, max_features='sqrt')
        rf.fit(X_train, y_train)
        rf.score(X_test, y_test)

        y_pred = rf.predict(X_test)
        rmse += (mean_squared_error(y_test, y_pred, squared=False))
        mae += (mean_absolute_error(y_test, y_pred))
    '''
    
    Research more the way it works
    '''

    print('\nRandom Forest\n')
    print('Score: ', rf.score(x_test, y_test))
    print('RMSE: ', rmse / kf.get_n_splits())
    print('MAE: ', mae / kf.get_n_splits())
    return rf


def random_forest_clsfr(df, features, obj_f, lamda, n_kfold):
    # =============================================================================
    # Random Forest
    # =============================================================================
    X = np.array(df[features])
    y = np.array(df[obj_f])  # has a normal distribution
    X, X_val, y, y_val = train_test_split(X, y, test_size=0.10, shuffle=True)
    # from imblearn.over_sampling import SMOTE
    # X_resampled, y_resampled = SMOTE().fit_resample(X, y)
    kf = KFold(n_splits=n_kfold)
    kf.get_n_splits(X)
    sc = MinMaxScaler()
    error = 0
    models = {}
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train = sc.fit_transform(x_train)
        X_test = sc.transform(x_test)
        crf = RandomForestClassifier(n_estimators=lamda, max_depth=2, random_state=0)
        # min_samples_split = 2,min_samples_leaf = 4,
        # bootstrap=True)
        crf.fit(X_train, y_train)
        crf.score(X_test, y_test)

        x_val = sc.transform(X_val)
        y_pred = crf.predict(x_val)
        cm = confusion_matrix(y_val, y_pred)
        if (y_pred == y_val).all():
            error += 1
        else:
            f1 = f1_score(y_val, y_pred)
            error += f1
        models[str(f1)] = (crf, sc, cm)
        # k=np.random.randint(2, size=math.ceil(df.shape[0]*0.1))
        # f1=f1_score(y_val, k)
        # print('random_values F1_score: ',f1)
    '''
    
    Research more the way it works
    '''
    print('\nRandom Forest\n')
    print('F1_score: ', error / kf.get_n_splits())
    # print('MAE: ',mae/kf.get_n_splits())
    return models


def svm(X, y, lamda, n_kfold):
    # =============================================================================
    # SVM
    # =============================================================================
    # X=np.array(df[features])
    # y=np.array(df[obj_f]) #has a normal distribution
    print('\nSVM\n')

    kf = KFold(n_splits=n_kfold)
    kf.get_n_splits(X)

    rmse = 0
    mae = 0
    score = 0
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        regr = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=1.0, epsilon=0.2))
        regr.fit(x_train, y_train)

        score += regr.score(x_test, y_test)
        y_pred = regr.predict(x_test)
        rmse += (mean_squared_error(y_test, y_pred, squared=False))
        mae += (mean_absolute_error(y_test, y_pred))

    print('SCORE: ', score / kf.get_n_splits())
    print('RMSE: ', rmse / kf.get_n_splits())
    print('MAE: ', mae / kf.get_n_splits())
    return regr


def svm_clsfr(df, features, obj_f, lamda, n_kfold):
    # =============================================================================
    # Random Forest
    # =============================================================================
    X = np.array(df[features])
    y = np.array(df[obj_f])  # has a normal distribution

    X, X_val, y, y_val = train_test_split(X, y, test_size=0.10, shuffle=True)
    kf = KFold(n_splits=n_kfold)
    kf.get_n_splits(X)
    # sc=MinMaxScaler()
    error = 0

    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # X_train = sc.fit_transform(x_train)
        # X_test = sc.transform (x_test)
        svm = make_pipeline(StandardScaler(), SVC(kernel='poly', gamma='auto'))
        svm.fit(X_train, y_train)
        svm.score(X_test, y_test)

        y_pred = svm.predict(X_val)
        if (y_pred == y_val).all():
            error += 1
        else:
            error += f1_score(y_val, y_pred)

    print('\nSVM\n')
    print('F1_score: ', error / kf.get_n_splits())
    # print('MAE: ',mae/kf.get_n_splits())
    return svm


def xgboost(df, features, obj_f):
    X = np.array(df[features])
    y = np.array(df[obj_f])  # has a normal distribution
    X, X_val, y, y_val = train_test_split(X, y, test_size=0.10, shuffle=True)
    # from imblearn.over_sampling import SMOTE
    # X_resampled, y_resampled = SMOTE().fit_resample(X, y)
    kf = KFold(n_splits=10)
    kf.get_n_splits(X)
    # sc=MinMaxScaler()
    error = 0
    models = {}
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train = x_train
        X_test = x_test
        clf_xgb = xgb.XGBClassifier(objective='binary:logistic', iterations=200)
        clf_xgb.fit(X_train,
                    y_train,

                    verbose=10,
                    early_stopping_rounds=10,
                    eval_metric='aucpr',
                    eval_set=[(X_test, y_test)])
        y_pred = clf_xgb.predict(X_val)
        y_prob = clf_xgb.predict_proba(X)
        # y_pred = crf.predict(x_val)
        cm = confusion_matrix(y_val, y_pred)
        if (y_pred == y_val).all():
            error += 1
        else:
            f1 = f1_score(y_val, y_pred)
            error += f1
        errors_prob = {}
        correct_prob = {}
        for i, res in enumerate(y_pred):
            if res != y_val[i]:
                errors_prob[i] = max(y_prob[i]), res
            else:
                correct_prob[i] = max(y_prob[i]), res
        models[str(f1)] = (clf_xgb, cm, errors_prob, correct_prob)

    return models
