import numpy as np
from src import Data_analysis_tools as Tools, algorithms
from time import time
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import xgboost as xgb
import os


def xgb_model(features, params, df, model_name):

    start = time()
    traduction = {
        'W': 1,
        'L': 0
        }

    df['W/L_Home'] = df['W/L_Home'].map(traduction)

    X = df[features]
    y = np.array(df['W/L_Home'])

    assert not(X.isnull().values.any()), 'Nan Values in data'

    case = np.isinf(X).values.sum()
    assert case is not None, 'Inf Values in data'

    X = np.array(X)
    X_k, X_val, y_k, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    kf = KFold(n_splits=10)
    kf.get_n_splits(X_k)

    accuracy = 0
    for train_index, test_index in kf.split(X_k):
        X_train, X_test = X_k[train_index], X_k[test_index]
        y_train, y_test = y_k[train_index], y_k[test_index]

        ml_model = xgb.XGBClassifier(objective='binary:logistic', iterations=1000)

        ml_model.set_params(**params)
        ml_model.fit(X_train,
                     y_train,
                     verbose=True,
                     early_stopping_rounds=10,
                     eval_metric='aucpr',
                     eval_set=[(X_test, y_test)])

        score = ml_model.score(X_test, y_test)
        print(f'test score = {score}')
        # y_prob = ml_model.predict_proba(X_val)

        y_pred = ml_model.predict(X_val)
        f1 = f1_score(y_val, y_pred)
        print(f'f1 val score = {f1}')
        cm = confusion_matrix(y_val, y_pred)
        print(f"Confusion Matrix: {cm}")

        accuracy += f1

    accuracy = accuracy / kf.n_splits
    print(f'Accuracy of the model: {accuracy}')

    ml_model = xgb.XGBClassifier(objective='binary:logistic', iterations=1000)
    ml_model.set_params(**params)
    ml_model.fit(X, y)
    default_dir = './ml_models'
    algorithms.check_dir(default_dir)

    model_name = os.path.join(default_dir, model_name)
    Tools.save_pickle(ml_model, model_name)
    end = time()

    print(f'Time Elapsed: {(end-start)/60} mins')
