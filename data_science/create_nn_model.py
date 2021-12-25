import numpy as np
from time import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from numpy import inf
from sklearn.model_selection import GridSearchCV
import pickle
from src import algorithms


def train_mlp(df, features):

    start = time()
    traduction = {
        'W': 1,
        'L': 0
        }

    df['W/L_Home'] = df['W/L_Home'].map(traduction)
    X = df[features]

    assert not(X.isnull().values.any()), 'Nan Values in data'

    case = np.isinf(X).values.sum()
    assert case is not None, 'Inf Values in data'

    X = np.array(X)
    # X[X == +inf] = 0



    y = np.array(df['W/L_Home'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_norm = scaler.transform(X_train)
    X_test_norm = scaler.transform(X_test)

    params = {
        'hidden_layer_sizes': [(32, 32), (64, 64), (128, 128), (256, 256)],
        'activation': ['logistic', 'tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'batch_size': [64],
        'learning_rate_init': [0.001]
            }

    mlp = MLPClassifier(tol=1e-6, max_iter=1000)
    grid = GridSearchCV(mlp, params, scoring='roc_auc',
                        n_jobs=-1, cv=10, refit=True, verbose=1)
    grid.fit(X_train_norm, y_train)
    mlp = grid.best_estimator_

    print(grid.best_score_)

    y_hat = mlp.predict(X_test_norm)
    y_hat = y_hat.reshape(-1, 1)

    f1 = f1_score(y_test, y_hat)
    print(f'f1 score: {f1}')

    default_dir = './ml_models'
    algorithms.check_dir(default_dir)

    with open('./ml_models/MLP_train_model.pickle', mode='wb') as fobj:
        pickle.dump(mlp, fobj)

    with open('./ml_models/x_train_scaler.pickle', mode='wb') as fobj:
        pickle.dump(scaler, fobj)

    end = time()
    print(f'Time Elapsed for training: {(end-start)/60} mins')

    return mlp
