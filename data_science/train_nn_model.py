import numpy as np
from src import Data_analysis_tools as Tools, algorithms
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
import os


def mlp_model(df, features, mlp_name, params, scaler_name):

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

    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    y = np.array(df['W/L_Home'])

    mlp = MLPClassifier(tol=1e-6, max_iter=1000)
    mlp.set_params(**params)
    print('Training ...')
    mlp.fit(X, y)
    print('Done')

    default_dir = './ml_models'
    algorithms.check_dir(default_dir)

    mlp_dir = os.path.join(default_dir, mlp_name)
    scaler_dir = os.path.join(default_dir, scaler_name)

    Tools.save_pickle(mlp, mlp_dir)
    Tools.save_pickle(scaler, scaler_dir)
