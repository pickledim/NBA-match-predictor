import pandas as pd
from data_science import create_XGB_model, create_nn_model, train_nn_model, hyperparameter_tuning as ht
from src import Data_analysis_tools as Tools

# =============================================================================
# inputs
# =============================================================================
features = Tools.load_pickle('./ml_models/features_for_prediction_new')

data = pd.read_csv('../validation_data_first_half_2015_2021.csv', index_col=0)

# mlp model
TRAIN = True
if TRAIN:
    trained_mlp = create_nn_model.train_mlp(data, features)
else:
    trained_mlp = Tools.load_pickle('./ml_models/MLP_train_model')

params = trained_mlp.get_params()

mlp_name = 'MLP_model'
scaler_name = 'x_scaler'

train_nn_model.mlp_model(data, features, mlp_name, params, scaler_name)

# xgb model
model_name = 'XGB'

hpt = True
if hpt:
    params = ht.hp_tuning(data, features)
else:
    params = Tools.load_pickle('./ml_models/params_from_ht_rs_xgb')

create_XGB_model.xgb_model(features, params, data, model_name)