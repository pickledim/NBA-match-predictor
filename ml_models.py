import pandas as pd
from data_science import create_XGB_model, create_nn_model, train_nn_model, hyperparameter_tuning as ht
from src import Data_analysis_tools as Tools

# =============================================================================
# inputs
# =============================================================================

features = Tools.load_pickle('./ml_models/features_for_prediction_new')

data = pd.read_csv('misc_data/validation_data_first_half_2015_2021_new.csv', index_col=0)

# mlp model
TRAIN = True
if TRAIN:
    trained_mlp = create_nn_model.train_mlp(data, features)
else:
    trained_mlp = Tools.load_pickle('./ml_models/MLP_train_model')

params = trained_mlp.get_params()

mlp_name = 'MLP_model_test25_10_2022'
scaler_name = 'x_scaler_test25_10_2022'
data = pd.read_csv('misc_data/validation_data_first_half_2015_2021_new.csv', index_col=0)
train_nn_model.mlp_model(data, features, mlp_name, params, scaler_name)

train_xgb = True
if train_xgb:
    # xgb model
    model_name = 'XGB25_10_2022'
    data = pd.read_csv('misc_data/validation_data_first_half_2015_2021_new.csv', index_col=0)
    hpt = True
    if hpt:
        params = ht.hp_tuning(data, features)
    else:
        params = Tools.load_pickle('./ml_models/params_from_ht_rs_xgb')
    data = pd.read_csv('misc_data/validation_data_first_half_2015_2021_new.csv', index_col=0)
    create_XGB_model.xgb_model(features, params, data, model_name)
