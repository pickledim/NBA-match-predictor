import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from src import algorithms, Data_analysis_tools as Tools, stats_scraper

pre_pros = False
scrap = True

url = 'https://www.nba.com/stats/teams/boxscores-traditional/?Season=2021-22&SeasonType=Regular%20Season&GameSegment' \
      '=First%20Half '

if pre_pros:
    if scrap:
        data = stats_scraper.web_scraper(url, f'../teams_boxscore_trad_2021_first_half.csv', boolean=True, boxscore=True,
                                         teams=True, live=False)
    else:
        data = pd.read_csv('../teams_boxscore_trad_2021_first_half.csv', index_col=0)

    data = algorithms.pre_process_cols(data)
    data = algorithms.hollinger_formula(data)
    data = algorithms.concat_home_away_stats(data)
    data = algorithms.get_dummies(data)
    data = algorithms.feature_eng(data)
    data = algorithms.clean_data(data)
    data.to_csv('validation_data_2021.csv')
else:
    data = pd.read_csv('../validation_data_2021.csv', index_col=0)

# =============================================================================
# predict
# =============================================================================

features = Tools.load_pickle('./ml_models/features_for_prediction_new')

xgb_model = Tools.load_pickle('./ml_models/XGB')

mlp_model = Tools.load_pickle('./ml_models/MLP_model')

scaler = Tools.load_pickle('./ml_models/x_scaler')

df = data[features]

assert not(df.isnull().values.any()), 'Nan Values in data'
case = np.isinf(df).values.sum()
assert case is not None, 'Inf Values in data'

X = np.array(df)

y_pred_xgb = xgb_model.predict(X)

X = scaler.transform(X)

y_pred_nn = mlp_model.predict(X)

mapp = {
    'W': 1,
    'L': 0
}

y_val = np.array(data['W/L_Home'].map(mapp))
y_prob_xgb = xgb_model.predict_proba(X)
y_prob_nn = mlp_model.predict_proba(X)
print(f'\nf1_score nn: {Tools.f1_eval(y_pred_nn, y_val)}\n')
cm = confusion_matrix(y_val, y_pred_nn)
print(cm)

print(f'\nf1_score xgb: {Tools.f1_eval(y_pred_xgb, y_val)}\n')
cm = confusion_matrix(y_val, y_pred_xgb)
print(cm)

# results = y_pred_nn==y_val
# (unique, counts) = np.unique(results, return_counts=True)
# errors=pd.Series(counts[0]/len(results))
# correct=pd.Series(counts[1]/len(results))

inv_map = {v: k for k, v in mapp.items()}
y_pred_series = pd.Series(y_pred_nn)
y_pred_series = y_pred_series.map(inv_map)

y_pred_series2 = pd.Series(y_pred_xgb)
y_pred_series2 = y_pred_series2.map(inv_map)

data['xgb_pred'] = y_pred_series2
data['xgb_prob'] = y_prob_xgb.max(axis=1)
data['nn_pred'] = y_pred_series
data['nn_prob'] = y_prob_nn.max(axis=1)

results = data[['TEAM_Home', 'TEAM_Away', 'DATE_Home', 'xgb_pred', 'xgb_prob', 'nn_pred', 'nn_prob', 'W/L_Home']]
threshold = 0.5

mask1 = (data['nn_prob'] >= threshold) & (data['nn_prob'] >= 0.58)
results_mlp = results[mask1]
mask = results['W/L_Home'] == data['nn_pred']
results_mlp_corr = results_mlp[mask]
results_mlp_err = results_mlp[~mask]

mask2 = data['xgb_prob'] >= threshold
mask = results['W/L_Home'] == data['xgb_pred']
results_xgb = results[mask2]
results_xgb_corr = results_xgb[mask]
results_xgb_err = results_xgb[~mask]
