import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from src import algorithms, Data_analysis_tools as Tools, stats_scraper

pre_pros = True
scrap = True

url = 'https://www.nba.com/stats/teams/boxscores-traditional/?Season=2021-22&SeasonType=Regular%20Season&GameSegment' \
      '=First%20Half '

if pre_pros:
    if scrap:
        data = stats_scraper.web_scraper(url, f'teams_boxscore_trad_2021_first_half.csv', boolean=True, boxscore=True,
                                         teams=True, live=False)
    else:
        data = pd.read_csv('teams_boxscore_trad_2021_first_half.csv', index_col=0)

    data = algorithms.pre_process_cols(data)
    data = algorithms.hollinger_formula(data)
    data = algorithms.concat_home_away_stats(data)
    data = algorithms.get_dummies(data)
    data = algorithms.feature_eng(data)
    data = algorithms.clean_data(data)
    data.to_csv('validation_data_2021.csv')
else:
    data = pd.read_csv('validation_data_2021.csv', index_col=0)

# =============================================================================
# predict
# =============================================================================

data = data.sort_values(by='DATE_Home', key=pd.to_datetime, ascending=False)

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

imap = {
        1: 'W',
        0: 'L'
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

inv_map = {v: k for k, v in mapp.items()}
y_pred_series = pd.Series(y_pred_nn)
y_pred_series = y_pred_series.map(inv_map)

y_pred_series2 = pd.Series(y_pred_xgb)
y_pred_series2 = y_pred_series2.map(inv_map)

data['xgb_pred'] = y_pred_series2
data['xgb_prob'] = y_prob_xgb.max(axis=1)
data['nn_pred'] = y_pred_series
data['nn_prob'] = y_prob_nn.max(axis=1)

kmeans_2008 = Tools.load_pickle('./ml_models/clusterer_2008')
kmeans_2012 = Tools.load_pickle('./ml_models/clusterer_2012')


results = data[['TEAM_Home', 'TEAM_Away', 'DATE_Home', 'xgb_pred', 'xgb_prob', 'nn_pred', 'nn_prob', 'W/L_Home']]

results['nn_pred'] = results['nn_pred'].map(mapp)
results['xgb_pred'] = results['xgb_pred'].map(mapp)

features = ['nn_prob']#, 'xgb_prob', 'nn_pred', 'xgb_pred']

results['clusters_2008'] = kmeans_2008.predict(results[features])
results['clusters_2012'] = kmeans_2012.predict(results[features])

results['nn_pred'] = results['nn_pred'].map(imap)
results['xgb_pred'] = results['xgb_pred'].map(imap)

results.to_csv('Results.csv')




#
# mlp_final = Tools.load_pickle('./ml_models/mlp_test')
# scaler_final = Tools.load_pickle('./ml_models/scaler_test')
# X = scaler_final.transform(np.array(results[['nn_prob', 'xgb_prob', 'nn_pred', 'xgb_pred']]))
# final_pred = mlp_final.predict(X)
# results['final_pred'] = final_pred
# # results['final_pred_prob'] = xgb_final.predict_proba(np.array(results[['nn_prob', 'xgb_prob', 'nn_pred', 'xgb_pred']]))
#
# results['nn_pred'] = results['nn_pred'].map(mapp)
# results['xgb_pred'] = results['xgb_pred'].map(mapp)
# results['final_pred'] = results['final_pred'].map(mapp)
# mask = results['nn_pred']!= results['xgb_pred']
# ambig_results = results[mask]
'''
spectrum = np.arange(0.5, .96, 1e-1)
prob_res = {}
for prob in spectrum:
    mask1 = (data['nn_prob'] >= prob) & (data['nn_prob'] <= prob + 1) & (data['nn_pred'] == 'W')
    results_mlp = results[mask1]
    mask = results['W/L_Home'] == data['nn_pred']
    results_mlp_corr = results_mlp[mask]
    results_mlp_err = results_mlp[~mask]

    mask2 = (data['xgb_prob'] >= prob) & (data['xgb_prob'] <= prob + 1) & (data['xgb_pred'] == 'W')
    mask = results['W/L_Home'] == data['xgb_pred']
    results_xgb = results[mask2]
    results_xgb_corr = results_xgb[mask]
    results_xgb_err = results_xgb[~mask]

    prob_res[prob] = {'mlp_corr': results_mlp_corr.shape[0],
                      'mlp_error': results_mlp_err.shape[0],
                      'xgb_corr': results_xgb_corr.shape[0],
                      'xgb_error': results_xgb_err.shape[0]
                      }

for keys, values in prob_res.items():
    print(f'\nAccuracy {keys} - {keys + 0.1} MLP:')
    print(values['mlp_corr']/(values['mlp_corr'] + values['mlp_error']))
'''
