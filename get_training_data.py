from src import algorithms
import numpy as np

data = algorithms.get_data_from_2015()
data = algorithms.pre_process_cols(data)
data = algorithms.hollinger_formula(data)
data = algorithms.concat_home_away_stats(data)
data = algorithms.get_dummies(data)
data = algorithms.feature_eng(data)
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data = data.fillna(0)
data.to_csv('validation_data_first_half_2015_2021.csv')
