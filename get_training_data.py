from src import algorithms

data = algorithms.get_data_from_2015()
data = algorithms.pre_process_cols(data)
data = algorithms.hollinger_formula(data)
data = algorithms.concat_home_away_stats(data)
data = algorithms.get_dummies(data)
data = algorithms.feature_eng(data)
data = algorithms.clean_data(data)

data.to_csv('validation_data_first_half_2015_2021.csv')
