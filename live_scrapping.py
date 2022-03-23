#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 14:08:31 2021

@author: dimitrisglenis
"""

import pandas as pd
import numpy as np
from src import algorithms, stats_scraper
import datetime
from stoiximan_screenshot import get_screenshot
import pandas as pd
import numpy as np
from src import Data_analysis_tools as Tools
import telegram_bot_nba as tbn
from keras.models import load_model


def name_mapping(abbr):

    name_map = {
        "BOS": "Boston Celtics",
        "BKN": "Brooklyn Nets",
        "NYK": "New York Knicks",
        "PHI": "Philadelphia 76ers",
        "TOR": "Toronto Raptors",
        "CHI": "Chicago Bulls",
        "CLE": "Cleveland Cavaliers",
        "DET": "Detroit Pistons",
        "IND": "Indiana Pacers",
        "MIL": "Milwaukee Bucks",
        "ATL": "Atlanta Hawks",
        "CHA": "Charlotte Hornets",
        "MIA": "Miami Heat",
        "ORL": "Orlando Magic",
        "WAS": "Washington Wizards",
        "DEN": "Denver Nuggets",
        "MIN": "Minnesota Timberwolves",
        "OKC": "Oklahoma City Thunder",
        "POR": "Portland Trail Blazers",
        "UTA": "Utah Jazz",
        "GSW": "Golden State Warriors",
        "LAC": "LA Clippers",
        "LAL": "Los Angeles Lakers",
        "PHX": "Phoenix Suns",
        "SAC": "Sacramento Kings",
        "DAL": "Dallas Mavericks",
        "HOU": "Houston Rockets",
        "MEM": "Memphis Grizzlies",
        "NOP": "New Orleans Pelicans",
        "SAS": "San Antonio Spurs",
        }

    return name_map[abbr]


def get_results(df, results, y_pred_mlp, y_proba_mlp, imap, send_message=False):
    min_bet = 2.5
    max_bet = 5

    past_data = pd.read_csv('clusters_accuracy.csv', index_col=0)
    coefs = past_data.groupby('clusters').head(1)
    coefs = coefs[['clusters', 'Accuracy']]
    cond = coefs['clusters'] == results['clusters'].unique()[0]
    accuracy = coefs['Accuracy'][cond].iloc[0]
    print(f'Accuracy: {accuracy}')
    if accuracy == 1.:
        print("\n ALL IN!\n")
        # get_screenshot(f'{TEAM1}vs{TEAM2}')
        send_message = True
        message1 = f"{name_mapping(df['TEAM_Home'].iloc[0])} - {name_mapping(df['TEAM_Away'].iloc[0])} "
        message2 = f"{name_mapping(df['TEAM_Away'].iloc[0])} \n"
        message3 = f"{round(100, 2)}\n"
        message4 = f"\nmetadata: \n\n{name_mapping(df['TEAM_Home'].iloc[0])} {imap[y_pred_mlp[0]]}\n"
        message5 = f"MLP Proba: {round((1 - y_proba_mlp[0]), 3)} "
        message6 = f"\nClustering Method Proba: {round(accuracy, 2)}"
        message = message1 + message2 + message3 + message4 + message5 + message6
    elif accuracy >= 0.83:
        x = [0.83, coefs['Accuracy'].max()]
        y = [min_bet, max_bet]
        bet = np.interp(accuracy, x, y)
        print(f"\n BET {round(bet,2)}e!")

        send_message = True
        message1 = f"{name_mapping(df['TEAM_Home'].iloc[0])} - {name_mapping(df['TEAM_Away'].iloc[0])} "
        message2 = f"{name_mapping(df['TEAM_Away'].iloc[0])} \n"
        message3 = f"{round(bet, 2)}\n"
        message4 = f"\nmetadata: \n\n{name_mapping(df['TEAM_Home'].iloc[0])} {imap[y_pred_mlp[0]]}\n"
        message5 = f"MLP Proba: {round((1 - y_proba_mlp[0]), 3)} "
        message6 = f"\nClustering Method Proba: {round(accuracy, 2)}"
        message = message1 + message2 + message3 + message4 + message5 + message6
    else:
        print("\nFuck it don't bet\n")
        x = [coefs['Accuracy'].min(), coefs['Accuracy'].max()]
        y = [min_bet, max_bet]
        bet = np.interp(accuracy, x, y)
        send_message = True
        message1 = f"{name_mapping(df['TEAM_Home'].iloc[0])} - {name_mapping(df['TEAM_Away'].iloc[0])} "
        message2 = f"{name_mapping(df['TEAM_Away'].iloc[0])} \n"
        message3 = f"\n{round(bet, 2)}\n"
        message4 = f"\nmetadata: \n\n{name_mapping(df['TEAM_Home'].iloc[0])} {imap[y_pred_mlp[0]]}\n"
        message5 = f"MLP Proba: {round((1 - y_proba_mlp[0]), 3)} "
        message6 = f"\nClustering Method Proba: {round(accuracy, 2)}"
        message = message1 + message2 + message3 + message4 + message5 + message6

    if send_message:
        tbn.send_message(message)
    return


def predict_match(d1, day, month, live, TEAM1, TEAM2):
    date = f'{day}_{month}'
    # date in yyyy/mm/dd/hh/mm format

    cont = True
    # =============================================================================
    # scrape
    # =============================================================================
    while cont:
        d2 = datetime.datetime.now()
        if d2 >= d1:
            if live:
                if day >= 10 and month >= 10:
                    url = f'https://www.nba.com/stats/teams/boxscores/?Season=2021-22&SeasonType=Regular%20Season&' \
                          f'DateFrom={month}%2F{day}%2F2022' # &DateTo={month}%2F{day}%2F2022

                else:
                    url = f'https://www.nba.com/stats/teams/boxscores/?Season=2021-22&SeasonType=Regular%20Season&' \
                          f'DateFrom=0{month}%2F{day}%2F2022' #&DateTo={month}%2F{day}%2F2022

                data = stats_scraper.web_scraper(url, f'teams_boxscore_trad_{date}_first_half.csv', boolean=False,
                                                  boxscore=True, teams=True, live=live)
            else:
                if day >= 10 and month >= 10:
                    url = f'https://www.nba.com/stats/teams/boxscores-traditional/?Season=2021-22&SeasonType=Regular%20' \
                          f'Season&GameSegment%27%20%5C%20%27=First%20Half&DateFrom={month}%2F{day}%2F2022&' \
                          f'& DateTo={month}%2F{day}%2F2022&GameSegment=First%20Half'


                else:
                    url = f'https://www.nba.com/stats/teams/boxscores-traditional/?Season=2021-22&SeasonType=Regular%20' \
                          f'Season&GameSegment%27%20%5C%20%27=First%20Half&DateFrom=0{month}%2F0{day}%2F2022&' \
                          f'& DateTo={month}%2F{day}%2F2022&GameSegment=First%20Half'
                data = stats_scraper.web_scraper(url, f'teams_boxscore_trad_{date}_first_half.csv', boolean=False,
                                                  boxscore=True, teams=True, live=live)

            data = pd.read_csv(f'teams_boxscore_trad_{date}_first_half.csv', index_col=0)  # Attention to the path

            data = algorithms.pre_process_cols(data)
            data = algorithms.hollinger_formula(data)
            data = algorithms.concat_home_away_stats(data)
            data = algorithms.get_dummies(data)
            data = algorithms.feature_eng(data)
            data = algorithms.clean_data(data)

            cond = (data.TEAM_Home == TEAM1) & (data.TEAM_Away == TEAM2)
            data1 = data[cond]

            # =============================================================================
            # inverse team order
            # =============================================================================

            cond = (data.TEAM_Home == TEAM2) & (data.TEAM_Away == TEAM1)
            data2 = data[cond]

            if live:
                if data1['MIN_Home'].iloc[0] >= 60:
                    cont = False
                else:
                    pass
            else:
                cont = False
        else:
            pass

    # =============================================================================
    # predict
    # =============================================================================

    features = Tools.load_pickle('./ml_models/features_for_prediction_new')
    # TBD
    teams = features[39:99]
    for team in teams:
        data1[team] = 0

    data1[f'team_{TEAM1}'] = 1
    data1[f'opponent_{TEAM2}'] = 1

    mlp_model = load_model('keras_model.h5')
    scaler = Tools.load_pickle('keras_scaler')
    kmeans = Tools.load_pickle('keras_clusterer')

    X = scaler.transform(np.array(data1[features]).reshape(1, -1))

    imap = {
        1: 'Win',
        0: 'Loss'
    }

    y_proba_mlp = mlp_model.predict(X)[0]
    cond = y_proba_mlp >= 0.5
    y_pred_mlp = np.where(cond, 1, 0)

    print('\nDL MLP y all!!!')
    print(data1['TEAM_Home'].iloc[0], imap[y_pred_mlp[0]], y_proba_mlp[0])

    data1['nn_pred'] = imap[y_pred_mlp[0]]
    data1['nn_prob'] = y_proba_mlp[0]

    results1 = data1[['TEAM_Home', 'TEAM_Away', 'DATE_Home', 'nn_pred', 'nn_prob']]
    results1['clusters'] = kmeans.predict(np.array(results1['nn_prob']).reshape(-1, 1))

    if imap[y_pred_mlp[0]] == 'Loss':
        get_results(data1, results1, y_pred_mlp, y_proba_mlp, imap)
        return

    # =============================================================================
    # inverse order
    # =============================================================================

    teams = features[39:99]
    for team in teams:
        data2[team] = 0

    data2[f'team_{TEAM2}'] = 1
    data2[f'opponent_{TEAM1}'] = 1

    df22 = data2[features]
    df22 = df22.apply(pd.to_numeric)
    X = scaler.transform(np.array(df22).reshape(1, -1))

    y_proba_mlp2 = mlp_model.predict(X)[0]
    cond = y_proba_mlp2 >= 0.5
    y_pred_mlp2 = np.where(cond, 1, 0)

    print('\nDL MLP y all!!!')
    print(data2['TEAM_Home'].iloc[0], imap[y_pred_mlp2[0]], y_proba_mlp2[0])

    data2['nn_pred'] = imap[y_pred_mlp2[0]]
    data2['nn_prob'] = y_proba_mlp2[0]

    results2 = data2[['TEAM_Home', 'TEAM_Away', 'DATE_Home', 'nn_pred', 'nn_prob']]
    results2['clusters'] = kmeans.predict(np.array(results2['nn_prob']).reshape(-1, 1))

    if imap[y_pred_mlp2[0]] == 'Loss':
        get_results(data2, results2, y_pred_mlp2, y_proba_mlp2, imap)

    # system(f'say Dude you have predictions for the match {TEAM1} vs {TEAM2} ')


if __name__ == "__main__":
    # =============================================================================
    # inputs
    # =============================================================================

    day = 13
    month = 3

    live = True
    TEAM1 = 'BOS'
    TEAM2 = 'DAL'
    # date in yyyy/mm/dd/hh/mm format
    d1 = datetime.datetime(2022, 3, 13, 21, 43)
    predict_match(d1, day, month, live, TEAM1, TEAM2)
