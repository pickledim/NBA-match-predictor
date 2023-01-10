import pandas as pd
import numpy as np
import os

from src import stats_scraper


def check_dir(directory):
    """
    Checks if the directory given exists and if not it creates it

    :param directory: str of the wanted directory
    :return:
    """

    dir_path = os.path.dirname(os.path.realpath(directory))
    check_folder = os.path.isdir(dir_path)

    # If folder doesn't exist, then create it.
    if not check_folder:
        os.makedirs(dir_path)
        print("created folder : ", dir_path)

    else:
        print(dir_path, "folder already exists.")


def pre_process_cols(data, training_dataset=False):
    """
    Keeps only the useful columns and renames them

    :param data: pd.DataFrame dataset
    :param training_dataset: boolean checks if it is for training or for live scrapping
    :return: pd.DataFrame dataset with modified columns
    """

    venue = []
    opp = []
    for i, row in data.MATCHUP.str.split(' ').iteritems():
        venue.append(row[1])
        opp.append(row[2])

    venue_map = {
        'vs.': 'Home',
        '@': "Away"
    }

    data.drop(['MATCHUP'], axis=1, inplace=True)
    data['OPPONENT'] = opp
    data['VENUE'] = venue
    data['VENUE'] = data.VENUE.map(venue_map)
    if training_dataset:

        drop = ['SEASON_YEAR',
                'BLKA',
                'PFD',
                'TEAM_ID',
                'TEAM_NAME',
                'GAME_ID',
                'GP_RANK',
                'W_RANK',
                'L_RANK',
                'W_PCT_RANK',
                'MIN_RANK',
                'FGM_RANK',
                'FGA_RANK',
                'FG_PCT_RANK',
                'FG3M_RANK',
                'FG3A_RANK',
                'FG3_PCT_RANK',
                'FTM_RANK',
                'FTA_RANK',
                'FT_PCT_RANK',
                'OREB_RANK',
                'DREB_RANK',
                'REB_RANK',
                'AST_RANK',
                'TOV_RANK',
                'STL_RANK',
                'BLK_RANK',
                'BLKA_RANK',
                'PF_RANK',
                'PFD_RANK',
                'PTS_RANK',
                'PLUS_MINUS_RANK']
    else:

        drop = ['TEAM_ID',
                'TEAM_NAME',
                'GAME_ID']

    mapper = {'TEAM_ABBREVIATION': 'TEAM',
              'VENUE': 'VENUE',
              'OPPONENT': 'OPPONENT',
              'GAME_DATE': 'DATE',
              'WL': 'W/L',
              'MIN': 'MIN',
              'PTS': 'PTS',
              'FGM': 'FGM',
              'FGA': 'FGA',
              'FG_PCT': 'FG%',
              'FG3M': '3PM',
              'FG3A': '3PA',
              'FG3_PCT': '3P%',
              'FTM': 'FTM',
              'FTA': 'FTA',
              'FT_PCT': 'FT%',
              'OREB': 'OREB',
              'DREB': 'DREB',
              'REB': 'REB',
              'AST': 'AST',
              'TOV': 'TOV',
              'STL': 'STL',
              'BLK': 'BLK',
              'PF': 'PF',
              'PLUS_MINUS': '+/-'}

    data.drop(drop, axis=1, inplace=True)
    data.rename(columns=mapper, inplace=True)

    return data


def hollinger_formula(data):
    """
    Creates a column that evaluates the team - basketball on paper Hollinger formula
    :param data: pd.DataFrame dataset
    :return: pd.DataFrame dataset
    """

    data['VALUE'] = data['PTS'] + 0.79 * data['AST'] + 0.85 * data['OREB'] + \
                    0.35 * data['DREB'] + 1.2 * data['STL'] + 0.85 * data['BLK'] \
                    - 0.85 * (data['FGA'] - data['FGM']) - 0.45 * (data['FTA'] - data['FTM']) - \
                    1.2 * data['TOV'] - 0.85 * data['PF']
    return data


def concat_home_away_stats(df):
    """
    Returns a df in which each match has all the data of Home and Away Teams

    :param df: pd.DataFrame dataset
    :return: pd.DataFrame dataset
    """

    def concat_stats(data, list_df):
        """
        For each date finds the rows where the home team is the corresponding away team
            of the same match and concatenates the stats
        :param data: pd.DataFrame rows grouped by date of the dataset
        :param list_df: list empty list that will store the data batches
        :return: pd.DataFrame rows grouped by date of the dataset with data form home & away teams
        """

        for i, row in data.iterrows():
            cond = (data.loc[:, 'TEAM'].isin([row['OPPONENT']]))  # and (tmp_df.loc[:,'GAME'].isin([row['TEAM']]))
            tmp_df = data[cond]
            team1 = np.array(row)
            team2 = np.array(tmp_df.iloc[0])
            arr = np.concatenate((team1, team2))
            list_df.append(arr)
        return list_df

    features = df.columns.tolist()
    list_df = []
    df.groupby(by=['DATE']).apply(concat_stats, list_df)
    df = pd.DataFrame(list_df)

    string1 = '_Home'
    string2 = '_Away'
    team_features = list(map(lambda orig_string: orig_string + string1, features))
    opp_features = list(map(lambda orig_string: orig_string + string2, features))
    cols = team_features + opp_features

    df.columns = cols
    return df


def get_dummies(data):
    """
    Creates dummie columns for all the teams for Home and Away categories
    :param data: pd.DataFrame dataset
    :return: pd.DataFrame dataset with the dummie columns
    """

    team_dummies = pd.get_dummies(data.TEAM_Home, prefix='team').astype(int)
    ven_dummies = pd.get_dummies(data.VENUE_Home, prefix='venue').astype(int)
    op_dummies = pd.get_dummies(data.TEAM_Away, prefix='opponent').astype(int)
    data = pd.concat([data, ven_dummies, team_dummies, op_dummies], axis=1)

    data['PTS'] = data['PTS_Home'] + data['PTS_Away']

    return data


def feature_eng(data):
    """
    Creates features taken from Basketball on paper book
    :param data: pd.DataFrame dataset
    :return: pd.DataFrame dataset
    """

    data['POSSESION_Home'] = 0.96 * (
            data['FGA_Home'] - data['OREB_Home'] + data['TOV_Home'] + 0.44 * data['FTA_Home'])

    data['POSSESION_Away'] = 0.96 * (
            data['FGA_Away'] - data['OREB_Away'] + data['TOV_Away'] + 0.44 * data['FTA_Away'])

    data['PACE'] = (data['POSSESION_Home'] + data['POSSESION_Away']) / 2
    # =============================================================================
    #     trads total            
    # =============================================================================
    data['tot_FGM'] = data['FGM_Home'] + data['FGM_Away']
    data['tot_FGA'] = data['FGA_Home'] + data['FGA_Away']

    data['tot_3PM'] = data['3PM_Home'] + data['3PM_Away']
    data['tot_3PA'] = data['3PA_Home'] + data['3PA_Away']

    data['tot_FTM'] = data['FTM_Home'] + data['FTM_Away']
    data['tot_FTA'] = data['FTA_Home'] + data['FTA_Away']

    data['tot_REB'] = data['REB_Home'] + data['REB_Away']
    data['tot_DREB'] = data['DREB_Home'] + data['DREB_Away']
    data['tot_OREB'] = data['OREB_Home'] + data['OREB_Away']

    data['tot_STL'] = data['STL_Home'] + data['STL_Away']
    data['tot_AST'] = data['AST_Home'] + data['AST_Away']
    data['tot_TOV'] = data['TOV_Home'] + data['TOV_Away']
    data['tot_BLK'] = data['BLK_Home'] + data['BLK_Away']
    data['tot_PF'] = data['PF_Home'] + data['PF_Away']

    # =============================================================================
    #   trads diff
    # =============================================================================
    data['diff_FGM'] = data['FGM_Home'] - data['FGM_Away']
    data['diff_FGA'] = data['FGA_Home'] - data['FGA_Away']

    data['diff_3PM'] = data['3PM_Home'] - data['3PM_Away']
    data['diff_3PA'] = data['3PA_Home'] - data['3PA_Away']

    data['diff_FTM'] = data['FTM_Home'] - data['FTM_Away']
    data['diff_FTA'] = data['FTA_Home'] - data['FTA_Away']

    data['diff_REB'] = data['REB_Home'] - data['REB_Away']
    data['diff_DREB'] = data['DREB_Home'] - data['DREB_Away']
    data['diff_OREB'] = data['OREB_Home'] - data['OREB_Away']

    data['diff_STL'] = data['STL_Home'] - data['STL_Away']
    data['diff_AST'] = data['AST_Home'] - data['AST_Away']
    data['diff_TOV'] = data['TOV_Home'] - data['TOV_Away']
    data['diff_BLK'] = data['BLK_Home'] - data['BLK_Away']
    data['diff_PF'] = data['PF_Home'] - data['PF_Away']

    # =============================================================================

    data['TOVg%_Home'] = data['TOV_Home'] / (data['TOV_Home'] + data['TOV_Away'])
    data['TOVg%_Away'] = data['TOV_Away'] / (data['TOV_Home'] + data['TOV_Away'])

    data['REBg%_Home'] = data['REB_Home'] / (data['REB_Home'] + data['REB_Away'])
    data['REBg%_Away'] = data['REB_Away'] / (data['REB_Home'] + data['REB_Away'])

    data['DREBg%_Home'] = data['DREB_Home'] / (data['DREB_Home'] + data['DREB_Away'])
    data['DREBg%_Away'] = data['DREB_Away'] / (data['DREB_Home'] + data['DREB_Away'])

    data['OREBg%_Home'] = data['OREB_Home'] / (data['OREB_Home'] + data['OREB_Away'])
    data['OREBg%_Away'] = data['OREB_Away'] / (data['OREB_Home'] + data['OREB_Away'])

    data['2PM_Home'] = data['FGM_Home'] - data['3PM_Home']
    data['2PM_Away'] = data['FGM_Away'] - data['3PM_Away']

    data['2PM%_Home'] = 2 * data['2PM_Home'] / data['PTS_Home']
    data['2PM%_Away'] = 2 * data['2PM_Away'] / data['PTS_Away']

    data['3PM%_Home'] = 3 * data['3PM_Home'] / data['PTS_Home']
    data['3PM%_Away'] = 3 * data['3PM_Away'] / data['PTS_Away']

    data['FTM%_Home'] = 1 * data['FTM_Home'] / data['PTS_Home']
    data['FTM%_Away'] = 1 * data['FTM_Away'] / data['PTS_Away']

    data['STLg%_Home'] = data['STL_Home'] / (data['STL_Home'] + data['STL_Away'])
    data['STLg%_Away'] = data['STL_Away'] / (data['STL_Home'] + data['STL_Away'])

    data['BLKg%_Home'] = data['BLK_Home'] / (data['BLK_Home'] + data['BLK_Away'])
    data['BLKg%_Away'] = data['BLK_Away'] / (data['BLK_Home'] + data['BLK_Away'])

    data['STL/TOV_Home'] = data['STL_Home'] / data['TOV_Home']
    data['STL/TOV_Away'] = data['STL_Away'] / data['TOV_Away']

    data['OFF_RTG_Home'] = data['PTS_Home'] / data['POSSESION_Home'] * 100
    data['OFF_RTG_Away'] = data['PTS_Away'] / data['POSSESION_Away'] * 100

    data['DEF_RTG_Home'] = data['PTS_Away'] / data['POSSESION_Home'] * 100
    data['DEF_RTG_Away'] = data['PTS_Home'] / data['POSSESION_Away'] * 100

    data['NET_RTG_Home'] = data['OFF_RTG_Home'] / data['DEF_RTG_Home']
    data['NET_RTG_Away'] = data['OFF_RTG_Away'] / data['DEF_RTG_Away']

    data['TSA_Home'] = data['FGA_Home'] + 0.44 * data['FTA_Home']
    data['TSA_Away'] = data['FGA_Away'] + 0.44 * data['FTA_Away']

    data['TS%_Home'] = data['PTS_Home'] / (2 * data['TSA_Home'])
    data['TS%_Away'] = data['PTS_Away'] / (2 * data['TSA_Away'])

    data['eFG%_Home'] = (data['FGM_Home'] + 0.5 * data['3PM_Home']) / data['FGM_Home']
    data['eFG%_Away'] = (data['FGM_Away'] + 0.5 * data['3PM_Away']) / data['FGM_Away']

    data['AST%_Home'] = data['AST_Home'] / data['FGM_Home'] * 100
    data['AST%_Away'] = data['AST_Away'] / data['FGM_Away'] * 100

    data['BLK%_Home'] = data['BLK_Home'] / (data['FGA_Away'] - data['3PA_Away']) * 100
    data['BLK%_Away'] = data['BLK_Away'] / (data['FGA_Home'] - data['3PA_Home']) * 100

    data['DREB%_Home'] = data['DREB_Home'] / (data['DREB_Home'] + data['OREB_Away']) * 100
    data['DREB%_Away'] = data['DREB_Away'] / (data['DREB_Away'] + data['OREB_Home']) * 100

    data['OREB%_Home'] = data['OREB_Home'] / (data['OREB_Home'] + data['DREB_Away']) * 100
    data['OREB%_Away'] = data['OREB_Away'] / (data['OREB_Away'] + data['DREB_Home']) * 100

    data['STL%_Home'] = data['STL_Home'] / data['POSSESION_Away'] * 100
    data['STL%_Away'] = data['STL_Away'] / data['POSSESION_Home'] * 100

    data['TOV%_Home'] = data['TOV_Home'] / (data['FGA_Home'] + 0.44 * data['FTA_Home'] + data['TOV_Home']) * 100
    data['TOV%_Away'] = data['TOV_Away'] / (data['FGA_Away'] + 0.44 * data['FTA_Away'] + data['TOV_Away']) * 100

    data['FF_Home'] = 0.4 * data['eFG%_Home'] + 0.25 * data['TOV%_Home'] + 0.2 * (
                data['OREB_Home'] + data['DREB_Home']) + 0.15 * data['FT%_Home']
    data['FF_Away'] = 0.4 * data['eFG%_Away'] + 0.25 * data['TOV%_Away'] + 0.2 * (
                data['OREB_Away'] + data['DREB_Away']) + 0.15 * data['FT%_Away']

    data['GS_Home'] = data['PTS_Home'] + 0.4 * data['FGM_Home'] - 0.7 * data['FGA_Home'] - 0.4 * (
                data['FTA_Home'] - data['FTM_Home']) \
                      + 0.7 * data['OREB_Home'] + 0.3 * data['DREB_Home'] + data['STL_Home'] + 0.7 * data[
                          'AST_Home'] - 0.4 * data['PF_Home'] - data['TOV_Home']
    data['GS_Away'] = data['PTS_Away'] + 0.4 * data['FGM_Away'] - 0.7 * data['FGA_Away'] - 0.4 * (
                data['FTA_Away'] - data['FTM_Away']) \
                      + 0.7 * data['OREB_Away'] + 0.3 * data['DREB_Away'] + data['STL_Away'] + 0.7 * data[
                          'AST_Away'] - 0.4 * data['PF_Away'] - data['TOV_Away']

    return data


def get_data_from_2015():
    """
    Scrappes all the matches played on Regular Season from 2015 up to 2021
    :return: pd.DataFrame dataset
    """

    kwargs = {'DateFrom': '',
              'DateTo': ''}

    list_df = []
    for i in range(15, 21):
        kwargs['Season'] = f'20{i:02}-{i+1:02}'
        print(f'Getting data from Season: {kwargs["Season"]}')
        _name = os.path.join(os.getcwd(), 'new_data', f'teams_boxscore_trad_2k{i}_first_half.csv')
        check_dir(_name)
        tmp_df = stats_scraper.web_scraper(_name, training_dataset=True, **kwargs)
        list_df.append(tmp_df)
    data = pd.concat(list_df, axis=0, ignore_index=True)

    return data


def clean_data(data):
    """
    Replaces infinty and nan values with zero

    :param data: pd.DataFrame dataset
    :return: pd.DataFrame dataset
    """

    data = data.apply(pd.to_numeric, errors='ignore')
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data = data.fillna(0)

    return data





