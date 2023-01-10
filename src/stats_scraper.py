#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 20:45:38 2022

@author: dimitrisglenis
"""

import requests
import pandas as pd


def web_scraper(output_file, training_dataset=False, **kwargs):
    """
    Scrappes the matches from nba.stats works for live scrapping as well as for static scrapping

    :param output_file: str Directory to save the results
    :param training_dataset: boolean specifies if it is for live or static scrapping
    :param kwargs: dict dictionary that specifies the dates and season of the wanted dataset

    :return: pd.DataFrame the wanted dataset
    """

    browser = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/108.0.0.0 Safari/537.36',
        'Referer': 'https://www.nba.com/'}

    if 'GameSegment' not in list(kwargs.keys()):
        kwargs['GameSegment'] = 'First Half'
        _entire = False
    else:
        _entire = True

    if kwargs['DateFrom'] == '' and kwargs['DateTo'] == '':

        all_games = True

    else:

        all_games = False
        date_from = kwargs['DateFrom'].split('/')
        day_from = date_from[0]
        month_from = date_from[1]
        year_from = date_from[2]

        date_to = kwargs['DateTo'].split('/')
        day_to = date_to[0]
        month_to = date_to[1]
        year_to = date_to[2]

    print('Fetching Data')
    
    if training_dataset:

        payload = {'DateFrom': kwargs['DateFrom'],
                    'DateTo': kwargs['DateTo'],
                    'GameSegment': kwargs['GameSegment'],
                    'LastNGames': '0',
                    'LeagueID': '00',
                    'Location': '',
                    'MeasureType': 'Base',
                    'Month': '0',
                    'OpponentTeamID': '0',
                    'Outcome': '',
                    'PORound': '0',
                    'PaceAdjust': 'N',
                    'PerMode': 'Totals',
                    'Period': '0',
                    'PlusMinus': 'N',
                    'Rank': 'N',
                    'Season': kwargs['Season'],
                    'SeasonSegment': '',
                    'SeasonType': 'Regular Season',
                    'ShotClockRange': '',
                    'VsConference': '',
                    'VsDivision': ''}

        if all_games:
            url = f'https://stats.nba.com/stats/teamgamelogs?DateFrom=&DateTo=&GameSegment=First%20Half&LastNGames=0&' \
                  f'LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&' \
                  f'PerMode=Totals&Period=0&PlusMinus=N&Rank=N&Season={kwargs["Season"]}&SeasonSegment=&SeasonType=' \
                  f'Regular%20Season&ShotClockRange=&VsConference=&VsDivision='
        else:
            url = f'https://stats.nba.com/stats/teamgamelogs?DateFrom={month_from}%2F{day_from}%2F{year_from}&DateTo=' \
                  f'{month_to}%2F{day_to}%2F{year_to}&GameSegment=First%20Half&LastNGames=0&LeagueID=00&Location=&' \
                  f'MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=Totals&Period' \
                  f'=0&PlusMinus=N&Rank=N&Season={kwargs["Season"]}&SeasonSegment=&SeasonType=Regular%20Season&Shot' \
                  f'ClockRange=&VsConference=&VsDivision='

        if _entire:
            url = f"https://stats.nba.com/stats/teamgamelogs?DateFrom=&DateTo=&GameSegment=&LastNGames=0&LeagueID=00&" \
                  f"Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=" \
                  f"Totals&Period=0&PlusMinus=N&Rank=N&Season={kwargs['Season']}&SeasonSegment=&SeasonType=" \
                  f"Regular Season&ShotClockRange=&VsConference=&VsDivision="


    else:

        payload = {
            'Counter': '1000',
            'DateFrom': kwargs['DateFrom'],
            'DateTo': kwargs['DateTo'],
            'Direction': 'DESC',
            'LeagueID': '00',
            'PlayerOrTeam': 'T',
            'Season': kwargs['Season'],
            'SeasonType': 'Regular Season',
            'Sorter': 'DATE'}

        url = f'https://stats.nba.com/stats/leaguegamelog?Counter=1000&DateFrom={month_from}%2F{day_from}%2F' \
              f'{year_from}&DateTo={month_to}%2F{day_to}%2F{year_to}&Direction=DESC&LeagueID=00&PlayerOrTeam=' \
              f'T&Season={kwargs["Season"]}&SeasonType=Regular%20Season&Sorter=DATE'

    jsonData = requests.get(url, headers=browser, params=payload).json()

    rows = jsonData['resultSets'][0]['rowSet']
    columns = jsonData['resultSets'][0]['headers']
    
    print('Done')
    
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_file)
    
    return df


