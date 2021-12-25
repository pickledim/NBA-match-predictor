#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 12:09:23 2020

@author: dimitrisglenis
"""

from selenium import webdriver
import time
import pandas as pd
import numpy as np
from selenium.webdriver.support.ui import Select
from selenium.webdriver import ActionChains
from selenium.webdriver.chrome.options import Options


def bx2df(text, teams, live=False):
    """
    Function that converts the scraps collected to dataframe.
    
    Parameters
    ----------
    text : string
        the scraps

    Returns
    -------
    df : dataframe
        the scraps in a dataframe format
        :param text:
        :param live:
        :param teams:

    """
    f = text.splitlines()
    col = f[0].split(' ')
    data = []
    if teams:
        for i in range(1, len(f)):
            data.append(f[i].split(' '))
    else:
        stats = []
        names = []

        for i in range(1, len(f), 2):
            names.append(f[i])
            stats.append(f[i + 1].split(' '))

        for i in range(len(names)):
            data.append([names[i]] + stats[i])

    data2 = data
    if live:
        data2 = []
        for i in range(len(data)):
            if len(data[i]) == 25:
                data2.append(data[i])
        col.remove('W/L')
    col = np.array(col)
    data = np.array(data2)
    df = pd.DataFrame(data)
    df = df.apply(pd.to_numeric, errors='ignore')
    df.columns = col

    return df


def scraps2df(text):
    """
    Function that converts the scraps collected to dataframe.
    
    Parameters
    ----------
    text : string
        the scraps

    Returns
    -------
    df : dataframe
        the scraps in a dataframe format

    """

    f = text.splitlines()
    col = f[0].split(' ')

    names = []
    stats = []

    for i in range(1, len(f), 3):
        names.append(f[i + 1])
        stats.append(f[i + 2].split(' '))

    data = []
    for i in range(len(names)):
        data.append([names[i]] + stats[i])

    col = np.array(col)
    data = np.array(data)
    df = pd.DataFrame(data)
    df = df.apply(pd.to_numeric, errors='ignore')
    df.columns = col

    return df


def boxscore_advanced(text, advanced=False, ff=False, misc=False):
    f = text.splitlines()

    if advanced:

        headers = ['TEAM', 'MATCH', 'UP', 'GAME', 'DATE', 'W/L', 'MIN', 'OFFRTG', 'DEFRTG', 'NETRTG', 'AST%', 'AST/TO',
                   'AST_RATIO', 'OREB%', 'DREB%', 'REB%', 'TOV%', 'EFG%', 'TS%', 'PACE', 'PIE']
    elif ff:
        headers = ['TEAM', 'MATCH', 'UP', 'GAME', 'DATE', 'W/L', 'MIN', 'EFG%', 'FTA_RATE', 'TOV%', 'OREB%', 'OPP_EFG%',
                   'OPP_FTA_RATE', 'OPP_TOV%', 'OPP_OREB%']
    elif misc:
        headers = ['TEAM', 'MATCH', 'UP', 'GAME', 'DATE', 'W/L', 'MIN', 'PTS_OFF_TO', '2ND_PTS', 'FBPS', 'PITP',
                   'OPP_PTS_OFF_TO', 'OPP_2ND_PTS', 'OPP_FBPS', 'OPP_PITP']
    else:
        headers = ['TEAM', 'MATCH', 'UP', 'GAME', 'DATE1', 'DATE2', 'W/L', 'MIN', '%FGA_2PT', 'GA_3PT', '%PTS_2PT',
                   '%PTS_2PT_ MR', '%PTS_3PT', '%PTS_FBPS', '%PTS_FT', '%PTS_OFF_TO', '%PTS_PITP', '2FGM_%AST',
                   '2FGM_%UAST', '3FGM_%AST', '3FGM_%UAST', 'FGM_%AST', 'FGM_%UAST']

    data = []
    for i in range(len(f)):
        data.append(f[i].split(' '))

    data = np.array(data)
    df = pd.DataFrame(data)
    df = df.apply(pd.to_numeric, errors='ignore')
    df.columns = headers

    return df


def web_scraper(url, output_file, boolean=True, boxscore=False, teams=False, boxscore_adv=False, advanced=False,
                ff=False, misc=False, live=False):
    """
    Main function that creat the df with the players statistics
    

    Parameters
    ----------
    url : string
        the url of the wanted website

    Returns
    -------
    df : dataframe
        The stats of all the players
        :param live:
        :param misc:
        :param advanced:
        :param boxscore_adv:
        :param teams:
        :param boxscore:
        :param boolean:
        :param output_file:
        :param url:
        :param ff:

    """

    start = time.time()

    options = webdriver.ChromeOptions()
    options.headless = False

    # run firefox webdriver from executable path of your choice
    driver = webdriver.Chrome(executable_path='/Users/dimitrisglenis/Downloads/chromedriver', options=options)
    print('\n================ \n')
    print('Loading the url:', url)

    # get web page
    driver.get(url)
    time.sleep(15)
    element = driver.find_element_by_xpath('//*[@id="onetrust-accept-btn-handler"]')
    ActionChains(driver).click(element).perform()
    time.sleep(20)
    if boolean:
        # show all pages
        select = Select(driver.find_element_by_xpath("/html/body/main/div/div/div[2]/div/div/nba-stat-table/div["
                                                     "1]/div/div/select"))
        select.select_by_visible_text("All")
        print('\n================ \n')
        print('Page manipulation: Successful')

    # sleep for 10s
    time.sleep(20)

    if not boxscore_adv:
        results = driver.find_elements_by_xpath("//*[@class='nba-stat-table']//*[@class='nba-stat-table__overflow']")
    else:
        results = driver.find_elements_by_xpath("/html/body/main/div/div/div[2]/div/div/nba-stat-table/div[2]/div["
                                                "1]/table/tbody")
    print('\n================ \n')
    print('Number of results', len(results))

    # loop over results
    for result in results:
        product_name = result.text

    # close driver 
    driver.quit()

    end = time.time()
    print('\n================ \n')
    if (end - start) < 120:
        print('Web Scraping duration :', end - start, 'sec')
    else:
        print('Web Scraping duration :', (end - start) / 60, 'min')

    # save to pandas dataframe
    if boxscore:
        df = bx2df(product_name, teams, live)
    elif boxscore_adv:
        df = boxscore_advanced(product_name, advanced, ff, misc)
    else:
        df = scraps2df(product_name)
    df.to_csv(output_file)

    return df


def web_scraper_fantasy_data(url, output_file, boolean=True):
    """
    Main function that create the df with the players statistics
    

    Parameters
    ----------
    url : string
        the url of the wanted website

    Returns
    -------
    df : dataframe
        The stats of all the players
        :param url:
        :param boolean:
        :param output_file:

    """

    start = time.time()

    options = Options()
    options.headless = False

    # run firefox webdriver from executable path of your choice
    driver = webdriver.Firefox(executable_path='/Users/dimitrisglenis/Downloads/geckodriver', options=options)
    print('\n================ \n')
    print('Loading the url:', url)

    # get web page
    driver.get(url)

    # sleep for 10s
    # time.sleep(10)
    if boolean:
        # show all pages
        print('\n================ \n')
        print('Page manipulation: Succesful')

    # sleep for 10s
    # time.sleep(20)

    results = driver.find_elements_by_xpath("/html/body/div[6]/div/div[4]/div[2]/div/div/div[2]/section/div[2]/div[5]")
    print('\n================ \n')
    print('Number of results', len(results))

    # loop over results
    for result in results:
        product_name = result.text

    # close driver 
    driver.quit()

    end = time.time()
    print('\n================ \n')
    print('Web Scraping duration :', end - start, 'sec')

    # save to pandas dataframe

    df = scraps2df(product_name)
    df.to_csv(output_file)
    return df
