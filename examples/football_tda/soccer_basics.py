#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 15:29:26 2019

@author: dantitussalajan
"""

import numpy as np


# adding some useful columns; data is dataframe that must contain the mentioned columnms -- see the parquet function
# best add it for all data at the begining
def useful_updates1(data):
    diff = np.sign(data['home_team_goal'] - data['away_team_goal'])
    # adding the results; 1=home win, 0=away win, 0.5=draw
    data['result'] = np.round((1 + diff) / 2, 1)
    # a market prediction column: 1=home has best odds, 0=away has best odds, 0.5=draw has best odds
    n = len(data)
    market_prediction = np.zeros(n) + 0.5
    for k in range(data.index[0], data.index[-1] + 1):
        r = np.argmin([data['B365A'][k], data['B365D'][k], data['B365H'][k]]) / 2
        market_prediction[k] = r
    data['market_prediction'] = market_prediction


# add elo standard vanilla elo ratings
# as there are online updates is best to compute it for all data again
def get_elo(data, K, handicap):
    n = len(data)
    # elos is the array of elos -- size 300000 is a hack to cover all possible teams ids . it must be > than all team ids to work
    # no_matches is the number of matches played by the team before this match (in the data)$
    # (this is needed because usually people use elos only after 30 matches -- rule of thumb)
    elos = np.zeros(300000) + 1500
    no_match = np.zeros(300000)
    # we construct the new columns...
    # the elo appearing in a match's row is the elo BEFORE the match
    elo_home = np.zeros(n) + 1500
    elo_away = np.zeros(n) + 1500
    match_home = np.zeros(n)
    match_away = np.zeros(n)
    for k in range(data.index[0], data.index[-1] + 1):
        # we are at match indexed by k
        # getting the teams in integer forms
        h = np.int(data['home_team_api_id'][k])
        a = np.int(data['away_team_api_id'][k])
        res = data['result'][k]
        # write elos and number of matches before the current match
        elo_home[k] = elos[h]
        elo_away[k] = elos[a]
        match_home[k] = no_match[h]
        match_away[k] = no_match[a]
        # get current elos/before the match to plug them in formulas
        elo_h = elos[h]
        elo_a = elos[a]
        # get the win/loss (no draw) probabilities coming from the current elo
        delta_h = elo_h - elo_a + handicap
        proba_h = 1 / (1 + np.power(10.0, -delta_h / 400))
        # update the elo of the two teams in the elos array; notice that it will be written in the dataset next time the teams play
        ammount_changed = K * (res - proba_h)
        elos[h] += ammount_changed
        elos[h] = np.round(elos[h])
        elos[a] -= ammount_changed
        elos[a] = np.round(elos[a])
        # update the number of matches so far
        no_match[h] += 1
        no_match[a] += 1
    data['elo_home'] = elo_home.astype(int)
    data['elo_away'] = elo_away.astype(int)
    data['match_home'] = match_home.astype(int)
    data['match_away'] = match_away.astype(int)


# add columns with market & elo probabilities
# notice this must be the same handicap (in principle) from get_elo
def useful_updates2(data, handicap):
    n = len(data)
    # market probabilities
    data['M1'] = (1 / data['B365H']) / (1 / data['B365H'] + 1 / data['B365D'] + 1 / data['B365A'])
    data['MX'] = (1 / data['B365D']) / (1 / data['B365H'] + 1 / data['B365D'] + 1 / data['B365A'])
    data['M2'] = (1 / data['B365A']) / (1 / data['B365H'] + 1 / data['B365D'] + 1 / data['B365A'])
    # elo probabilities
    E1 = np.zeros(n) + 1 / 3
    EX = np.zeros(n) + 1 / 3
    E2 = np.zeros(n) + 1 / 3
    for k in range(data.index[0], data.index[-1] + 1):
        # as Elo ratings do not include draws, we take the draw proba of the matket
        # the binary probabilities from the
        EX[k] = data['MX'][k]
        delta_h = data['elo_home'][k] - data['elo_away'][k] + handicap
        # the Elo initial probabilities
        proba_h = 1 / (1 + np.power(10.0, -delta_h / 400))
        proba_a = 1 - proba_h
        # rescale
        E1[k] = proba_h * (1 - EX[k])
        E2[k] = proba_a * (1 - EX[k])
    data['E1'] = E1
    data['EX'] = EX
    data['E2'] = E2


# Elo based prediction with >=30 matches condition

def ternary_prediction(data, barrier):
    count_elo = 0
    ok_elo = 0
    a = np.sign(data.elo_home - data.elo_away)
    for k in range(data.index[0], data.index[-1]):
        if (data['match_home'][k] < barrier) or (data['match_away'][k] < barrier):
            continue
        count_elo += 1
        elo_prediction = (1 + a[k]) / float(2)
        if data['result'][k] == elo_prediction:
            ok_elo += 1
    print('accuracy', np.round(ok_elo / float(count_elo), 3))


# FROM NOW ON CHAMPIONSHIP SIMULATIONS

# putting smaller ids for the team -- between 0-20 if data=one championship
def team_index(data):
    n = len(data)
    teams = np.sort(data.home_team_api_id.unique())
    home_team = np.zeros(n)
    away_team = np.zeros(n)
    for k in range(n):
        home_team[k] = np.where(teams == data['home_team_api_id'][k])[0][0]
        away_team[k] = np.where(teams == data['away_team_api_id'][k])[0][0]
    data['home_team'] = home_team.astype(int)
    data['away_team'] = away_team.astype(int)


# reading 3 arrays of probabilities
def read_probabilities(data, p1, px, p2):
    data['P1'] = p1
    data['PX'] = px
    data['P2'] = p2


# n is data size, d+1 is the number of teams; simulation of one championship
def one_champion(data, n, d):
    points = np.zeros(d + 1)
    for k in range(n):
        i = data['home_team'][k]
        j = data['away_team'][k]
        r = np.random.choice([3, 1, 0], p=[data['P1'][k], data['PX'][k], data['P2'][k]])
        points[i] += r
        if r == 1:
            points[j] += 1
        else:
            points[j] += 3 - r
    return points.astype(int)


# mapping the points to rankings
def points_to_rankings(points):
    d = len(points)
    ranks = np.arange(d)
    # create an intermediate array
    M = np.max(points)
    interim = np.zeros(M + 1)
    for k in range(d):
        interim[points[k]] += 1
    # assigning ranks to the team
    # ranks[j]=i means team i is raked #j
    current_rank = 0
    for k in range(M, -1, -1):
        if interim[k] > 0:
            a = np.where(points == k)[0]
            # just a trick to get a random permutation of teams with k points
            b = np.random.choice(a, len(a), replace=False)
            ranks[current_rank:current_rank + len(a)] = b
            current_rank += len(a)
    return ranks.astype(int)


# simultation, many replays, of a championships
# batches of 100 championships
def simulation_champion(data, p1, px, p2, experiment_size):
    n = len(data)
    team_index(data)
    read_probabilities(data, p1, px, p2)
    d = np.max(data.home_team)
    avg_points = np.zeros(d + 1)
    all_ranks = np.zeros((d + 1, d + 1))
    for k in range(1, experiment_size + 1):
        if k % 100 == 0:
            print('simulation batch', k // 100)
        # print('simulation',k)
        points = one_champion(data, n, d)
        ranks = points_to_rankings(points)
        for j in range(d + 1):
            all_ranks[ranks[j]][j] += 1
        avg_points = avg_points * ((k - 1) / k) + points * (1 / k)
    return avg_points, all_ranks / experiment_size


# print the expected rankings
# x=expected points
# y[i][j]=probability of team #i being ranked #j
# teams are the ordered id teams
# names must be the names of the teams, ordered by ids
def printer_ranks(x, y, teams, names):
    d = len(x)
    z = np.flip(np.sort(x))
    rankings = np.zeros(d)
    current_rank = 0
    for k in z:
        i = np.where(x == k)[0][0]
        rankings[current_rank] = i
        current_rank += 1
    # print(rankings)
    for j in range(d):
        i = np.int(rankings[j])
        print(j + 1, names[i], np.round(x[i]))
    print('probabilities to win the title, to be top 4, to be last 3')
    for j in range(d):
        i = np.int(rankings[j])
        print(j + 1, names[i], np.round(y[i][0], 3), np.round(np.sum(y[i][0:4]), 2), np.round(np.sum(y[i][-3:]), 2))
    return rankings
