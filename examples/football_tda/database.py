import sqlite3
import pandas as pd
import numpy as np
import os

from compute_statistics import *
from openml.datasets import get_dataset

import wget

url = 'https://storage.googleapis.com/l2f-open-models/football_tda/database.sqlite'
if not os.path.isfile('database.sqlite'):
    filename = wget.download(url)
else:
    filename = 'database.sqlite'

class Database:
    pl_players = 0
    pl_matches = 0
    pl_team = 0
    conn = 0
    season = "2014/2015"
    date_time_str = "2014-08-01 00:00:00"
    rank = 0
    league = 0
    all_players_stats_id = 42194
    all_players_name_id = 42199

    def __init__(self):
        """ prepare all table """
        self.conn = sqlite3.connect(filename)
        self.pl_players = get_dataset(self.all_players_name_id).get_data(dataset_format='dataframe')[0]
        self.pl_players_attributes = get_dataset(self.all_players_stats_id).get_data(dataset_format='dataframe')[0]
        self.pl_players_attributes["date"] = pd.to_datetime(self.pl_players_attributes["date"])
        self.pl_players_attributes_small = self.pl_players_attributes[
            self.pl_players_attributes['date'] > self.date_time_str].groupby(['player_fifa_api_id']).agg(
            {'overall_rating': ['mean']})

    def calculate_ranking(self):
        self.home = pd.read_sql("SELECT home_team_api_id, SUM("
                                "  CASE "
                                "   WHEN home_team_goal > away_team_goal "
                                "       THEN 3"
                                "   WHEN  home_team_goal = away_team_goal "
                                "       THEN 1 "
                                "   ELSE 0 END ) "
                                " AS home_points "
                                " FROM Match "
                                " WHERE  league_id  "
                                " IN ( SELECT id "
                                " FROM League "
                                " WHERE name = ? ) "
                                " AND season = ? "
                                " GROUP BY home_team_api_id ",
                                self.conn,
                                params=(self.league, self.season)).dropna()
        self.away = pd.read_sql("SELECT away_team_api_id, SUM("
                                "  CASE "
                                "   WHEN home_team_goal < away_team_goal "
                                "       THEN 3"
                                "   WHEN  home_team_goal = away_team_goal "
                                "       THEN 1 "
                                "   ELSE 0 END ) "
                                " AS away_points "
                                " FROM Match "
                                " WHERE  league_id  "
                                " IN ( SELECT id "
                                " FROM League "
                                " WHERE name = ? ) "
                                " AND season = ? "
                                " GROUP BY away_team_api_id ",
                                self.conn,
                                params=(self.league, self.season)).dropna()
        self.rank = self.away.set_index('away_team_api_id').join(self.home.set_index('home_team_api_id'))
        self.rank['total_point'] = self.rank['away_points'] + self.rank['home_points']

        self.pl_team = pd.read_sql("SELECT DISTINCT M.away_team_api_id , TA.team_long_name, TA.team_short_name "
                                   "FROM Match AS M "
                                   "JOIN Team AS TA "
                                   "ON(M.away_team_api_id = TA.team_api_id) "
                                   "WHERE league_id "
                                   "IN( SELECT id "
                                   "FROM League "
                                   "WHERE name = ?) "
                                   "AND season = ? ",
                                   self.conn, params=(self.league, self.season)).dropna()

        self.pl_team = self.pl_team.set_index("away_team_api_id").join(self.rank['total_point']).sort_values(
            ['total_point'], ascending=False)

    def find_player_id_by_name(self, name):
        """ search the player in table and return the id"""

        for item in self.pl_players[['player_api_id', 'player_name', 'player_fifa_api_id']].values:
            if " " in str(item[1]):
                first_name, last_name = str(item[1]).split(" ", 1)
                if " " in name:
                    f_name, l_name = str(name).split(" ", 1)

                    if str.lower(last_name) == str.lower(l_name) and str.lower(first_name) == str.lower(f_name):
                        return item[0]
                else:

                    if str.lower(last_name) == str.lower(name):
                        return item[0]


            else:

                if str.lower(item[1]) == str.lower(name):
                    return item[0]

        return -1

    def switch_to_players_by_id(self, f_id, s_id):
        """ switch the two id player_api_id in the principal player and in the attribute player table in order to invert
        the match played by the two players"""

        i = 0

        for item in self.pl_players['player_api_id']:
            if item == f_id:
                # print(self.pl_players.at[i, 'player_api_id'])
                self.pl_players.at[i, 'player_api_id'] = s_id
                # print(self.pl_players.at[i, 'player_api_id'])
            if item == s_id:
                # print(self.pl_players.at[i, 'player_api_id'])
                self.pl_players.at[i, 'player_api_id'] = f_id
                # print(self.pl_players.at[i, 'player_api_id'])

            i = i + 1

        i = 0

        for item in self.pl_players_attributes['player_api_id']:
            if item == f_id:
                # print(self.pl_players.at[i, 'player_api_id'])
                self.pl_players_attributes.at[i, 'player_api_id'] = s_id
                # print(self.pl_players.at[i, 'player_api_id'])
            if item == s_id:
                # print(self.pl_players.at[i, 'player_api_id'])
                self.pl_players_attributes.at[i, 'player_api_id'] = f_id
                # print(self.pl_players.at[i, 'player_api_id'])

            i = i + 1

        self.pl_players.to_parquet('pl_players2.parquet')

    def switch_players_by_name(self, f_name, s_name):
        """ mask function that call Find player by id two times and then switch player by id"""

        f_id = self.find_player_id_by_name(f_name)
        s_id = self.find_player_id_by_name(s_name)

        self.switch_to_players_by_id(f_id, s_id)

    def select_player_from_team(self, team_name):
        """ Function that allow to select by terminal the player from a team. It computes the appearance of each player
        in the team  it offers a testual interface"""

        team_id = pd.read_sql('SELECT team_api_id FROM Team WHERE team_long_name = ?', self.conn,
                              params={team_name})

        player_for_team_home = pd.read_sql('SELECT home_player_2, home_player_3, '
                                           'home_player_4, home_player_5, '
                                           'home_player_6, home_player_7, home_player_8, '
                                           'home_player_9, home_player_10, home_player_11 '
                                           'FROM Match WHERE home_team_api_id = ? AND season = ?',
                                           self.conn,
                                           params=(str(team_id['team_api_id'][0]), self.season)).dropna()

        player_for_team_away = pd.read_sql('SELECT away_player_2, away_player_3, '
                                           'away_player_4, away_player_5, '
                                           'away_player_6, away_player_7, away_player_8, '
                                           'away_player_9, away_player_10, away_player_11 '
                                           'FROM Match WHERE away_team_api_id = ? AND season = ?',
                                           self.conn,
                                           params=(str(team_id['team_api_id'][0]), self.season)).dropna()

        unique_away, counts_away = np.unique(player_for_team_away, return_counts=True)
        unique_home, counts_home = np.unique(player_for_team_home, return_counts=True)

        unique_home = np.array(unique_home).astype(np.int64)
        unique_away = np.array(unique_away).astype(np.int64)

        dataframe_home_appearance = pd.DataFrame({'appearance': counts_home}, index=unique_home)
        dataframe_away_appearance = pd.DataFrame({'appearance': counts_away}, index=unique_away)

        for item in unique_away:

            if item in dataframe_home_appearance['appearance']:

                dataframe_home_appearance.at[item, 'appearance'] = dataframe_home_appearance.at[item, 'appearance'] + \
                                                                   dataframe_away_appearance.at[
                                                                       item, 'appearance']

            else:

                dataframe_home_appearance.loc[item] = dataframe_away_appearance.at[item, 'appearance']

        players = dataframe_home_appearance.join(self.pl_players.set_index('player_api_id')).set_index(
            'player_fifa_api_id').join(self.pl_players_attributes_small).sort_values(by=['appearance'], ascending=False)

        print(players[['player_name', 'appearance', ('overall_rating', 'mean')]].to_string(index=False))

    def hire_player(self):
        league_input = str.lower(input('Choose one league between "serie a" and "Premier League".\n'))

        flag = True
        while flag:
            if league_input == "serie a" or league_input == 'italy serie a':
                self.league = 'Italy Serie A'
                self.season = "2015/2016"
                self.date_time_str = '2015-08-01 00:00:00'
                flag = False
            elif league_input == 'premier league' or league_input == 'england premier league':
                self.league = 'England Premier League'
                flag = False
                self.season = "2014/2015"
                self.date_time_str = '2014-08-01 00:00:00'
            else:
                print('Warning: we don\' have ', league_input, ' please choose between serie a and premier league\n')
                league_input = str.lower(input(
                    'Choose one league between serie a and Premier League. \n'))

        self.calculate_ranking()
        print(self.pl_team.to_string(index=False))
        first_player = input('Which player do you want to insert? Please insert the name and surname \n')
        team = input('Which team? Please, mind to insert the full name \n ')
        self.select_player_from_team(team)
        second_player = input('Which player? Please insert the name and surname \n')
        # self.switch_players_by_name('Messi', second_player)
        # print('The players have been moved\n\n')
        # self.select_player_from_team(team)
        first_player_id = self.find_player_id_by_name(first_player)
        second_player_id = self.find_player_id_by_name(second_player)
        return replace_player_with_chosen_one(self.league, second_player_id, hired_player_id=first_player_id)
