import pandas as pd
import numpy as np
from openml.datasets import get_dataset

home_best_attack_col = "home_best_attack"
home_best_defense_col = "home_best_defense"
away_best_attack_col = "away_best_attack"
away_best_defense_col = "away_best_defense"

home_avg_attack_col = "home_avg_attack"
home_avg_defense_col = "home_avg_defense"
away_avg_attack_col = "away_avg_attack"
away_avg_defense_col = "away_avg_defense"

home_std_attack_col = "home_std_attack"
home_std_defense_col = "home_std_defense"
away_std_attack_col = "away_std_attack"
away_std_defense_col = "away_std_defense"

home_players_base_str = "home_player_{number}"
away_players_base_str = "away_player_{number}"

premier_league_matches_id = 42195
serie_a_matches_id = 42196
all_players_stats_id = 42194


GK_COLUMNS = ["overall_rating"]

ATTACK_COLUMNS = ["positioning", "crossing", "finishing", "heading_accuracy", "short_passing",
                  "reactions", "volleys", "dribbling", "curve", "free_kick_accuracy", "acceleration", "sprint_speed",
                  "agility", "penalties", "vision", "shot_power", "long_shots"]

DEFENSE_COLUMNS = ["interceptions", "aggression", "marking", "standing_tackle", "sliding_tackle", "long_passing"]

COLS_TO_KEEP = ['date', "home_team_api_id", "away_team_api_id",
                'gk_home_player_1', 'gk_away_player_1', 'home_avg_attack', 'home_avg_defense',
                'home_std_attack', 'home_std_defense', 'home_best_attack', 'home_best_defense',
                'away_avg_attack', 'away_avg_defense', 'away_std_attack', 'away_std_defense',
                'away_best_attack', 'away_best_defense']

gk_column = "gk"
attack_column = "attack"
defense_column = "defense"

match_columns = ["id", "country_id", "league_id", "season", "stage", "date", "match_api_id", "home_team_api_id",
                 "away_team_api_id", "home_team_goal", "away_team_goal", "home_player_X1", "home_player_X2",
                 "home_player_X3",
                 "home_player_X4", "home_player_X5", "home_player_X6", "home_player_X7", "home_player_X8",
                 "home_player_X9",
                 "home_player_X10", "home_player_X11", "away_player_X1", "away_player_X2", "away_player_X3",
                 "away_player_X4",
                 "away_player_X5", "away_player_X6", "away_player_X7", "away_player_X8", "away_player_X9",
                 "away_player_X10",
                 "away_player_X11", "home_player_Y1", "home_player_Y2", "home_player_Y3", "home_player_Y4",
                 "home_player_Y5",
                 "home_player_Y6", "home_player_Y7", "home_player_Y8", "home_player_Y9", "home_player_Y10",
                 "home_player_Y11",
                 "away_player_Y1", "away_player_Y2", "away_player_Y3", "away_player_Y4", "away_player_Y5",
                 "away_player_Y6",
                 "away_player_Y7", "away_player_Y8", "away_player_Y9", "away_player_Y10", "away_player_Y11",
                 "home_player_1",
                 "home_player_2", "home_player_3", "home_player_4", "home_player_5", "home_player_6", "home_player_7",
                 "home_player_8",
                 "home_player_9", "home_player_10", "home_player_11", "away_player_1", "away_player_2", "away_player_3",
                 "away_player_4",
                 "away_player_5", "away_player_6", "away_player_7", "away_player_8", "away_player_9", "away_player_10",
                 "away_player_11",
                 "goal", "shoton", "shotoff", "foulcommit", "card", "cross", "corner", "possession", "B365H", "B365D",
                 "B365A"
                 ]

COLS_TO_DROP = ["shoton", "shotoff", "foulcommit", "card", "cross", "corner", "home_player_X1", "home_player_X2",
                "home_player_X3", "home_player_X4", "home_player_X5", "home_player_X6", "home_player_X7",
                "home_player_X8",
                "home_player_X9", "home_player_X10", "home_player_X11", "away_player_X1", "away_player_X2",
                "away_player_X3",
                "away_player_X4", "away_player_X5", "away_player_X6", "away_player_X7", "away_player_X8",
                "away_player_X9",
                "away_player_X10", "away_player_X11", "home_player_Y1", "home_player_Y2", "home_player_Y3",
                "home_player_Y4",
                "home_player_Y5", "home_player_Y6", "home_player_Y7", "home_player_Y8", "home_player_Y9",
                "home_player_Y10",
                "home_player_Y11", "away_player_Y1", "away_player_Y2", "away_player_Y3", "away_player_Y4",
                "away_player_Y5",
                "away_player_Y6", "away_player_Y7", "away_player_Y8", "away_player_Y9", "away_player_Y10",
                "away_player_Y11"
                ]


def _aggregate_player_attributes(player_df, columns):
    """Compute the mean for all the players for the given columns

    Parameters
    ----------
    player_df: pd.DataFrame
        The DataFrame containing the statistics of all the players
    columns: list
        The columns on which to calculate the mean

    Returns
    -------
    player_df_with_stats: pd.DataFrame
        The original DataFrame with also the aggregate stats
    """
    return player_df[columns].mean(axis=1, skipna=True)


def add_aggregate_player_stats(player_df):
    """For all the players, compute the mean of the attack columns, the defensive columns and the gk columns

    Parameters
    ----------
    player_df: pd.DataFrame
        The DataFrame containing the statistics of all the players

    Returns
    -------
    player_df_with_stats: pd.DataFrame
        The original DataFrame with also the aggregate stats

    """
    player_df[attack_column] = _aggregate_player_attributes(player_df, ATTACK_COLUMNS)
    player_df[defense_column] = _aggregate_player_attributes(player_df, DEFENSE_COLUMNS)
    player_df[gk_column] = _aggregate_player_attributes(player_df, GK_COLUMNS)
    return player_df


def retrieve_latest_stats_by_player(player_df, player_id, str_date):
    """Retrieve the latest statistics for a given player, based on a target date

    Parameters
    ----------
    player_df: pd.DataFrame
        The DataFrame containing the statistics of all the players
    player_id: int
        The id of the target player
    str_date: str
        The target date

    Returns
    -------
    latest_stats: pd.DataFrame
        The DataFrame containing the latest stats with respect to the str_date
    """
    date = pd.Timestamp(str_date)
    player_id_df = player_df[player_df.player_api_id == player_id]
    player_id_df.loc[:, "date"] = pd.to_datetime(player_id_df.loc[:, "date"])
    sorted_df = player_id_df.sort_values(by=['date'], ascending=False)
    all_stats_before_date = sorted_df[sorted_df.date < date].dropna(axis=0)
    return all_stats_before_date.iloc[0, :]


def compute_stats(match, base_player_column, stat_name):
    """For all the players of one of the two teams (goalkeeper excluded), compute the average, std and best of the
    stat_name statistics (either attack or defense)

    Parameters
    ----------
    match: pd.Series
        A series containing the match
    base_player_column: str
        The base format of the column (in our dataset, either 'home_player_{number}' or 'away_player_{number}')
    stat_name: str
        The name of the statistic for which compute the avg, std and best (in our case, either 'attack' or 'defense'

    Returns
    -------
    stats: tuple
        A tuple containing the average, the std and the best of the chosen statistic
    """
    player_stats = []
    for player_number in range(2, 12):
        base_player_col = base_player_column.format(number=player_number)
        stat_player_col = stat_name + "_" + base_player_col
        player_stats.append(match[stat_player_col])

    avg = np.nanmean(player_stats)
    std = 100 - (np.nanstd(player_stats) / np.nanmean(player_stats)) * 100
    best = np.nanmax(player_stats)

    return avg, std, best


def _aggregate_stats_per_match(match):
    """For a single match, compute the aggregate statistics of both teams and add the corresponding columns

    Parameters
    ----------
    match: pd.Series
        A single match

    Returns
    -------
    match_with_attributes: pd.Series
        The match containing the aggregate statistics
    """
    avg_home_attack, std_home_attack, best_home_attack = compute_stats(match, home_players_base_str, "attack")
    avg_home_defense, std_home_defense, best_home_defense = compute_stats(match, home_players_base_str, "defense")
    avg_away_attack, std_away_attack, best_away_attack = compute_stats(match, away_players_base_str, "attack")
    avg_away_defense, std_away_defense, best_away_defense = compute_stats(match, away_players_base_str, "defense")

    match[home_avg_attack_col] = avg_home_attack
    match[home_avg_defense_col] = avg_home_defense
    match[away_avg_attack_col] = avg_away_attack
    match[away_avg_defense_col] = avg_away_defense

    match[home_std_attack_col] = std_home_attack
    match[home_std_defense_col] = std_home_defense
    match[away_std_attack_col] = std_away_attack
    match[away_std_defense_col] = std_away_defense

    match[home_best_attack_col] = best_home_attack
    match[home_best_defense_col] = best_home_defense
    match[away_best_attack_col] = best_away_attack
    match[away_best_defense_col] = best_away_defense

    return match


def compute_aggregate_stats_per_team(df_matches):
    """For all the matches,  compute the aggregate statistics of all the teams and add the corresponding columns

    Parameters
    ----------
    df_matches: pd.DataFrame
        The DataFrame containing all the matches

    Returns
    -------
    df_matches_with_stats: pd.DataFrame
        The matches containing the aggregate statistics
    """
    matches_with_stats = []

    for index, match in df_matches.iterrows():
        matches_with_stats.append(_aggregate_stats_per_match(match))

    return pd.DataFrame(matches_with_stats)


def _insert_players_team(match, base_player_column, df_players):
    for player_number in range(1, 12):
        player_col = base_player_column.format(number=player_number)
        player_id = match[player_col]
        match_date = match['date']
        try:
            latest_player_stats = retrieve_latest_stats_by_player(df_players, player_id, match_date)
            gk_value = latest_player_stats[gk_column]
            attack_value = latest_player_stats[attack_column]
            defense_value = latest_player_stats[defense_column]
        except IndexError:
            gk_value = np.nan
            attack_value = np.nan
            defense_value = np.nan

        if player_number == 1:
            new_gk_column = gk_column + "_" + player_col
            match[new_gk_column] = gk_value
        else:
            new_attack_column = attack_column + "_" + player_col
            match[new_attack_column] = attack_value
            new_defense_column = defense_column + "_" + player_col
            match[new_defense_column] = defense_value
    return match


def _insert_player_stats(df_matches, df_players):
    matches_with_stats = []

    for index, match in df_matches.iterrows():
        match_with_home_stats = _insert_players_team(match, home_players_base_str, df_players)
        match_full_stats = _insert_players_team(match_with_home_stats, away_players_base_str, df_players)
        matches_with_stats.append(match_full_stats)

    return pd.DataFrame(matches_with_stats)


def _load_matches(league):
    """Load the DataFrame according to the league

    Parameters
    ----------
    league: str
        The name of the league

    Returns
    -------
    matches_df: pd.DataFrame
        The requested matches
    """
    if league == "England Premier League":
        target_id = premier_league_matches_id
    else:
        target_id = serie_a_matches_id

    matches_df = get_dataset(target_id).get_data(dataset_format='dataframe')[0]
    matches_df["date"] = pd.to_datetime(matches_df["date"])
    return matches_df


def _load_players():
    """Load the DataFrame containing the players' attributes

    Returns
    -------
    players_df: pd.DataFrame
        The DataFrame containing the players
    """
    players_df = get_dataset(all_players_stats_id).get_data(dataset_format='dataframe')[0]
    players_df["date"] = pd.to_datetime(players_df["date"])
    return players_df


def replace_player_with_messi(league, replaced_player_id, messi_id=30981):
    """Replace the player with id 'replaced_player_id' with the id of messi for all the matches

    Parameters
    ----------
    league: str
        The string corresponding to the name of the league
    replaced_player_id: int
        The id of the player to be replaced
    messi_id: int
        The id of messi

    Returns
    -------
    df_matches_with_messi: pd.DataFrame
        The matches with Messi
    """

    df_matches = _load_matches(league)
    df_players = _load_players()

    df_players_with_stats = add_aggregate_player_stats(df_players)

    df_matches_with_messi = df_matches.replace(replaced_player_id, messi_id)

    df_matches_stats_with_messi = _insert_player_stats(df_matches_with_messi, df_players_with_stats)

    df_matches_full_stats_with_messi = compute_aggregate_stats_per_team(df_matches_stats_with_messi)

    df_matches_useful_stats_with_messi = df_matches_full_stats_with_messi

    return df_matches_useful_stats_with_messi
