import pandas as pd
import numpy as np
from openml.datasets import get_dataset

from utils import read_pickle
import soccer_basics

from giotto.pipeline import Pipeline
from sub_space_extraction import SubSpaceExtraction
from giotto.homology import VietorisRipsPersistence

from cross_validation import extract_features_for_prediction


COLUMNS_TO_KEEP = ["home_best_attack", "home_best_defense", "home_avg_attack", "home_avg_defense",
                   "home_std_attack", "home_std_defense", "gk_home_player_1",
                   "away_avg_attack", "away_avg_defense", "away_std_attack", "away_std_defense",
                   "away_best_attack", "away_best_defense", "gk_away_player_1"
                   ]

pl_team_names = ['Burnley', 'Leicester City', 'Chelsea', 'Manchester City', 'Southampton', 'Sunderland',
                 'Tottenham Hotspur', 'Liverpool', 'West Ham United', 'West Bromwich Albion', 'Hull City',
                 'Everton', 'Arsenal', 'Crystal Palace', 'Swansea City', 'Queens Park Ranger', 'Stoke City',
                 'Aston Villa', 'Manchester United', 'Newcastle United']

serie_a_team_names = ['Sassuolo', 'Atalanta', 'Chievo Verona', 'Empoli', 'Fiorentina', 'Palermo', 'Lazio', 'Milan',
                      'Udinese', 'Inter', 'Roma', 'Torino', 'Bologna', 'Napoli', 'Hellas Verona', 'Sampdoria',
                      'Juventus', 'Frosinone', 'Genoa', 'Carpi']

teams_with_messi = pd.DataFrame([('Chelsea', '%+d' % 0, ' 0.25', '0.84', '0.00', ' 0.41', '0.94', '0.00'),
                                 ('Manchester City', '%+d' % 1, '0.59', '0.97', '0.00', '0.58', '0.97', '0.00'),
                                 ('Arsenal', '%+d' % 0, '0.05', '0.56', '0.00', '0.17', '0.82', '0.00'),
                                 ('Manchester United', '%+d' % 1, ' 0.10', '0.69', '0.00', '0.17', '0.81', '0.00'),
                                 ('Tottenham', '%+d' % 1, '0.03', '0.44', '0.00', '0.06', '0.56', '0.00'),
                                 ('Liverpool', '%+d' % 3, '0.01', ' 0.17', '0.01', '0.10', '0.66', '0.00'),
                                 ('Southampton', '%+d' % 0, '0.00', '0.01', '0.14', '0.01', '0.19', '0.01'),
                                 ('Swansea City', '%+d' % 0, '0.00', '0.01', '0.16', '0.01', '0.04', '0.08'),
                                 ('Stoke City', '%+d' % 2, '0.00', '0.01', '0.15', '0.01', '0.16', '0.01'),
                                 ('Crystal Palace', '%+d' % 2, '0.00', '0.00', '0.29', '0.00', '0.05', '0.07'),
                                 ('Everton', '%+d' % 4, '0.01', '0.22', '0.01', '0.09', '0.20', '0.01'),
                                 ('West Ham United', '%+d' % 4, '0.00', '0.01', '0.18', '0.01', '0.11', '0.03'),
                                 ('West Bromwich', '%+d' % 5, '0.00', '0.01', '0.25', '0.00', '0.06', '0.07'),
                                 ('Leicester City', '%+d' % 5, '0.00', '0.00', '0.42', '0.19', '0.72', '0.00'),
                                 ('Newcastle United', '%+d' % 9, '0.00', '0.02', '0.11', '0.01', '0.24', '0.01'),
                                 ('Aston Villa', '%+d' % 8, '0.00', '0.01', '0.15', '0.01', '0.11', '0.02'),

                                 ('Sunderland', '%+d' % 9, '0.00', '0.01', '0.31', '0.00', '0.04', '0.08'),

                                 ('Hull City', '%+d' % 9, '0.00', '0.01', '0.30', '0.00', '0.03', '0.09'),
                                 ('Burnley', '%+d' % 10, '0.00', '0.00', '0.31', '0.00', '0.04', '0.11'),
                                 ('QPR', '%+d' % 11, '0.00', '0.00', '0.22', '0.00', '0.13', '0.02'),

                                 ], columns=['Team', 'Delta Pos.', 'Pr. Win', 'Pr. TOP 4', 'Pr. Rel.',
                                             'Pr. Win. with Messi', 'Pr. TOP 4. with Messi', 'Pr. Rel. with Messi']
                                )


def compute_final_standings(prob_match_df, championship='premier league'):
    p1 = prob_match_df.home_team_prob.reset_index(drop=True)
    px = prob_match_df.draw_prob.reset_index(drop=True)
    p2 = prob_match_df.away_team_prob.reset_index(drop=True)
    x, y = soccer_basics.simulation_champion(prob_match_df, p1, px, p2, 1000)

    if championship == 'premier league':
        names = pl_team_names
    else:
        names = serie_a_team_names

    teams = np.sort(prob_match_df.home_team_api_id.unique())
    soccer_basics.printer_ranks(x, y, teams, names)


def get_pipeline(top_feat_params):
    pipeline = Pipeline([('extract_point_clouds', SubSpaceExtraction(**top_feat_params)),
                         ('create_diagrams', VietorisRipsPersistence(n_jobs=-1))])
    return pipeline


def get_best_params():
    cv_output = read_pickle('cv_output.pickle')
    best_model_params, top_feat_params, top_model_feat_params, *_ = cv_output

    return top_feat_params, top_model_feat_params


def get_useful_cols(players_df):
    return players_df[COLUMNS_TO_KEEP]


def load_dataset():
    x_y = get_dataset(42188).get_data(dataset_format='array')[0]
    x_train_with_topo = x_y[:, :-1]
    y_train = x_y[:, -1]
    return x_train_with_topo, y_train


def extract_x_test_features(x_train, y_train, players_df, pipeline):
    """Extract the topological features from the test set. This requires also the train set

    Parameters
    ----------
    x_train:
        The x used in the training phase
    y_train:
        The y used in the training phase
    players_df: pd.DataFrame
        The DataFrame containing the matches with all the players, from which to extract the test set
    pipeline: Pipeline
        The Giotto pipeline

    Returns
    -------
    x_test:
        The x_test with the topological features
    """
    x_train_no_topo = x_train[:, :14]
    y_test = np.zeros(len(players_df))  # Artificial y_test for features computation
    x_test_topo = extract_features_for_prediction(x_train_no_topo, y_train, players_df.values, y_test, pipeline)

    return x_test_topo


def get_team_ids(players_df):
    """Get the team ids contained in the players_df DataFrame

    Parameters
    ----------
    players_df: pd.DataFrame
        The DataFrame containing all the matches
    """
    return players_df[['home_team_api_id', 'away_team_api_id']]


def get_probabilities(model, x_test, team_ids):
    """Get the probabilities on the outcome of the matches contained in the test set

    Parameters
    ----------
    model:
        The model (must have the 'predict_proba' function)
    x_test:
        The test set
    team_ids: pd.DataFrame
        The DataFrame containing, for each match in the test set, the ids of the two teams
    Returns
    -------
    probabilities:
        The probabilities for each match in the test set
    """
    prob_pred = model.predict_proba(x_test)
    prob_match_df = pd.DataFrame(data=prob_pred, columns=['away_team_prob', 'draw_prob', 'home_team_prob'])
    prob_match_df = pd.concat([team_ids.reset_index(drop=True), prob_match_df], axis=1)
    return prob_match_df
