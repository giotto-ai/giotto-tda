import pandas as pd
import numpy as np

import giotto as o
from giotto.pipeline import Pipeline
from giotto.homology import VietorisRipsPersistence
from giotto.diagrams import Amplitude
from openml.datasets import get_dataset
from tqdm import tqdm

from itertools import chain, combinations

from sklearn.ensemble import RandomForestClassifier

from sub_space_extraction import SubSpaceExtraction
from utils import write_pickle


def extract_topological_features(diagrams):
    metrics = ['bottleneck', 'wasserstein', 'landscape', 'betti', 'heat']
    new_features = []
    for metric in metrics:
        amplitude = Amplitude(metric=metric)
        new_features.append(amplitude.fit_transform(diagrams))
    new_features = np.concatenate(new_features, axis=1)
    return new_features


def compute_match_result(df):
    return np.sign(df['home_team_goal'] - df['away_team_goal'])


def extract_features_for_prediction(x_train, y_train, x_test, y_test, pipeline):
    shift = 10
    top_features = []
    all_x_train = x_train
    all_y_train = y_train
    for i in tqdm(range(0, len(x_test), shift)):
        if i+shift > len(x_test):
            shift = len(x_test) - i
        batch = np.concatenate([all_x_train, x_test[i: i + shift]])
        batch_y = np.concatenate([all_y_train, y_test[i: i + shift].reshape((-1,))])
        diagrams_batch, _ = pipeline.fit_transform_resample(batch, batch_y)
        new_features_batch = extract_topological_features(diagrams_batch[-shift:])
        top_features.append(new_features_batch)
        all_x_train = np.concatenate([all_x_train, batch[-shift:]])
        all_y_train = np.concatenate([all_y_train, batch_y[-shift:]])
    final_x_test = np.concatenate([x_test, np.concatenate(top_features, axis=0)], axis=1)
    return final_x_test


def _check_no_repetitions(tuple_list):
    elems = [x[0] for x in tuple_list]
    return len(np.unique(elems)) == len(tuple_list)


def _powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)  # allows duplicate elements
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def construct_model_param_dictionary(parameters):
    tuple_dictionary = []

    for key in parameters.keys():
        for value in parameters[key]:
            tuple_dictionary.append((key, value))

    valid_combinations = []
    for i, combo in enumerate(_powerset(tuple_dictionary), 1):
        if len(combo) == len(parameters):
            if _check_no_repetitions(combo):
                print(combo)
                valid_combinations.append(combo)

    return valid_combinations


def best_combination(list_of_dictionaries):
    return sorted(list_of_dictionaries, key=lambda x: x["score"])[-1]


class CrossValidation:
    def __init__(self, k_mins, k_maxs, dist_percentages, **model_parameters):
        self.dist_percentages = dist_percentages
        self.k_mins = k_mins
        self.k_maxs = k_maxs
        self.model_parameters = model_parameters

    def _validate_k_fold_top(self, model, x_train, y_train, x_test, y_test):
        validation_quantities = []

        for k_min in self.k_mins:
            for k_max in self.k_maxs:
                for dist_percentage in self.dist_percentages:
                    print(f"k_min, k_max, dist_percentage: {k_min}, {k_max}, {dist_percentage}")
                    pipeline_list = [('extract_subspaces', SubSpaceExtraction(dist_percentage=dist_percentage,
                                                                              k_min=k_min, k_max=k_max,
                                                                              metric="euclidean", n_jobs=-1)),
                                     ('compute_diagrams', VietorisRipsPersistence(n_jobs=-1))]
                    top_pipeline = Pipeline(pipeline_list)

                    diagrams_train, _ = top_pipeline.fit_transform_resample(x_train, y_train)

                    top_features_train = extract_topological_features(diagrams_train)

                    x_train_model = np.concatenate([x_train, top_features_train], axis=1)
                    model.fit(x_train_model, y_train)

                    x_test_model = extract_features_for_prediction(x_train, y_train, x_test, y_test, top_pipeline)

                    score = model.score(x_test_model, y_test)
                    output_dictionary = {"k_min": k_min, "k_max": k_max,
                                         "dist_percentage": dist_percentage, "score": score}
                    validation_quantities.append(output_dictionary)

        return validation_quantities

    def _validate_k_fold_model(self, x_train, y_train, x_test, y_test):
        validation_quantities = []

        valid_combinations = construct_model_param_dictionary(self.model_parameters)
        for combination in valid_combinations:
            dictionary = {key: value for key, value in combination}

            model = RandomForestClassifier(**dictionary)
            model.fit(x_train, y_train)
            score = model.score(x_test, y_test)
            dictionary["score"] = score
            validation_quantities.append(dictionary)

        return validation_quantities

    def cross_validate(self, full_x, full_y, splitting_dates):
        train_split_date = splitting_dates[0]
        val_split_date = splitting_dates[1]
        end_date = splitting_dates[2]

        train_x = full_x[(full_x.date < train_split_date) | (full_x.date >= end_date)]
        train_y = full_y[(full_x.date < train_split_date) | (full_x.date >= end_date)]

        val_x = full_x[(full_x.date >= train_split_date) & (full_x.date < val_split_date)]
        val_y = full_y[(full_x.date >= train_split_date) & (full_x.date < val_split_date)]

        test_x = full_x[(full_x.date >= val_split_date) & (full_x.date < end_date)]
        test_y = full_y[(full_x.date >= val_split_date) & (full_x.date < end_date)]

        train_x.pop("date")
        val_x.pop("date")
        test_x.pop("date")

        train_x = train_x.values
        train_y = train_y.values
        val_x = val_x.values
        val_y = val_y.values
        test_x = test_x.values
        test_y = test_y.values

        print("START VALIDATING MODEL")
        models_cv = self._validate_k_fold_model(train_x, train_y, val_x, val_y)
        best_model_params = best_combination(models_cv)
        best_model_params.pop("score")
        best_model = RandomForestClassifier(**best_model_params)

        best_model.fit(train_x, train_y)

        score = best_model.score(test_x, test_y)
        print(f'score no_top {score}')
        print(f'best model parameters no_top {best_model_params}')

        print("START VALIDATING PARAMS")
        topo_cv = self._validate_k_fold_top(best_model, train_x, train_y, val_x, val_y)
        best_topo = best_combination(topo_cv)
        best_topo.pop("score")
        best_topo_pipeline_list = [('extract_subspaces', SubSpaceExtraction(**best_topo)),
                                   ('compute_diagrams', VietorisRipsPersistence(n_jobs=-1))]
        best_topo_pipeline = Pipeline(best_topo_pipeline_list)

        train_x_for_test = np.concatenate([train_x, val_x], axis=0)
        train_y_for_test = np.concatenate([train_y, val_y], axis=0)

        diagrams_train, _ = best_topo_pipeline.fit_transform_resample(train_x_for_test, train_y_for_test)

        print("EXTRACTING TOPOLOGICAL FEATURES TRAIN")
        top_features_train = extract_topological_features(diagrams_train)

        x_train_model = np.concatenate([train_x_for_test, top_features_train], axis=1)
        best_model.fit(x_train_model, train_y_for_test)

        print("EXTRACTING TOPOLOGICAL FEATURES TEST")
        x_test_model = extract_features_for_prediction(x_train_model, train_y_for_test,
                                                       test_x, test_y, best_topo_pipeline)

        score_top = best_model.score(x_test_model, test_y)

        val_x_with_topo = extract_features_for_prediction(train_x, train_y, val_x, val_y, best_topo_pipeline)

        print('START VALIDATING MODEL WITH OPTIMAL TOPOLOGY')
        model_config_with_topo = self._validate_k_fold_model(x_train_model, train_y, val_x_with_topo, val_y)
        best_model_config_with_topo = best_combination(model_config_with_topo)
        best_model_config_with_topo.pop('score')

        best_model_with_topo = RandomForestClassifier(**best_model_config_with_topo)
        best_model_with_topo.fit(x_train_model, train_y_for_test)

        score_best_topo_and_model = best_model_with_topo.score(x_test_model, test_y)
        print(f'score best model and topo_feat {score_best_topo_and_model}')

        return best_model_params, best_topo, best_model_config_with_topo, score, score_top, score_best_topo_and_model


if __name__ == "__main__":
    COLUMNS_TO_KEEP = ["date", "home_team_goal", "away_team_goal",
                       "home_best_attack", "home_best_defense", "home_avg_attack", "home_avg_defense",
                       "home_std_attack", "home_std_defense", "gk_home_player_1",
                       "away_avg_attack", "away_avg_defense", "away_std_attack", "away_std_defense",
                       "away_best_attack", "away_best_defense", "gk_away_player_1"
                       ]

    train_split_date = pd.Timestamp("2013-08-01")
    val_split_date = pd.Timestamp("2014-08-01")
    end_date = pd.Timestamp("2015-08-01")

    k_mins = [25, 50, 75]
    k_maxs = [75, 125, 175]
    distances = [0.05, 0.10]
    model_params = {"n_estimators": [1000], "max_depth": [None, 10, 20], 'random_state': [52],
                    "max_features": [None, 'sqrt', 'log2', 1/3, 1/2]}

    df = get_dataset(42197).get_data(dataset_format='dataframe')[0]
    df = df[COLUMNS_TO_KEEP]
    y = compute_match_result(df)
    df.pop('home_team_goal')
    df.pop('away_team_goal')

    cv = CrossValidation(k_mins=k_mins, k_maxs=k_maxs, dist_percentages=distances, **model_params)
    cv_output = cv.cross_validate(df, y, (train_split_date, val_split_date, end_date))
    print(cv_output)
    write_pickle("cv_output.pickle", cv_output)
