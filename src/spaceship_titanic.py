import mlflow
import numpy as np
import pandas as pd
from functools import partial
from hyperopt import space_eval
from feature_engine.selection import DropFeatures
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from typing import Any
from src import hyper_tuning
from src.generic_pipe import get_constant_pipe, get_configurable_pipe
from src.gen import get_xy_from_dataframe
from src.kaggle_api import load_dataset
from src.settings import DATA_PATH, RNG_STATE

DATASET = "spaceship-titanic"
TARGET = "Transported"
DATASET_PATH = DATA_PATH / DATASET
TRAIN, TEST = load_dataset(DATASET)
# TODO: Have incoming variable that sets new experiment if not None
# exp_id = mlflow.create_experiment(NAME)


def imputer(df: pd.DataFrame) -> pd.DataFrame:

    # Filling
    # TODO: Some empty values are in a group with non-empty values
    # train["HomePlanet"] = train["HomePlanet"].fillna(value="Unknown")
    # train["CryoSleep"] = train["CryoSleep"].fillna(method="Median")
    # train["Cabin"] = train["CryoSleep"].fillna(value="NA/NA/NA")
    # train["Destination"] = train["Destination"].fillna(method="Mode")

    # TODO: Cabin 0 exists, try using NaN if encoder function allows missing, otherwise max + 1?
    # TODO: Sum spend first, use within IterativeImputer?
    column_fills = {
        # "HomePlanet": "unknown",
        # "CryoSleep": df["CryoSleep"].mode()[0],
        # "Destination": df["Destination"].mode()[0],
        # "Age": df["Age"].median(),
        # "VIP": df["VIP"].mode()[0],
        # "Name": "unknown unknown",
        # "Cabin": "unknown/0/unknown",
        "RoomService": 0,
        "FoodCourt": 0,
        "ShoppingMall": 0,
        "Spa": 0,
        "VRDeck": 0
    }

    df = df.fillna(column_fills)

    return df


def get_features(df):

    df = imputer(df)

    # TODO: How to avoid data leakage?
    trans_col = "PassengerId"
    groups = df[trans_col].apply(lambda x: x.split("_")[0]).astype(float)
    df["GroupMembers"] = groups.map(groups.value_counts())
    df = df.drop(columns=[trans_col])

    trans_col = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    df["LuxSpend"] = df[trans_col].sum(axis=1)
    df = df.drop(columns=trans_col)

    df["CryoSleep"] = df["CryoSleep"].astype(float)

    # trans_col = "Name"
    # df[["FirstName", "LastName"]] = df["Name"].str.split(" ", expand=True)
    # df["LastName"] = df[trans_col].apply(lambda x: x.split(" ")[1])
    # df = df.drop(columns=[trans_col])

    trans_col = "Cabin"
    split_cols = ["CabinDeck", "CabinNum", "CabinSide"]
    df[split_cols] = df[trans_col].str.split("/", expand=True)
    df["CabinNum"] = df["CabinNum"].astype(float)
    df = df.drop(columns=trans_col)

    return df


def custom_constant_pipe():

    dataset_constant_pipe = Pipeline(steps=[
        ("get_features", FunctionTransformer(get_features)),
        ('drop_columns', DropFeatures(["Name"]))
    ])

    params = {"encoder__imputer__min_value": 0}

    generic_constant_pipe = get_constant_pipe()
    generic_constant_pipe.set_params(**params)

    return Pipeline(steps=dataset_constant_pipe.steps + generic_constant_pipe.steps)


def preprocess():

    x_train, y_train = get_xy_from_dataframe(TRAIN, TARGET)
    x_test, _ = get_xy_from_dataframe(TEST, TARGET)

    constant_pipe = custom_constant_pipe()

    # Testing pipeline steps (not including classifier)
    # x_train_trans = Pipeline(steps=pipe.steps[:-1]).fit_transform(x_train, y_train)

    x_train = constant_pipe.fit_transform(x_train)
    x_test = constant_pipe.transform(x_test)

    return x_train, y_train, x_test


def cv_mlflow_score(estimator, x, y, params=None, score_func=None, **kwargs):

    with mlflow.start_run():

        if params is None:
            params = estimator.get_params()

        for k, v in params.items():
            mlflow.log_param(k.split("__")[-1], v)

        out = cross_validate(estimator, x, y, **kwargs)

        if score_func is None:
            score = out.get("test_score")

            if score is None:
                raise KeyError("test_score not found in estimator output, if using a custom or multi-scorer, a "
                               "score_func must be provided")

        else:
            score = score_func(out)

    return score


def scorer(out: dict[str, Any]):

    overall_score = np.inf
    for metric, values in out.items():
        if "test_" in metric:
            # Taking worst case
            score = values.min()
            mlflow.log_metric(metric, score)
            # TODO: Temporary, one metric may always be less than another etc.
            overall_score = min(overall_score, score)

    return overall_score


def tune():
    x_train, y_train, x_test = preprocess()

    pipe, pipe_space = get_configurable_pipe()
    cv = KFold(n_splits=10, shuffle=True, random_state=RNG_STATE)

    eval_func = partial(
        cv_mlflow_score,
        score_func=scorer,
        cv=cv,
        scoring=("accuracy", "f1")
    )

    best = hyper_tuning.tune_pipe(x_train, y_train, pipe, pipe_space, eval_func=eval_func)

    best_params = space_eval(pipe_space, best)
    pipe.set_params(**best_params)
    pipe.fit(x_train, y_train)

    TEST[TARGET] = pipe.predict(x_test)
    TEST.to_csv(DATASET_PATH / "tuned_pipeline_prediction.csv", columns=["PassengerId", TARGET], index=False)


def main():
    x_train, y_train, x_test = preprocess()

    config_pipe, pipe_space = get_configurable_pipe()
    config_pipe.fit(x_train, y_train)

    print(config_pipe.steps[-1][1].feature_importances_)

    TEST[TARGET] = config_pipe.predict(x_test)
    TEST.to_csv(DATASET_PATH / "pipeline_prediction.csv", columns=["PassengerId", TARGET], index=False)


if __name__ == "__main__":

    main()
