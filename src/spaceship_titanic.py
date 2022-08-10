import numpy as np
import pandas as pd
from functools import partial
from hyperopt import hp, fmin, tpe, space_eval
from feature_engine.selection import DropFeatures, DropConstantFeatures, DropDuplicateFeatures
from mlflow import set_tracking_uri
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from src.gen import get_xy_from_dataframe
from src.hyper_example import objective
from src.kaggle_api import load_dataset
from src.settings import DATA_PATH, RNG_STATE

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

DATASET = "spaceship-titanic"
TARGET = "Transported"
DATASET_PATH = DATA_PATH / DATASET
set_tracking_uri(f"file://{str(DATASET_PATH)}/mlruns")


def get_clean_data() -> tuple[pd.DataFrame, pd.DataFrame]:

    train, test = load_dataset(DATASET)
    train = imputer(train)
    test = imputer(test)

    return train, test


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


def get_pipeline():

    # Feature engineering
    preprocessor = FunctionTransformer(get_features)
    imp = IterativeImputer(min_value=0, random_state=RNG_STATE)

    # Encoding + column transforms
    encoder = ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(), make_column_selector(dtype_include=object)),
            ("imputer", imp, make_column_selector(dtype_exclude=object))
        ],
        remainder="passthrough"
    )

    preprocess_pipe = Pipeline(
        steps=[
            ("get_features", preprocessor),
            ('drop_columns', DropFeatures(["Name"])),
            ('drop_constant_values', DropConstantFeatures(tol=1, missing_values='ignore')),
            ('drop_duplicates', DropDuplicateFeatures()),
            ("encoder", encoder)
        ],
        memory=str(DATASET_PATH / "tmp" / "cache")
    )

    config_pipe = Pipeline(
        steps=[
            ("selector", SelectPercentile(chi2, percentile=50)),
            ("classifier", GradientBoostingClassifier())
        ]
    )

    pipe = Pipeline(
        steps=[
            ("preprocessing", preprocess_pipe),
            ("configurable", config_pipe)
        ]
    )

    return pipe


def main():

    train, test = get_clean_data()
    train.info()
    test.info()

    x_train, y_train = get_xy_from_dataframe(train, TARGET)

    pipe = get_pipeline()

    # Testing pipeline steps (not including classifier)
    # x_train_trans = Pipeline(steps=pipe.steps[:-1]).fit_transform(X_train, y_train)

    pipe.fit(x_train, y_train)
    print(pipe.steps[-1][1].feature_importances_)
    test[TARGET] = pipe.predict(test)

    test.to_csv(DATASET_PATH / "pipeline_prediction.csv", columns=["PassengerId", TARGET], index=False)


def tune():

    pipe = get_pipeline()

    train_data, test_data = get_clean_data()
    x_train, y_train = get_xy_from_dataframe(train_data, TARGET)

    # Fit pipeline first (without classifier) to save any cache components
    Pipeline(steps=pipe.steps[:-1]).fit(x_train, y_train)

    cv = KFold(n_splits=10, shuffle=True, random_state=RNG_STATE)
    obj_func = partial(objective, estimator=pipe, x=x_train, y=y_train, cv=cv, scoring=("accuracy", "f1"))

    pipe_space = {
        "configurable__selector__percentile": hp.choice('percentile', list(range(10, 100, 10))),
        "configurable__classifier__n_estimators": hp.choice('n', [100, 200, 500]),
        "configurable__classifier__learning_rate": hp.loguniform('learning_rate', np.log(0.001), np.log(0.2)),
        "configurable__classifier__min_samples_split": hp.uniform('min_samples_split', 0.01, 0.1),
        "configurable__classifier__min_samples_leaf": hp.uniform('min_samples_leaf', 0.01, 0.1),
        "configurable__classifier__max_depth": hp.randint('max_depth', 3, 8),
        "configurable__classifier__subsample": hp.uniform('subsample', 0.6, 1.0),
    }

    best = fmin(obj_func, pipe_space, algo=tpe.suggest, max_evals=100)

    best_params = space_eval(pipe_space, best)
    pipe.set_params(**best_params)
    pipe.fit(x_train, y_train)
    test_data[TARGET] = pipe.predict(test_data)

    test_data.to_csv(DATASET_PATH / "tuned_pipeline_prediction.csv", columns=["PassengerId", TARGET], index=False)


if __name__ == "__main__":

    tune()
