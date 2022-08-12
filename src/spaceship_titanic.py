import pandas as pd
from feature_engine.selection import DropFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from src.generic_pipe import get_constant_pipe, get_configurable_pipe
from src.generic_prediction import run_generic
from src.gen import get_xy_from_dataframe
from src.kaggle_api import load_dataset
from src.settings import DATA_PATH

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


def main():
    x_train, y_train, x_test = preprocess()

    config_pipe, pipe_space = get_configurable_pipe()
    config_pipe.fit(x_train, y_train)

    print(config_pipe.steps[-1][1].feature_importances_)

    TEST[TARGET] = config_pipe.predict(x_test)
    TEST.to_csv(DATASET_PATH / "pipeline_prediction.csv", columns=["PassengerId", TARGET], index=False)


if __name__ == "__main__":

    dataset_preprocess_pipe = Pipeline(steps=[
        ("get_features", FunctionTransformer(get_features)),
        ('drop_columns', DropFeatures(["Name"]))
    ])

    const_pipe_params = {"encoder__imputer__min_value": 0}

    run_generic(
        DATASET,
        target=TARGET,
        write_columns=["PassengerId", TARGET],
        preprocess_pipe=dataset_preprocess_pipe,
        const_pipe_params=const_pipe_params,
        tune=True
    )
