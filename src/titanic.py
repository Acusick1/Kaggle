import pandas as pd
from feature_engine.selection import DropFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from src.generic_prediction import run_generic

TARGET = "Survived"
DATASET = "titanic"


def get_title(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to extract title from name
    :param df: Input titanic dataframe
    :return: Dataframe with encoded titles
    """
    # TODO: Function is specific to titanic dataset, currently no checks to validate format of input dataframe
    # TODO: Currently overworked, this should just extract title and other functions should deal with outliers and
    #  encoding.
    df["Title"] = df["Name"].str.extract(r",\s?(\w*).{1}")

    is_male = df["Sex"] == "male"
    is_female = df["Sex"] == "female"
    outlier_male = is_male & (~df["Title"].isin(["Mr", "Master"]))
    df.loc[outlier_male, "Title"] = "Mr"

    # All men under 18 = Master, over = Mr
    df.loc[is_male & (df["Age"] >= 18), "Title"] = "Mr"
    df.loc[is_male & (df["Age"] < 18), "Title"] = "Master"

    outlier_female = is_female & (~df["Title"].isin(["Miss", "Mrs"]))
    df.loc[outlier_female, "Title"] = "Mrs"

    # All women over 18 = Mrs, under = Miss
    df.loc[is_female & (df["Age"] >= 18), "Title"] = "Mrs"
    df.loc[is_female & (df["Age"] < 18), "Title"] = "Miss"

    return df


def get_features(df):

    df = get_title(df)
    # TODO: Make configurable?
    df["SibSp"] = df["SibSp"].clip(upper=3)
    df["Parch"] = df["Parch"].clip(upper=2)

    return df


def custom_pipeline():

    drop_cols = ["PassengerId", "Cabin", "Ticket", "Name", "Sex"]

    pipe = Pipeline(steps=[
        ("get_features", FunctionTransformer(get_features)),
        ("drop_columns", DropFeatures(drop_cols))
    ])

    return pipe


if __name__ == "__main__":

    run_generic(
        DATASET,
        target=TARGET,
        write_columns=["PassengerId", TARGET],
        preprocess_pipe=custom_pipeline(),
        tune=True
    )
