import pandas as pd
from feature_engine.selection import DropFeatures, DropConstantFeatures, DropDuplicateFeatures
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from src.gen import get_xy_from_dataframe
from src.kaggle_api import load_dataset
from src.settings import DATA_PATH


def imputer(df: pd.DataFrame) -> pd.DataFrame:

    # Filling
    # TODO: Some empty values are in a group with non-empty values
    # train["HomePlanet"] = train["HomePlanet"].fillna(value="Unknown")
    # train["CryoSleep"] = train["CryoSleep"].fillna(method="Median")
    # train["Cabin"] = train["CryoSleep"].fillna(value="NA/NA/NA")
    # train["Destination"] = train["Destination"].fillna(method="Mode")

    # TODO: Cabin 0 exists, try using NaN if encoder function allows missing, otherwise max + 1?
    column_fills = {
        "HomePlanet": "unknown",
        "CryoSleep": df["CryoSleep"].mode()[0],
        "Destination": df["Destination"].mode()[0],
        "Age": df["Age"].median(),
        "VIP": df["VIP"].mode()[0],
        "Name": "unknown unknown",
        "Cabin": "unknown/0/unknown",
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


if __name__ == "__main__":

    dataset = "spaceship-titanic"
    dataset_path = DATA_PATH / dataset
    train, test = load_dataset(dataset)
    target = "Transported"

    train.info()
    print(train)
    for c in train.columns:
        print(train[c].value_counts())

    train = imputer(train)
    test = imputer(test)
    train.info()
    test.info()

    X_train, y_train = get_xy_from_dataframe(train, target)

    # Feature engineering
    preprocessor = FunctionTransformer(get_features)
    xx = preprocessor.fit_transform(X_train, y_train)

    encoder = ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(), make_column_selector(dtype_include=object)),
        ],
        remainder="passthrough"
    )

    pipe = Pipeline(steps=[
        ("get_features", preprocessor),
        ('drop_columns', DropFeatures(["Name"])),
        ('drop_constant_values', DropConstantFeatures(tol=1, missing_values='ignore')),
        ('drop_duplicates', DropDuplicateFeatures()),
        ("encoder", encoder),
        ("selector", SelectPercentile(chi2, percentile=50)),
        ("clf", GradientBoostingClassifier())
    ])

    pipe.fit(X_train, y_train)
    print(pipe.steps[-1][1].feature_importances_)
    test[target] = pipe.predict(test)

    test.to_csv(dataset_path / "pipeline_prediction.csv", columns=["PassengerId", target], index=False)
