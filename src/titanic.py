import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, FunctionTransformer, StandardScaler
from src.kaggle_api import get_dataset
from src.gen import train_test_from_null, get_xy_from_dataframe
from src.settings import DATA_PATH


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the raw train and test datasets for a given dataset.
    :return: train_dataset, test_dataset
    """
    # TODO: Abstract this, have a string input for dataset name, optional kwargs for train/test names.
    path = get_dataset("titanic")
    raw_train_data = pd.read_csv(path / "train.csv")
    raw_test_data = pd.read_csv(path / "test.csv")

    return raw_train_data, raw_test_data


def load_clean_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load pre-imputed train and test datasets
    :return: clean train and test datasets
    """

    clean_comb_data = pd.read_csv(DATA_PATH / "titanic" / "all_data_clean.csv", index_col=0)
    clean_train_data, clean_test_data = train_test_from_null(clean_comb_data, "Survived")

    return clean_train_data, clean_test_data


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
    out = pd.get_dummies(df["Title"], drop_first=True)

    return out


def clean_dataset_pandas(df):
    pass


def prepare_dataset_pandas(dfs: list[pd.DataFrame], scale=True, drop=None, target=None) -> list[pd.DataFrame]:
    """
    Wrangle dataset to get necessary features using pandas.
    :param dfs: Input datasets to be wrangled where the training dataset is the first item in the list to allow
        scaler to be fit.
    :param scale: Scale data based on min-max.
    :param drop: Columns to drop.
    :param target: Columns to be predicted. These will not be scaled.
    :return: Prepared datasets.
    """

    # numerical_columns = ["Fare"]
    # ordinal_columns = ["Pclass", "Age", "SibSp", "Parch"]
    categorical_columns = ["Title", "Embarked"]

    scaler = StandardScaler()

    out_df = []
    for i, df in enumerate(dfs):

        if drop is not None:
            df = df.drop(drop, axis=1)

        df["SibSp"] = df["SibSp"].clip(upper=3)
        df["Parch"] = df["Parch"].clip(upper=2)

        # Get one-hot vectors for categorical variables
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

        if scale:
            # TODO: This should be predefined but get_dummies creates additional labels, find workaround
            if target is None:
                scale_columns = df.columns
            else:
                scale_columns = [c for c in df.columns if c not in target]

            if i == 0:
                scaled_data = scaler.fit_transform(df[scale_columns])
            else:
                scaled_data = scaler.transform(df[scale_columns])

            df[scale_columns] = scaled_data

        out_df.append(df)

    return out_df


def clean_pipeline_example():
    """
    Example of a preprocessing > classification pipeline
    :return:
    """
    embarked_transformer = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder())
    clip_sibsp = FunctionTransformer(lambda x, kwargs: x.clip(**kwargs), kw_args={"kwargs": {"upper": 3}})
    clip_parch = FunctionTransformer(lambda x, kwargs: x.clip(**kwargs), kw_args={"kwargs": {"upper": 2}})

    preprocessor = ColumnTransformer(
        transformers=[
            ("fare", SimpleImputer(strategy="mean"), ["Fare"]),
            ("Embarked", embarked_transformer, ["Embarked"]),
            ("pclass_age", OrdinalEncoder(), ["Pclass", "Age"]),
            ("sibsp", clip_sibsp, ["SibSp"]),
            ("parch", clip_parch, ["Parch"]),
            ("title", OneHotEncoder(), ["Title"]),
            # ("cat",         categorical_transformer, categorical_features)
            # ("cat",         categorical_transformer, selector(dtype_include="category")),
            # ("num",         numeric_transformer, selector(dtype_exclude="category")),
        ]
    )

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("classifier", GradientBoostingClassifier(loss="log_loss", criterion="friedman_mse", n_estimators=50))]
    )

    return pipe


def run_clean_pipeline_example():
    # TODO: Convert to test
    target = "Survived"

    train_data, test_data = load_clean_data()
    _, y_train = get_xy_from_dataframe(train_data, target)

    p = clean_pipeline_example()

    p.fit(train_data, y_train)
    test_data[target] = p.predict(test_data)
    print(test_data)


def main():
    target = "Survived"
    drop_columns = ["PassengerId", "Cabin", "Ticket", "Name", "Sex"]
    clean_csv_path = DATA_PATH / "titanic" / "all_data_clean.csv"

    clean_data = pd.read_csv(clean_csv_path, index_col=0)
    clean_train_data, clean_test_data = train_test_from_null(clean_data, target)

    data = prepare_dataset_pandas(
        [clean_train_data, clean_test_data],
        drop=drop_columns,
        target=target
    )

    print(data[0])


if __name__ == "__main__":

    main()
