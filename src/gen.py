import pandas as pd
from sklearn.pipeline import Pipeline


def train_test_from_null(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:

    train_data = df.dropna(subset=target)
    test_data = df[~df.index.isin(train_data.index)]

    return train_data, test_data


def get_xy_from_dataframe(df, target):

    x = df[[c for c in df.columns if c != target]]
    y = df[target] if target in df else None

    return x, y


def debug_pipeline(pipe: Pipeline, x, y=None):

    for step in pipe.steps:
        temp = Pipeline(steps=[step])
        x = temp.fit_transform(x, y)
