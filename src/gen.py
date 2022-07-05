def train_test_from_null(df, target):

    train_data = df.dropna(subset=target)
    test_data = df[~df.index.isin(train_data.index)]

    return train_data, test_data


def get_xy_from_dataframe(df, target):

    X = df[[c for c in df.columns if c != target]]
    y = df[target] if target in df else None

    return X, y