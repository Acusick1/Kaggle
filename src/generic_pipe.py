import numpy as np
from hyperopt import hp
from feature_engine.selection import DropConstantFeatures, DropDuplicateFeatures
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from typing import Any
from src.settings import RNG_STATE

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

"""
Currently using separate pipelines for constant (preprocessing) and variable (configurable) steps. These could be used
in a single pipeline if constant steps could be cached. However the memory parameter of pipeline does not have the 
desired effect, so to avoid repetition of expensive and constant preprocessing (e.g. IterativeImputer), separate
pipelines are required.
"""


def get_constant_pipe():

    # Encoding + column transforms
    encoder = ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(), make_column_selector(dtype_include=object)),
            ("imputer", IterativeImputer(random_state=RNG_STATE), make_column_selector(dtype_exclude=object))
        ],
        remainder="passthrough"
    )

    preprocess_pipe = Pipeline(
        steps=[
            ('drop_constant_values', DropConstantFeatures(tol=1, missing_values='ignore')),
            ('drop_duplicates', DropDuplicateFeatures()),
            ("encoder", encoder)
        ],
    )

    return preprocess_pipe


def get_configurable_pipe() -> tuple[Pipeline, dict[str, Any]]:

    config_pipe = Pipeline(
        steps=[
            ("selector", SelectPercentile(chi2)),
            ("classifier", GradientBoostingClassifier())
        ]
    )

    pipe_space = {
        "selector__percentile": hp.choice('percentile', list(range(10, 100, 10))),
        "classifier__n_estimators": hp.choice('n', [100, 200, 500]),
        "classifier__learning_rate": hp.loguniform('learning_rate', np.log(0.001), np.log(0.2)),
        "classifier__min_samples_split": hp.uniform('min_samples_split', 0.01, 0.1),
        "classifier__min_samples_leaf": hp.uniform('min_samples_leaf', 0.01, 0.1),
        "classifier__max_depth": hp.randint('max_depth', 3, 8),
        "classifier__subsample": hp.uniform('subsample', 0.6, 1.0),
    }

    return config_pipe, pipe_space


if __name__ == "__main__":

    get_constant_pipe()
    get_configurable_pipe()
