import mlflow
import numpy as np
from datetime import datetime
from functools import partial
from hyperopt import space_eval
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from typing import Any, Optional
from src import hyper_tuning
from src.gen import get_xy_from_dataframe
from src.generic_pipe import get_constant_pipe, get_configurable_pipe
from src.kaggle_api import get_dataset, load_dataset
from src.settings import RNG_STATE


def run_generic(
        dataset_name: str,
        target: str,
        write_columns: Optional[list[str]] = None,
        preprocess_pipe: Optional[Pipeline] = None,
        const_pipe_params: Optional[dict[str, Any]] = None,
        tune: bool = False
        ) -> None:

    dataset_path = get_dataset(dataset_name)
    train, test = load_dataset(dataset_name)
    constant_pipe = get_constant_pipe()
    config_pipe, config_params = get_configurable_pipe()

    if const_pipe_params:
        constant_pipe.set_params(**const_pipe_params)

    if preprocess_pipe:
        constant_pipe = Pipeline(steps=preprocess_pipe.steps + constant_pipe.steps)

    x_train, y_train = get_xy_from_dataframe(train, target)
    x_test, _ = get_xy_from_dataframe(test, target)

    # For debugging
    # from src.gen import debug_pipeline
    # debug_pipeline(constant_pipe, x_train, y_train)

    x_train = constant_pipe.fit_transform(x_train)
    x_test = constant_pipe.transform(x_test)

    out_file = "pipeline_prediction.csv"

    if tune:
        exp_id = mlflow.create_experiment(f"{dataset_name}-{datetime.now()}".replace(" ", "-"))
        cv = KFold(n_splits=10, shuffle=True, random_state=RNG_STATE)

        eval_func = partial(
            cv_mlflow_score,
            score_func=scorer,
            cv=cv,
            scoring=("accuracy", "f1"),
            exp_id=exp_id
        )

        best = hyper_tuning.tune_pipe(x_train, y_train, config_pipe, param_space=config_params, eval_func=eval_func)

        best_params = space_eval(config_params, best)
        config_pipe.set_params(**best_params)

        out_file = "_".join(["tuned", out_file])

    config_pipe.fit(x_train, y_train)

    test[target] = config_pipe.predict(x_test)

    if write_columns is None:
        write_columns = [target]

    elif target not in write_columns:
        write_columns.append(target)

    test.to_csv(dataset_path / out_file, columns=write_columns, index=False)


def cv_mlflow_score(estimator, x, y, params=None, score_func=None, exp_id=None, **kwargs):

    with mlflow.start_run(experiment_id=exp_id):

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
