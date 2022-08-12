from functools import partial
from hyperopt import fmin, tpe, STATUS_OK
from sklearn.pipeline import Pipeline
from typing import Any


def objective(params, estimator, x, y, eval_func=None, **kwargs):

    estimator.set_params(**params)

    if eval_func is not None:
        score = eval_func(estimator, x, y, params=params, fit_params=kwargs)

    else:
        score = estimator.fit(x, y, **kwargs)

    return {"loss": -score, "status": STATUS_OK}


def tune_pipe(x, y, pipe: Pipeline, param_space: dict[str, Any], eval_func=None):

    # Fit pipeline first (without classifier) to save any cache components
    # TODO: Does this actually cache for the input pipe since we are creating a new pipeline?
    Pipeline(steps=pipe.steps[:-1]).fit(x, y)
    obj_func = partial(objective, estimator=pipe, eval_func=eval_func, x=x, y=y)

    return fmin(obj_func, param_space, algo=tpe.suggest, max_evals=100)
