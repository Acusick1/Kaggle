import numpy as np
from functools import partial
from hyperopt import hp, fmin, tpe
from sklearn.model_selection import cross_validate
from src.gen import get_xy_from_dataframe
from src.titanic import load_clean_data, clean_pipeline_example


def objective(params, estimator=None, x=None, y=None):

    estimator.set_params(**params)
    out = cross_validate(estimator, x, y)
    loss = 1 - out["test_score"].mean()

    return loss


def main():

    pipe = clean_pipeline_example()

    target = "Survived"

    train_data, test_data = load_clean_data()
    _, y_train = get_xy_from_dataframe(train_data, target)

    obj_func = partial(objective, estimator=pipe, x=train_data, y=y_train)

    pipe_space = {
        "classifier__learning_rate": hp.loguniform('learning_rate', np.log(0.005), np.log(0.2)),
        "classifier__min_samples_split": hp.uniform('min_samples_split', 0.1, 0.5),
        "classifier__min_samples_leaf": hp.uniform('min_samples_leaf', 0.1, 0.5),
        "classifier__max_depth": hp.choice('max_depth', [5, 8]),
        "classifier__subsample": hp.uniform('subsample', 0.1, 0.5),
    }

    for i in range(10):
        fmin(obj_func, pipe_space, algo=tpe.suggest, max_evals=100)


if __name__ == "__main__":

    main()
