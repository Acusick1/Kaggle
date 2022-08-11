import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from sklearn.utils import all_estimators
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score
from src.settings import RNG_STATE

ESTIMATOR_TYPES = ("classifier", "regressor", "cluster", "transformer")
DEFAULT_KFOLD = KFold(n_splits=10, shuffle=True, random_state=RNG_STATE)


# create the dataset
def get_random_dataset(n_samples=100, type_filter=ESTIMATOR_TYPES[0]):

    # TODO: kwargs to specify below arguments
    if type_filter == "classifier":

        features, target = make_classification(
            n_samples=n_samples,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            random_state=1
        )

    elif type_filter == "regressor":

        features, target = make_regression(
            n_samples=n_samples,
            n_features=20,
            n_informative=15,
            random_state=1
        )
    else:
        raise NotImplementedError(f"Random dataset generation for type: {type_filter} not implemented.")

    return features, target


def get_estimators(type_filter: str = ESTIMATOR_TYPES[0]):

    if type_filter not in ESTIMATOR_TYPES:
        ValueError(f"Input type_filter ({type_filter}) must be one of: {ESTIMATOR_TYPES}")

    estimators = all_estimators(type_filter=type_filter)

    all_types = []
    for name, est in estimators:
        try:
            print('Appending', name)
            all_types.append(est())
        except TypeError as e:
            print(e)

    return all_types


# evaluate the model using a given test condition
def evaluate_model(cv, model, features, target, type_filter):

    if type_filter == ESTIMATOR_TYPES[0]:
        scoring = "accuracy"
    elif type_filter == ESTIMATOR_TYPES[1]:
        scoring = "neg_root_mean_squared_error"
    else:
        raise NotImplementedError()

    # evaluate the model
    scores = cross_val_score(model, features, target, scoring=scoring, cv=cv, n_jobs=-1)
    # return scores
    return np.mean(scores), np.std(scores)


def plot_results(res):

    fig, axs = plt.subplots(2)

    colors = cm.turbo(np.linspace(0, 1, len(res)))

    # scatter plot of results
    for model_res, colour in zip(res, colors):

        output = model_res.get("output")

        if output:
            axs[0].scatter(output["cv_mean"], output["ideal_mean"], label=model_res["model"], color=colour)
            axs[1].scatter(output["cv_std"], output["ideal_std"], label=model_res["model"], color=colour)
            # plt.errorbar(output["cv_mean"], output["ideal_mean"],
            #             xerr=output["cv_std"],
            #             yerr=output["ideal_std"])

    # label the plot
    fig.suptitle('10-fold CV vs LOOCV')
    axs[0].set(xlabel='Mean Accuracy (10-fold CV)', ylabel='Mean Accuracy (LOOCV)')
    axs[1].set(xlabel='Std Accuracy (10-fold CV)', ylabel='Std Accuracy (LOOCV)')
    handles, labels = axs[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc="center left")
    fig.legend(handles, labels, bbox_to_anchor=(0.85, 0.5), loc='center left', ncol=1)
    fig.tight_layout()
    # show the plot
    plt.show()


def test_estimators(features, target, models=None, type_filter=ESTIMATOR_TYPES[0], kfold=DEFAULT_KFOLD):

    ideal_cv = LeaveOneOut()

    if models is None:
        # Get all possible models
        models = get_estimators(type_filter)

    results = []
    # evaluate each model
    for model in models:

        name = type(model).__name__
        output = dict()

        # evaluate model using each test condition
        try:
            cv_mean, cv_std = evaluate_model(kfold, model, features, target, type_filter)
            ideal_mean, ideal_std = evaluate_model(ideal_cv, model, features, target, type_filter)

            output = {
                "cv_mean": cv_mean,
                "cv_std": cv_std,
                "ideal_mean": ideal_mean,
                "ideal_std": ideal_std
            }

            print(
                '>%s: ideal=%.3f \u00B1 %.3f, cv=%.3f \u00B1 %.3f'
                % (name, float(ideal_mean), float(ideal_std), float(cv_mean), float(cv_std))
            )

        except ValueError as e:
            print(f"{name} failed. Error: \n {e}.")

        results.append({
            "model": name,
            "output": output
        })

    plot_results(results)


if __name__ == "__main__":

    model_types = ESTIMATOR_TYPES[1]
    x, y = get_random_dataset(type_filter=model_types)
    test_estimators(x, y, type_filter=model_types)
