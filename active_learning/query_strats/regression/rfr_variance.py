import numpy as np

from sklearn.ensemble import RandomForestRegressor


def _score(estimators, points):
    # find the prediction that each estimator would place on each point
    # predictions: [n_samples, n_estimators]
    predictions = np.hstack([tree.predict(points).reshape(-1, 1) for tree in estimators])

    # calculate the variance in the estimators' predictions for each sample
    variances = np.var(predictions, axis=1)

    return variances


def rfr_variance(problem: dict, train_ixs: np.ndarray, obs_values: np.ndarray, unlabeled_ixs: np.ndarray,
                 batch_size: int, **kwargs):
    """
    AL for regression based on RandomForestRegressor.  Like uncertainty sampling in the classification setting!
    Use estimates from the trees in the forest, compute variance as uncertainty.
    :param problem: dictionary that defines the problem, containing keys:
        * points:       an (n_samples, n_dim) matrix of points in the space
        * model:        the RandomForestRegressor model we are training
    :param train_ixs:
    :param obs_values:
    :param unlabeled_ixs:
    :param batch_size:
    :param kwargs:
    :return:
    """
    points = problem['points']
    model: RandomForestRegressor = problem['model']
    assert isinstance(model, RandomForestRegressor)

    trees = model.estimators_
    unlabeled_pts = points[unlabeled_ixs]

    variances = _score(trees, unlabeled_pts)

    # uncertain samples
    uncertain_ixs = np.argpartition(variances, -batch_size)[-batch_size:]

    return unlabeled_ixs[uncertain_ixs]