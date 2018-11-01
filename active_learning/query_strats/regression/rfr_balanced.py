import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors


def _score_variances(estimators, points):
    # find the prediction that each estimator would place on each point
    # predictions: [n_samples, n_estimators]
    predictions = np.hstack([tree.predict(points).reshape(-1, 1) for tree in estimators])

    # calculate the variance in the estimators' predictions for each sample
    variances = np.var(predictions, axis=1)

    return variances


def _score_densities(nn: NearestNeighbors, points: np.ndarray):
    """
    Average distance of each point to its ten nearest neighbors--gives estimate of density of neighborhood around
    each point
    """
    dists, ixs = nn.kneighbors(points, n_neighbors=11)

    # lop off first column of all 0s due to points being their own closest neighbor
    dists = dists[:, 1:]

    return 1 / dists.mean(axis=1)


def rfr_balanced(problem: dict, train_ixs: np.ndarray, obs_values: np.ndarray, unlabeled_ixs: np.ndarray,
                 batch_size: int, **kwargs):
    """
    Active learning for regression based on random forest regressor.  Balances three criteria: uncertainty, density,
    and greediness.
    :param problem: dictionary that defines the problem, containing keys:
        * points:       an (n_samples, n_dim) matrix of points in the space
        * model:        RandomForestRegressor
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

    # use nearest neighbors computer to estimate densities, and if possible store the model
    # to avoid recomputing
    nn_model: NearestNeighbors = problem.get("_nn_model")
    if nn_model is None:
        nn_model = NearestNeighbors().fit(points)
        problem["_nn_model"] = nn_model

    trees = model.estimators_
    unlabeled_pts = points[unlabeled_ixs]

    # compute variance in predictions - measures uncertainty
    variances = _score_variances(trees, unlabeled_pts)
    variances = variances / variances.max()

    # potentially recompute density of neighborhood around each point
    # density = how representative a point is of overall data
    if "densities" not in problem:
        problem['densities'] = _score_densities(nn_model, points)
    densities = problem['densities'][unlabeled_ixs]
    densities = densities / densities.max()

    # predict value for each point = greedy criterion
    pred_vals = -1 * model.predict(unlabeled_pts)
    pred_vals = pred_vals - pred_vals.min()
    pred_vals = pred_vals / pred_vals.max()

    scores = variances + densities + pred_vals

    # uncertain samples
    uncertain_ixs = np.argpartition(scores, -batch_size)[-batch_size:]

    return unlabeled_ixs[uncertain_ixs]
