import numpy as np


def greedy_regression(problem: dict, train_ixs: np.ndarray, obs_values: np.ndarray, unlabeled_ixs: np.ndarray,
                      batch_size: int, **kwargs):
    """
    Simple greedy approach to black box function minimization.  Model predicts values for unlabeled points, choose
    the points with the minimum values predicted by model.
    :param problem: dictionary that defines the problem, containing keys:
        * points:       an (n_samples, n_dim) matrix of points in the space
        * model:        the sk-learn model we are training
    :param train_ixs: index into `points` of the training examples
    :param obs_labels: labels for the training examples
    :param unlabeled_ixs: np.array of the indices of the unlabeled examples
    :param batch_size: size of the batch to select
    :return:
    """
    points = problem['points']
    model = problem['model']

    # delta e is what I was predicting for materials
    delta_e_pred = model.predict(points[unlabeled_ixs])
    scores = -delta_e_pred

    # samples that are likely very small delta e
    min_delta_e_ixs = np.argpartition(scores, -batch_size)[-batch_size:]

    return unlabeled_ixs[min_delta_e_ixs]
