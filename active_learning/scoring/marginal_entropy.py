import numpy as np


def marginal_entropy(problem: dict, train_ixs: np.ndarray, obs_labels: np.ndarray, unlabeled_ixs: np.ndarray,
                     batch_size: int, **kwargs) -> np.ndarray:
    """
    Score is -p(x)log[p(x)] i.e. marginal entropy of the point.
    :param problem: dictionary that defines the problem, containing keys:
        * points:       an (n_samples, n_dim) matrix of points in the space
        * num_classes:  the number of different classes [0, num_classes)
        * batch_size:   number of points to query each iteration
        * num_queries:  the max number of queries we can make on the data
        * model:        the sk-learn model we are training
    :param train_ixs: index into `points` of the training examples
    :param obs_labels: labels for the training examples
    :param unlabeled_ixs: indices into problem['points'] to score
    :param kwargs: unused
    :return: scores for each of selected_ixs
    """
    points = problem['points']
    model = problem['model']

    test_X = points[unlabeled_ixs]

    p_x = model.predict_proba(test_X)
    p_x = p_x.clip(1e-9, 1 - 1e-9)

    logp_x = np.log(p_x)

    return -1 * (p_x * logp_x).sum(axis=1)
    # return 1/ np.abs(model.decision_function(test_X))
