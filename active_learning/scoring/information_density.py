import numpy as np
from scipy.spatial import distance
from sklearn.metrics.pairwise import pairwise_distances


def information_density(problem: dict, train_ixs: np.ndarray, obs_labels: np.ndarray, unlabeled_ixs: np.ndarray,
                        batch_size: int, **kwargs) -> np.ndarray:
    """
    Score is uncertainty(x) * representativeness(x)--In particular marginal entropy of the point times its
    average distance to all other points.
    :param problem: dictionary that defines the problem, containing keys:
        * points:       an (n_samples, n_dim) matrix of points in the space
        * num_classes:  the number of different classes [0, num_classes)
        * batch_size:   number of points to query each iteration
        * num_queries:  the max number of queries we can make on the data
        * model:        the sk-learn model we are training
    :param train_ixs: index into `points` of the training examples
    :param obs_labels: labels for the training examples
    :param unlabeled_ixs: indices into problem['points'] to score
    :param batch_size:
    :param kwargs: unused
    :return: scores for each of selected_ixs
    """
    points = problem['points']
    model = problem['model']

    test_X = points[unlabeled_ixs]

    p_x = model.predict_proba(test_X)
    p_x = p_x.clip(1e-9, 1 - 1e-9)
    logp_x = np.log(p_x)
    uncertainties = -1 * (p_x * logp_x).sum(axis=1)

    # TODO: rather than use average distance, use GMM to estimate density?
    # pdistances = distance.pdist(points, metric="sqeuclidean")
    # pdistances = distance.squareform(pdistances)
    pdistances = pairwise_distances(points, metric="l1",n_jobs=-1)
    avg_distances = pdistances.sum(axis=1) / (len(points) - 1.)
    avg_distances = avg_distances[unlabeled_ixs]

    return avg_distances * uncertainties
