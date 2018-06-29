import numpy as np
from sklearn.mixture import GaussianMixture


def _calc_diversities(gmm, points, S, X_test):
    S_scores = gmm.score(points[S])
    S_score = np.sum(S_scores)

    log_probs = gmm.score(X_test)
    scores = -(log_probs + S_score) / (len(S) + 1)

    return scores


def three_ds(problem, L, obs_labels, U, batch_size, **kwargs):
    """
    Selects points based on distance, density, and diversity as found in
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5949421
    Importantly:
        * distance(x) = log[ p(c1|x) / p(c2|x) ]
          where :math:`c1 = argmax_{c} p(c|x)` and :math:`c2 = argmax_{C\setminus\{c1\}} p(c|x)`.
        * density(x) = p(x)
          according to the mixture model used (Gaussian Mixture Model)

    :param problem:
    :param L: np.ndarray of the indices of labeled points
    :param obs_labels:
    :param U: np.ndarray of the indices of unlabeled points
    :param score:
    :param batch_size:
    :param kwargs:
    :return:
    """
    points = problem['points']
    model = problem['model']

    dwc = kwargs.get("dwc", 0.5)  # diversity weighting coefficient

    X_train = points[L]
    X_test = points[U]

    # calculate weighting factor
    probs = model.predict_proba(X_test)
    eps = 1.0 / len(probs) * np.sum(1 - np.argmax(probs, axis=1))

    # pseudo-distance that works on models that do not have a real decision boundary
    log_probs = np.log(probs)
    maxes = log_probs.max(axis=1)
    mins = log_probs.min(axis=1)
    distances = maxes - mins

    # density function via mixture model
    gmm = GaussianMixture()
    gmm.fit(X_train)
    densities = gmm.score_samples(X_test)

    # select the first point
    x = U[np.argmax((1-eps) * (1-distances) + eps * densities)]

    alpha = (1 - dwc) * (1 - eps)  # coefficient normalization
    beta = (1 - dwc) - alpha       # coefficient normalization

    S = np.array([x], dtype=int)

    while len(S) < batch_size:
        diversities = _calc_diversities(gmm, points, S, X_test)
        x = U[np.argmax(alpha*(1-distances) + beta*densities + dwc*diversities)]

        S = np.append(S, x)

    return S