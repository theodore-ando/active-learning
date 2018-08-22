from collections import defaultdict

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.svm import SVR


def _density(pts):
    """Sort of density for a set of points"""
    pts = np.array(pts)
    dists = pdist(pts)
    return len(pts) / dists.max(), squareform(dists)


def mcal_regression(problem: dict, train_ixs: np.ndarray, obs_labels: np.ndarray, unlabeled_ixs: np.ndarray,
                    batch_size: int, **kwargs) -> np.ndarray:
    """
    Multiple criteria active regression for SVMs (https://doi.org/10.1016/j.patcog.2014.02.001).
    :param problem: dictionary that defines the problem, containing keys:
        * points:       an (n_samples, n_dim) matrix of points in the space
        * model:        SVM regressor we are training
    :param train_ixs: index into `points` of the training examples
    :param obs_labels: labels for the training examples
    :param unlabeled_ixs: np.array of the indices of the unlabeled examples
    :param batch_size: size of the batch to select
    :return:
    """
    points: np.ndarray = problem['points']
    model: SVR = problem['model']
    assert isinstance(model, SVR)

    # split training points into support vectors and non support vectors
    support_mask = np.zeros(len(train_ixs), dtype=bool)
    support_mask[model.support_] = True
    train_sv_ixs = train_ixs[support_mask]
    train_not_sv_ixs = train_ixs[~support_mask]

    # train clusterer
    # extra arrays and whatnot to track indices into the points array and whether a given points was
    # a training point or not
    clusterer = DBSCAN(eps=1.0)
    clst_ixs = np.concatenate([train_not_sv_ixs, unlabeled_ixs])
    train_mask = np.zeros(clst_ixs.shape, dtype=bool)
    train_mask[:len(train_not_sv_ixs)] = True
    clst_pts = points[clst_ixs]
    clst_labels = clusterer.fit_predict(clst_pts)

    # group by cluster labels
    clst2pts = defaultdict(list)
    for pt, label, is_train, ix in zip(clst_pts, clst_labels, train_mask, clst_ixs):
        clst2pts[label].append((pt, is_train, ix))

    # find clusters that do not contain any non support vectors from training
    good_clsts = [
        label
        for label, pts in clst2pts.items()
        if not any(is_train for pt, is_train, ix in pts)
    ]

    # find the "densest" clusters
    densities = [
        (i, _density([pt for pt, is_train, ix in clst2pts[i]]))
        for i in good_clsts
    ]

    n_samples = min(batch_size, len(good_clsts))
    k_densest = sorted(densities, key=lambda x: x[1][0], reverse=True)[:n_samples]

    # sample one point from each of the densest clusters
    selected = []
    for i, (density, dists) in densities:
        dists = np.mean(dists, axis=1)
        dense_ix = np.argmin(dists)
        selected.append(clst2pts[i][dense_ix][2])

    return np.array(selected, dtype=int)
