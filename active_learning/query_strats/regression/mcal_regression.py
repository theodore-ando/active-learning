from active_learning.query_strats.base import ModelBasedQueryStrategy
from active_learning.problem import ActiveLearningProblem

from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.svm import SVR
from collections import defaultdict
from random import sample
from typing import List
import numpy as np


def _density(pts):
    """Compute a density-like metric for a set of points

    Args:
        pts ([[float]]): Distances for a set of points
    Returns:
        - (float): Density metrics
        - (np.ndarray): Distances between each point
    """
    pts = np.array(pts)
    dists = pdist(pts)
    return len(pts) / dists.max(), squareform(dists)


class MCALSelection(ModelBasedQueryStrategy):
    """The Multiple Criteria Active Learning method for support vector regression

    Uses the methods described by
    `Demir and Bruzzone <https://www.sciencedirect.com/science/article/abs/pii/S0031320314000375>`_
    to select points for evaluation based on:

    1. *Relevancy*: Whether each point is likely to be important in model fitting
    2. *Diversity*: Whether the points are different regions of the search space
    3. *Density*: Whether the points are from regions that contain many other points
    """

    def __init__(self, svm_options: dict = None):
        """Initialize the model

        Args:
            svm_options (dict): Any options for the SVM
        """
        # Make the SVR model
        model = SVR(**(svm_options if svm_options is not None else {}))
        super(MCALSelection, self).__init__(model, fit_model=True)

    def select_points(self, problem: ActiveLearningProblem, n_to_select: int) -> List[int]:
        # Fit the SVR model
        self._fit_model(problem)

        # split training points into support vectors and non support vectors
        train_ixs = np.array(problem.get_labeled_ixs())
        support_mask = np.zeros(len(train_ixs), dtype=bool)
        support_mask[self.model.support_] = True
        train_not_sv_ixs = train_ixs[~support_mask]

        # train clusterer
        # extra arrays and whatnot to track indices into the points array
        # and whether a given points was a training point or not
        clusterer = DBSCAN(eps=1.0)
        unlabeled_ixs = problem.get_unlabeled_ixs()
        clst_ixs = np.concatenate([train_not_sv_ixs, unlabeled_ixs])
        train_mask = np.zeros(clst_ixs.shape, dtype=bool)
        train_mask[:len(train_not_sv_ixs)] = True
        clst_pts = problem.points[clst_ixs]
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

        n_samples = min(n_to_select, len(good_clsts))
        k_densest = sorted(densities, key=lambda x: x[1][0], reverse=True)[:n_samples]

        # sample one point from each of the densest clusters
        selected = []
        for i, (density, dists) in k_densest:
            dists = np.mean(dists, axis=1)
            dense_ix = np.argmin(dists)
            selected.append(clst2pts[i].pop(dense_ix)[2])

        # Randomly select from good clusters, if selection not met
        #  Picking randomly from the list of unlabeled indices to target "density"
        if len(selected) < n_to_select:
            good_ixs = sum([list(map(lambda x: x[2], clst2pts[c])) for c in good_clsts], list())
            unselected_ixs = set(good_ixs).difference(selected)
            if len(unselected_ixs) <= n_to_select - len(selected):
                # Add all to the list
                selected.extend(unselected_ixs)
            else:
                # Add a random subset
                selected.extend(sample(unselected_ixs, n_to_select - len(selected)))

        # Randomly pick points from all the clusters, even the bad ones
        if len(selected) < n_to_select:
            unselected_ixs = set(unlabeled_ixs).difference(selected)
            selected.extend(sample(unselected_ixs, n_to_select - len(selected)))

        return np.array(selected, dtype=int)
