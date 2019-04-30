from active_learning.query_strats.base import ModelBasedQueryStrategy
from active_learning.problem import ActiveLearningProblem
from sklearn.base import BaseEstimator
from sklearn.mixture.base import BaseMixture
from sklearn.mixture import GaussianMixture
from typing import List
import numpy as np


def _calc_diversities(gmm: BaseMixture, points: np.ndarray,
                      S: List[int], X_test: np.ndarray):
    """Compute the diversity of points in the unlabeled set

    Args:
        gmm (BaseMixture): Gaussian mixture model trained on the labeled points
        points (ndarray): All points in the problem space
        S ([int]): Points that have already been selected in this batch
        X_test ([int]): Possible points to include in the
    Returns:
        (np.ndarray) Diversity score for each point
    """
    S_scores = gmm.score(points[S])
    S_score = np.sum(S_scores)

    # TODO (lw): The state of S does not change the point with highest score?
    log_probs = gmm.score_samples(X_test)
    scores = -(log_probs + S_score) / (len(S) + 1)

    return scores


class ThreeDs(ModelBasedQueryStrategy):
    """Select points based on distance, density, and diversity.

    Based on work by `Reitmaier and Sick <http://ieeexplore.ieee.org/document/5949421/>`_.

    Distance is based on how far a point is from the decision boundary. The distance is computed
    based on the ratio between the probability of the 1st and 2nd-most likely classes. Points
    farther from the boundary are more distant.

    Density is related to whether the points are in regions with many points in the search space.
    The density is determined by the probability an entry is from the search space given a Gaussian
    mixture model. Points with a higher density are desirable.

    Diversity is achieved by selecting a set of points with large differences between each other.
    The algorithm assumes the distribution of point in the set is equal to the search space,
    so the effect of this factor is to pick points in low-density regions.
    """

    def __init__(self, model: BaseEstimator, dwc: float = 0.5,
                 gmm: BaseMixture = GaussianMixture()):
        """Initialize the strategy

        Args:
            model (BaseEstimator): Model used to generate the "distance" metric
            dwc (float): Diversity weighting coefficient (larger to weigh diversity more)
            gmm (BaseMixture): Strategy used to model the distribution of the search space
        """
        super().__init__(model)
        self.dwc = dwc
        self.gmm = gmm

    def select_points(self, problem: ActiveLearningProblem, n_to_select: int):
        # Get the model and search space
        points = problem.points

        # Fit the model on the new problem
        self._fit_model(problem)

        # Get the unlabeled indices and points
        U = problem.get_unlabeled_ixs()
        X_test = points[U]

        # calculate weighting factor
        probs = self.model.predict_proba(X_test)
        eps = 1.0 / len(probs) * np.sum(1 - np.argmax(probs, axis=1))

        # pseudo-distance that works on models that do not have a real decision boundary
        probs = np.apply_along_axis(np.sort, 1, probs)
        c1 = probs[:, -1]
        c2 = probs[:, -2]
        distances = np.log(c1) - np.log(c2)  # Only work for binary classification

        # density function via mixture model
        #  The paper describes traininng the model with initial unlabled set and a small labeled set
        #  As we do not assume the unlabled set is the initial one or that the labeled set is small,
        #  I use the entire search space to train a mixture model.
        self.gmm.fit(points)
        densities = np.exp(self.gmm.score_samples(X_test))

        # select the first point
        x = np.argmax((1-eps) * (1-distances) + eps * densities)
        S = [U.pop(x)]

        # Update the tests
        X_test = np.delete(X_test, x, 0)
        distances = np.delete(distances, x, 0)
        densities = np.delete(densities, x, 0)

        # Get the weighting factors
        alpha = (1 - self.dwc) * (1 - eps)  # coefficient normalization
        beta = (1 - self.dwc) - alpha       # coefficient normalization

        # Pick points considering density
        while len(S) < n_to_select:
            # Generate the densities of the test set
            diversities = _calc_diversities(self.gmm, points, S, X_test)

            # Select the maximum point
            x = np.argmax(alpha*(1-distances) + beta*densities + self.dwc*diversities)
            S.append(U.pop(x))

            # Update the tests
            X_test = np.delete(X_test, x, 0)
            distances = np.delete(distances, x, 0)
            densities = np.delete(densities, x, 0)

        return S
