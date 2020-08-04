"""Bayesian active learning methods"""
from inspect import signature
from typing import List

import numpy as np
from scipy.stats import norm

from active_learning.problem import ActiveLearningProblem
from active_learning.query_strats.base import IndividualScoreQueryStrategy


# Following: http://krasserm.github.io/2018/03/21/bayesian-optimization/

class ExpectedImprovement(IndividualScoreQueryStrategy):
    """Bayesian 'Expected Improvement' active learning

    Determines which points have the largest expected improvement over
    the best labeled point to date.

    Each point is assigned a value equal to the expected/mean improvement
    of that point's value over a threshold.
    """

    def __init__(self, model, refit_model: bool = True, epsilon: float = 0):
        """
        Args:
            model: Scikit-learn model used to make inferences
        """
        super().__init__()
        self.model = model
        self.refit_model = refit_model
        self.epsilon = epsilon

        # Check if the function supports "return_std"
        if 'return_std' not in signature(self.model.predict).parameters:
            raise ValueError('The model must have "return_std" in the predict methods')

    def select_points(self, problem: ActiveLearningProblem, n_to_select: int):
        if self.refit_model:
            self.model.fit(*problem.get_labeled_points())
        return super().select_points(problem, n_to_select)

    def _score_chunk(self, inds: List[int], problem: ActiveLearningProblem):
        y_mean, y_std = self.model.predict(problem.points[inds], return_std=True)

        # Compute the EI
        # TODO (wardlt): Support minimization
        _, known_labels = problem.get_labeled_points()
        threshold = np.max(known_labels)  # f(x^+) in the
        z_score = (y_mean - threshold - self.epsilon) / y_std

        ei = (y_mean - threshold - self.epsilon) * norm.cdf(z_score) + y_std * norm.pdf(z_score)

        return ei

