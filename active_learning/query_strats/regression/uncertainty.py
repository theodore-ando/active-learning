from active_learning.query_strats.base import IndividualScoreQueryStrategy, ModelBasedQueryStrategy
from active_learning.problem import ActiveLearningProblem
from inspect import signature
from typing import List


class UncertaintySampling(ModelBasedQueryStrategy, IndividualScoreQueryStrategy):
    """Select the entries with the largest uncertainty"""

    def __init__(self, model, refit_model: bool = True):
        super().__init__(model, refit_model)

        # Check if the function supports "return_std"
        if 'return_std' not in signature(self.model.predict).parameters:
            raise ValueError('The model must have "return_std" in the predict methods')

    def select_points(self, problem: ActiveLearningProblem, n_to_select: int):
        if self.fit_model:
            self.model.fit(*problem.get_labeled_points())
        return super().select_points(problem, n_to_select)

    def _score_chunk(self, inds: List[int], problem: ActiveLearningProblem):
        _, y_std = self.model.predict(problem.points[inds], return_std=True)
        return y_std
