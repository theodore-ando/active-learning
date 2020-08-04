from ...problem import ActiveLearningProblem
from active_learning.query_strats.base import IndividualScoreQueryStrategy, ModelBasedQueryStrategy
from typing import List


class GreedySelection(ModelBasedQueryStrategy, IndividualScoreQueryStrategy):
    """Select the points with this highest predicted values of the output function"""

    def select_points(self, problem: ActiveLearningProblem, n_to_select: int):
        if self.fit_model:
            self.model.fit(*problem.get_labeled_points())
        return super().select_points(problem, n_to_select)

    def _score_chunk(self, inds: List[int], problem: ActiveLearningProblem):
        return -1 * problem.objective_fun.score(self.model.predict(problem.points[inds]))
