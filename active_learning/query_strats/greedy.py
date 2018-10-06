from active_learning.problem import ActiveLearningProblem
from . import IndividualScoreQueryStrategy
from typing import List


class GreedySearch(IndividualScoreQueryStrategy):
    """Query strategy where you pick the score most likely to be the target label"""

    def _score_chunk(self, inds: List[int], problem: ActiveLearningProblem):
        probs = problem.model.predict_proba(problem.points[inds])
        return probs[:, problem.positive_label]
