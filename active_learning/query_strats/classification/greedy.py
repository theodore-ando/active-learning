from active_learning.problem import ActiveLearningProblem
from active_learning.query_strats.base import IndividualScoreQueryStrategy, ModelBasedQueryStrategy
from typing import List


class GreedySearch(ModelBasedQueryStrategy, IndividualScoreQueryStrategy):
    """Query strategy where you pick the score most likely to be the target label"""

    def _score_chunk(self, inds: List[int], problem: ActiveLearningProblem):
        probs = self.model.predict_proba(problem.points[inds])
        return probs[:, problem.target_label]
