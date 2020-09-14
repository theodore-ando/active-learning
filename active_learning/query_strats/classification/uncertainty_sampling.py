from active_learning.problem import ActiveLearningProblem
from active_learning.query_strats.base import IndividualScoreQueryStrategy, ModelBasedQueryStrategy
from typing import List
import numpy as np


class UncertaintySampling(ModelBasedQueryStrategy, IndividualScoreQueryStrategy):
    """Sample entries with the highest uncertainty in the classification score"""

    def _score_chunk(self, inds: List[int], problem: ActiveLearningProblem):
        probs = self.model.predict_proba(problem.points[inds])
        probs = probs.clip(1e-9, 1 - 1e-9)
        return -1 * (probs * np.log(probs)).sum(axis=1)
