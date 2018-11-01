from active_learning.problem import ActiveLearningProblem
from active_learning.query_strats import IndividualScoreQueryStrategy
from typing import List
import numpy as np


class UncertaintySampling(IndividualScoreQueryStrategy):
    """Sample entries with the highest uncertainty in the classification score"""

    def _score_chunk(self, inds: List[int], problem: ActiveLearningProblem):
        probs = problem.model.predict_proba(problem.points[inds])
        probs = probs.clip(1e-9, 1 - 1e-9)
        return -1 * (probs * np.log(probs)).sum(axis=1)
