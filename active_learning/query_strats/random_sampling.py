from ..problem import ActiveLearningProblem
from . import IndividualScoreQueryStrategy
from random import random
from typing import List


class RandomQuery(IndividualScoreQueryStrategy):
    """Randomly select entries from the unlabeled set"""

    def _score_chunk(self, inds: List[int], problem: ActiveLearningProblem):
        return [random() for _ in inds]
