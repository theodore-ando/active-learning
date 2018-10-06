from ..problem import ActiveLearningProblem
from . import IndividualScoreQueryStrategy
from random import random


class RandomQuery(IndividualScoreQueryStrategy):
    """Randomly select entries from the unlabeled set"""

    def score(self, ind: int, problem: ActiveLearningProblem):
        return random()
