from ...problem import ActiveLearningProblem
from .. import ModelBasedQueryStrategy
from typing import List
import numpy as np


class GreedySelection(ModelBasedQueryStrategy):
    """Select the points with this highest predicted values of the output function"""

    def select_points(self, problem: ActiveLearningProblem, n_to_select: int) -> List[int]:
        super()._fit_model(problem)
        unlabeled_ixs = np.array(problem.get_unlabeled_ixs())
        scores = problem.objective_fun.score(self.model.predict(problem.points[unlabeled_ixs]))
        return unlabeled_ixs[np.argpartition(scores, n_to_select)][:n_to_select]
