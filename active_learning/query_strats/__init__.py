from active_learning.problem import ActiveLearningProblem
from multiprocessing import Pool
from functools import partial
from typing import List
import numpy as np


class BaseQueryStrategy:
    """Base class for all active learning strategies."""

    def select_points(self, problem: ActiveLearningProblem, n_to_select: int) -> List[int]:
        """Identify which points should be queried next

        Args:
            problem (ActiveLearningProblem): Problem definition
            n_to_select (int): Number of points to select
        Returns:
            ([int]): List of which points to query next
        """
        raise NotImplementedError()


class IndividualScoreQueryStrategy(BaseQueryStrategy):
    """"Base class for query strategies that rate each point independently"""

    def __init__(self, n_cpus: int = 1, chunks_per_thread: int = 32):
        """Initialize the query strategy

        Args:
            n_cpus (int): Number of processors to use
            chunks_per_thread (int): Number of chunks of indices per thread when multiprocessing
        """
        self.n_cpus = n_cpus
        self.chunks_per_thread = chunks_per_thread

    def score(self, ind: int, problem: ActiveLearningProblem) -> float:
        """Score a single entry

        Entries with higher scores should be selected

        Args:
            ind (int): Index of entry to be assessed
            problem (ActiveLearningProblem): Active learning problem definition
        Returns:
            (float) Score of entry
        """
        raise NotImplementedError()

    def _score_many(self, inds: List[int], problem: ActiveLearningProblem) -> List[float]:
        """Score a list of indices

        Used to prevent copying the active learning problem definition many times

        Args:
            inds ([int]): List of indices to score
            problem (ActiveLearningProblem): Active learning problem definition
        Returns:
            ([float]) Scores for each entry
        """

        return [self.score(i, problem) for i in inds]

    def select_points(self, problem: ActiveLearningProblem, n_to_select: int):
        # Get the unlabeled indices as an ndarray (for easy indexing)
        unlabled_ixs = np.array(problem.get_unlabeled_ixs())

        # Get the scores
        if self.n_cpus == 1:
            scores = self._score_many(unlabled_ixs, problem)
        else:
            # Make the chunks of indices to evaluate in parallel
            chunks = np.array_split(unlabled_ixs, self.n_cpus * self.chunks_per_thread)

            # Make the function to be evaluated
            fun = partial(self._score_many, problem=problem)

            # Score each chunk
            with Pool(self.n_cpus) as p:
                scores = p.map(fun, chunks)

            # De-chunk the scores
            scores = np.concatenate(scores)

        # Get the top entries
        return unlabled_ixs[np.argpartition(scores, -n_to_select)[-n_to_select:]].tolist()
