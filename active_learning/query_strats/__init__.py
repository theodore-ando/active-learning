from active_learning.problem import ActiveLearningProblem
from multiprocessing import Pool
from functools import partial
from typing import List, Tuple
import numpy as np
from sklearn.base import BaseEstimator


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
    """"Base class for query strategies that rate each point independently

    Queries are determined by evaluating the score of batches of entries inpendently,
    which allows for parallelization.
    """

    def __init__(self, n_cpus: int = 1, chunks_per_thread: int = 32):
        """Initialize the query strategy

        Args:
            n_cpus (int): Number of processors to use
            chunks_per_thread (int): Number of chunks of indices per thread when multiprocessing
        """
        self.n_cpus = n_cpus
        self.chunks_per_thread = chunks_per_thread

    def _score_chunk(self, inds: List[int], problem: ActiveLearningProblem) -> List[float]:
        """Score a list of indices

        Used to prevent copying the active learning problem definition many times

        Args:
            inds ([int]): List of indices to score
            problem (ActiveLearningProblem): Active learning problem definition
        Returns:
            ([float]) Scores for each entry
        """

        raise NotImplementedError()

    def select_points(self, problem: ActiveLearningProblem, n_to_select: int):
        # Get the unlabeled indices as an ndarray (for easy indexing)
        unlabled_ixs, scores = self.score_all(problem)

        # Get the top entries
        return unlabled_ixs[np.argpartition(scores, -n_to_select)[-n_to_select:]]

    def score_all(self, problem: ActiveLearningProblem) -> Tuple[List[int], List[float]]:
        """Determine the scores for all non-labeled points

        Args:
            problem (ActiveLearningProblem): Active learning problem definition
        Return:
            - [int] Indices of all unlabeled points
            - [float] Scores for each point
        """
        # Get the unlabeled indices as an ndarray (for easy indexing)
        unlabled_ixs = np.array(problem.get_unlabeled_ixs())

        # Get the scores
        if self.n_cpus == 1:
            scores = self._score_chunk(unlabled_ixs, problem)
        else:
            # Make the chunks of indices to evaluate in parallel
            chunks = np.array_split(unlabled_ixs, self.n_cpus * self.chunks_per_thread)

            # Make the function to be evaluated
            fun = partial(self._score_chunk, problem=problem)

            # Score each chunk
            with Pool(self.n_cpus) as p:
                scores = p.map(fun, chunks)

            # De-chunk the scores
            scores = np.concatenate(scores)

        return unlabled_ixs, scores


class ModelBasedQueryStrategy(BaseQueryStrategy):
    """Mixin for query strategies that use an model to make predictions

    Model objects must satisfy the scikit-learn API"""

    def __init__(self, model: BaseEstimator, model_is_fitted: bool = False, **kwargs):
        """
        Args:
            model (BaseEstimator): Model to use for querying
            model_is_fitted (bool): Whether the model has been fitted already
        """
        super().__init__(**kwargs)
        self.model = model
        self._model_is_fitted = model_is_fitted

    def _fit_model(self, problem: ActiveLearningProblem):
        """Fit the model on the current active learning problem

        Args:
              problem (ActiveLearningProblem): Description of the active learning problem
        """
        X, y = problem.get_labeled_points()
        self.model.fit(X, y)
        self._model_is_fitted = True
