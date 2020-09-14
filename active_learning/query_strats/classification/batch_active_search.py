from active_learning.problem import ActiveLearningProblem
from active_learning.query_strats.base import ModelBasedQueryStrategy, BaseQueryStrategy
from sklearn.base import BaseEstimator
from copy import deepcopy
from typing import Union
import numpy as np


"""
https://bayesopt.github.io/papers/2017/12.pdf
"""

# -----------------------------------------------------------------------------
#                   Fictional Oracles to Simulate Sequential
# -----------------------------------------------------------------------------

# TODO: Add oracles for regression problems


class FictionalOracle:
    """Class to emulate a labeling function"""

    def label(self, model: BaseEstimator, x: np.array, target: int):
        """Assign a label to a certain data point

        Args:
            model (BaseEstimator): Model trained on all labeled points
            x (ndarray): Point to be "labeled"
            target (int): Index of the target class
        Returns:
            (int) Label for this point
        """
        raise NotImplementedError()


class SamplingOracle(FictionalOracle):
    """Pick a random label weighted by the predicted probabilities from the model"""
    def label(self, model: BaseEstimator, x: np.array, target: int):
        probs = model.predict_proba([x])
        probs = probs.reshape(2)
        return np.random.binomial(1, probs[1])


class MostLikelyOracle(FictionalOracle):
    """Return the most likely label for x according to model, i.e.: argmax p(y|x)"""
    def label(self, model: BaseEstimator, x: np.array, target: int):
        probs = model.predict_proba([x])
        probs = probs.reshape(2)
        raise np.argmax(probs)


class PessimisticOracle(FictionalOracle):
    """Assume the prediction is not the target class.
     Assumes that the model binary classification"""
    def label(self, model: BaseEstimator, x: np.array, target: int):
        return 1 - target


class OptimisticOracle(FictionalOracle):
    """Assume that the prediction is the target class"""
    def label(self, model: BaseEstimator, x: np.array, target: int):
        return target


_FICTIONAL_ORACLES = {
    "sampling": SamplingOracle(),
    "most_likely": MostLikelyOracle(),
    "pessimistic": PessimisticOracle(),
    "optimistic": OptimisticOracle()
}


class SequentialSimulatedBatchSearch(ModelBasedQueryStrategy):
    """Batch active learning strategy where you simulate multiple, sequential steps of an
    active learning process.

    TBD: Better description after reading the paper"""

    def __init__(self, model: BaseEstimator, query_strategy: BaseQueryStrategy,
                 fictional_oracle: Union[str, FictionalOracle], fit_model: bool = True):
        """Initialize strategy

        Args:
            model (BaseEstimator): Model used to guide the search
            fit_model (bool): Whether to fit the model before querying
            query_strategy (BaseQueryStrategy): Strategy to perform sequential selection
            fictional_oracle (string): Function used to emulate labeling
        """
        super().__init__(model=model, fit_model=fit_model)
        self.query_strategy = query_strategy
        if isinstance(fictional_oracle, str):
            self.fictional_oracle = _FICTIONAL_ORACLES[fictional_oracle]
        else:
            self.fictional_oracle = fictional_oracle

    def select_points(self, problem: ActiveLearningProblem, n_to_select: int):
        # Make a copy of the active learning problem
        #  TODO: Copying the entire problem could be costly. Could we just copy model/labels? -lw
        problem = deepcopy(problem)

        # Start accumulation for the batch
        batch_ixs = []

        for _ in range(n_to_select - 1):
            # Select a single point
            x = self.query_strategy.select_points(problem, 1)

            # since we requested only one point, get value of singleton
            x = x[0]
            batch_ixs.append(x)

            # Query the fictional oracle
            y = self.fictional_oracle.label(self.model, problem.points[x], problem.target_label)

            # Update the active learning problem
            problem.add_label(x, y)
            self._fit_model(problem)

            # Decrement the budget
            problem.budget -= 1

        # Select a single point
        x = self.query_strategy.select_points(problem, 1)

        # since we requested only one point, get value of singleton
        x = x[0]
        batch_ixs.append(x)

        return batch_ixs
