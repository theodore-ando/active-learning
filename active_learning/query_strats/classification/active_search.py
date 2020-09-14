from sklearn.base import BaseEstimator, clone
from active_learning.problem import ActiveLearningProblem
from active_learning.query_strats.base import IndividualScoreQueryStrategy, ModelBasedQueryStrategy
from typing import List
import numpy as np


def _lookahead(points: np.ndarray, model: BaseEstimator,
               train_ixs: List[int], obs_labels: List[float],
               x: np.ndarray, label: float):
    """
    Does a lookahead at what the model would be if (x, label) were added to the
    known set.  If the model implements the partial_fit API from sklearn, then
    that will be used.  Otherwise, the model is retrained from scratch

    Args:
        model (BaseEstimator): sklearn model to be retrained
        train_ixs (ndarray): Indices of currently-labeled set
        obs_labels (ndarray): Labels for each labeled entry
        x (ndarray): Data point to simulate being labeled
        label (float): Simulated label
    """
    # If partial-fit available, use it
    if hasattr(model, "partial_fit"):
        return model.partial_fit([x], [label], [0, 1])

    # Update the training set
    X_train = np.concatenate([points[train_ixs], [x]])
    obs_labels = np.concatenate([obs_labels, [label]])

    # Refit the model
    model.fit(X_train, obs_labels)


def _split_lookahead(problem, points_and_models, train_ixs, obs_labels):
    """
    """
    return [
        (_lookahead(problem['points'], models[0], train_ixs, obs_labels, x, 0),
         _lookahead(problem['points'], models[1], train_ixs, obs_labels, x, 1))
        for x, models in points_and_models
    ]


def _expected_future_utility(model: BaseEstimator, test_set: np.ndarray,
                             budget: int, target_label: int):
    """
    The expected future utility of all remaining points is the sum top `budget`
    number of probabilities that the model predicts on the test set.  This is
    assuming that the utility function is the number of targets found, and that
    we can only make `budget` queries.

    Args:
        model (BaseEstimator): Model trained on training set + potential new point
        test_set (ndarray): Test set for the model
        budget (int): number of points that we will be able to query
        target_label (int): Index of target label

    Returns:
        (float) Expected utility
    """

    # Predict the probability of each entry in the test set
    probs = model.predict_proba(test_set)
    positives = probs[:, target_label]

    # sum only the top `budget` probabilities!  Even if there are more, we can
    # only possibly gain `budget` more targets.
    klargest = positives.argpartition(-budget)[-budget:]
    u = np.sum(positives[klargest])

    return u


class ActiveSearch(ModelBasedQueryStrategy, IndividualScoreQueryStrategy):
    """Efficient Non-Myopic Active Search.

    Based on an algorithm by
    `Jiang et al. <http://proceedings.mlr.press/v70/jiang17d/jiang17d.pdf>`_.
    Automatically balances between the desire to greedily query points
    highly likely to be the target class and those which, if queried,
    will lead to model improvements that will lead to more
    targets to be found later on."""

    def _score_chunk(self, inds: List[int], problem: ActiveLearningProblem):
        # Get a list of *all* unlabeled entries
        all_unlabeled_ixs = np.array(problem.get_unlabeled_ixs())

        # Get the probabilities for each class
        probs = self.model.predict_proba(problem.points[all_unlabeled_ixs])

        # Get scores for this chunk
        chunk_scores = []
        for ix, unlabeled_ix in enumerate(inds):
            # Make the new test set
            test_set = problem.points[all_unlabeled_ixs[all_unlabeled_ixs != unlabeled_ix]]

            # Make a copy of the model to use in lookahead
            model = clone(self.model)

            # Get the probabilities for this point
            my_probs = probs[ix, :]

            # If the budget is only for one more label, the score is just the probability
            if problem.budget - 1 == 0:
                chunk_scores.append(my_probs[problem.target_label])
            else:
                # If not, assess the effect of labeling this point

                # Get the expected utility of the entry being labeled each class
                expected_util = []
                for i in range(my_probs.shape[0]):
                    _lookahead(problem.points, model, problem.labeled_ixs,
                               problem.labels, problem.points[unlabeled_ix],
                               label=i)
                    expected_util.append(_expected_future_utility(self.model, test_set,
                                                                  problem.budget - 1,
                                                                  problem.target_label))

                # Compute the score for this point
                #  This is equal to the probability of it being positive (p1)
                #  plus the expected number of positives found in the
                #  "budget - 1" remaining entries if this point is labeled,
                #  which is equal to the product of the probability of "1" or "0"
                #  times the number of positives in the top budget-1 if "1" or "0"
                chunk_scores.append(my_probs[problem.target_label] +
                                    np.dot(my_probs, expected_util))

        return chunk_scores
