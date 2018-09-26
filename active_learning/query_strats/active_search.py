from multiprocessing import cpu_count

from sklearn.externals.joblib import Parallel, delayed
import numpy as np
from sklearn.base import clone

from active_learning.utils import chunks
from . import argmax

"""
The general strategy in this file is taken from the paper 
http://proceedings.mlr.press/v70/jiang17d/jiang17d.pdf.
It tries to make an efficient, non-myopic approximation for the future utility
of every unlabeled point.
"""


def _lookahead(points, model, train_ixs, obs_labels, x, label):
    """
    Does a lookahead at what the model would be if (x, label) were added to the
    known set.  If the model implements the partial_fit API from sklearn, then
    that will be used.  Otherwise, the model is retrained from scratch on 
    problem['points'] + x.
    :param problem: dictionary that defines the problem, containing keys:
        * points:       an (n_samples, n_dim) matrix of points in the space
        * num_classes:  the number of different classes [0, num_classes)
        * batch_size:   number of points to query each iteration
        * num_queries:  the max number of queries we can make on the data
        * model:        the sk-learn model we are training 
    :param model: sklearn model to be trained.  This is likely a copy of the
        model in the problem dictionary because many copies of model trained on
        different potential new points to be queried.
    :param train_ixs: index into `points` of the training examples
    :param obs_labels: labels for the training examples
    :param x: the data point in question to lookahead at
    :param label: potential label for x (not the real one!)
    :return: a newly trained model
    """
    if hasattr(model, "partial_fit"):
        return model.partial_fit([x], [label], [0, 1])

    X_train = np.concatenate([points[train_ixs], [x]])
    obs_labels = np.concatenate([obs_labels, [label]])
    return model.fit(X_train, obs_labels)


def _split_lookahead(problem, points_and_models, train_ixs, obs_labels):
    """
    """
    return [
        (_lookahead(problem['points'], models[0], train_ixs, obs_labels, x, 0),
         _lookahead(problem['points'], models[1], train_ixs, obs_labels, x, 1))
        for x, models in points_and_models
    ]


def _expected_future_utility(model, points, test_set, budget):
    """
    The expected future utility of all remaining points is the sum top `budget`
    number of probabilities that the model predicts on the test set.  This is
    assuming that the utility function is the number of targets found, and that
    we can only make `budget` queries.
    :param model: model trained on training set + potential new point
    :param points: all points in the space
    :param test_set: set of unlabeled points - potential new point
    :param budget: number of points that we will be able to query
    :return: utility
    """
    probs = model.predict_proba(points[test_set])
    positives = probs[:,1]

    # sum only the top `budget` probabilities!  Even if there are more, we can
    # only possibly gain `budget` more targets.
    klargest = positives.argpartition(-budget)[-budget:]
    u = np.sum(positives[klargest])
    return u


def _split_future_utility(models, points, test_set, budget):
    """
    dumb utility function to evaluate _expected_future_utility on two models.
    The two models come from the two possible labels {0, 1} for a potential new
    point, and this function is useful for parallelizing the evaluation.
    """
    return (
        _expected_future_utility(models[0], points, test_set, budget),
        _expected_future_utility(models[1], points, test_set, budget)
    )


def _mem_saver(model, points, train_ixs, obs_labels, unlabeled_chunk, all_unlabeled_ixs, budget):
    model_copy = clone(model)
    chunk_scores = []
    for unlabeled_ix in unlabeled_chunk:
        # Make the new test set
        test_set = np.delete(all_unlabeled_ixs, np.argwhere(all_unlabeled_ixs == unlabeled_ix))

        # Get the probabilities of membership in either class
        p0, p1 = model.predict_proba(points[[unlabeled_ix]]).reshape(-1)

        # If the budget is only for one more label, the score is just the probability
        if budget - 1 == 0:
            chunk_scores.append(p1)
        else:
            # If not, assess the effect of labeling this point

            # Evaluate the effect of the point being labeled negative
            m0 = _lookahead(points, model_copy, train_ixs, obs_labels, points[unlabeled_ix], label=0)
            s0 = _expected_future_utility(m0, points, test_set, budget-1)

            # Evaluate the effect of the point being labeled positive
            m1 = _lookahead(points, model_copy, train_ixs, obs_labels, points[unlabeled_ix], label=1)
            s1 = _expected_future_utility(m1, points, test_set, budget-1)

            # Compute the score for this point
            #  This is equal to the probability of it being positive (p1) 
            #  plus the expected number of positives found in the 
            #  "budget - 1" remaining entries if this point is labeled,
            #  which is equal to the product of the probability of "1" or "0"
            #  times the number of positives in the top budget-1 if "1" or "0"
            chunk_scores.append(p1 + (p0 * s0 + p1 * s1))
        del test_set

    return np.array(chunk_scores)


def _search_score(problem, train_ixs, obs_labels, unlabeled_ixs, batch_size, **kwargs):
    model = problem['model']
    points = problem['points']

    # num queries remaining
    budget = kwargs['budget']
    assert budget > 0

    # OS X requires you to use "threading" rather than "multiprocessing"
    # because it doesn't support BLAS calls on both 'sides' of a fork
    # however, we cannot just use threading because RandomForest is not thread safe...
    backend = problem.get("parallel_backend", "threading")

    n_cpu = cpu_count()
    n_chunks = n_cpu * 100

    with Parallel(n_jobs=n_cpu, max_nbytes=1e6, backend=backend, verbose=2) as parallel:
        real_budget = min(budget * batch_size, len(unlabeled_ixs)-1)
        expected_future_utilities = parallel(
            delayed(_mem_saver)(model, points, train_ixs, obs_labels, chunk, unlabeled_ixs, real_budget)
            for chunk in chunks(unlabeled_ixs, n_chunks)
        )

        expected_future_utilities = np.concatenate(expected_future_utilities)

    # print(expected_future_utilities)
    return expected_future_utilities


def active_search(problem, train_ixs, obs_labels, unlabeled_ixs, npoints, **kwargs):
    score = _search_score

    return argmax(problem, train_ixs, obs_labels, unlabeled_ixs, score, npoints, **kwargs)

