import logging
import numpy as np
from collections import Iterable
from tqdm import tqdm

from active_learning.query_strats import uncertainty_sampling
from active_learning.selectors import identity_selector


def _actively_learn(problem, train_ixs, obs_labels, oracle,
                    selector=identity_selector,
                    query_strat=uncertainty_sampling,
                    callback=None, **kwargs):
    """
    The main active learning loop
    :param problem: dictionary that defines the problem, containing keys:
        * points:       an (n_samples, n_dim) matrix of points in the space
        * num_classes:  the number of different classes [0, num_classes)
        * batch_size:   number of points to query each iteration
        * num_queries:  the max number of queries we can make on the data
        * model:        the sk-learn model we are training
        Optional:
        - partition: list of np.arrays of indices into problem['points'] partitioning the space.
            This can restrict the batch to be from one partition!
    :param train_ixs: index into `points` of the training examples
    :param obs_labels: labels for the training examples
    :param oracle: gets the true labels for a set of points 
    :param selector: gets the indexes of all available points to be tested
    :param query_strat: gets the indexes of points we wish to test next 
    :param callback: function or list of funcs to call at end of each iteration (retrain the model?)
    :return: None (yet)
    """

    problem['num_initial'] = len(train_ixs)
    num_queries = problem['num_queries']
    batch_size = problem['batch_size']

    for i in tqdm(range(num_queries)):
        print(f"Query {i} / {num_queries}")

        # get the available points we might want to query
        unlabeled_ixs = selector(problem, train_ixs, obs_labels, **kwargs)
        logging.debug(f"{len(unlabeled_ixs)} available unlabeled points")

        if len(unlabeled_ixs) == 0:
            logging.debug("No available unlabeled points")
            return

        # choose points from the available points
        selected_ixs = query_strat(problem, train_ixs, obs_labels, unlabeled_ixs, batch_size, budget=num_queries-i, **kwargs)

        # get the true labels from the oracle
        true_labels = oracle(problem, train_ixs, obs_labels, selected_ixs, **kwargs)
        logging.debug(selected_ixs)
        logging.debug(true_labels)

        # add the new labeled points to the training set
        train_ixs = np.concatenate([train_ixs, selected_ixs])
        obs_labels = np.concatenate([obs_labels, true_labels])

        # presumably the call back will update the model
        if callback is None:
            continue
        elif isinstance(callback, Iterable):
            for func in callback:
                func(problem, train_ixs, obs_labels, query=selected_ixs, **kwargs)
        else:
            callback(problem, train_ixs, obs_labels, query=selected_ixs, **kwargs)


# class ActiveLearner(BaseEstimator, ClassifierMixin):
#     def __init__(self, batch_size=1, num_queries=0.1, L_size=2):
