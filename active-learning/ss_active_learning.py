import logging
import numpy as np

#TODO: make this query a kNN sort of model or maybe clustering for predicting how likely
from collections import Iterable

from sklearn.svm import SVC

from active_learning.query_strats import uncertainty_sampling
from active_learning.selectors import identity_selector


def _standardize_is_ssl_round(is_ssl_round):
    if not callable(is_ssl_round):
        if isinstance(is_ssl_round, int):
            return lambda i, n: i % is_ssl_round == 0 and i != 0
        else:
            raise TypeError("SSL_every key in `problem` dict must be callable or int")
    return is_ssl_round


def _default_ssl_config():
    return {
        "is_ssl_round": (lambda i, n: i >= n // 2 and i % 3 == 0),
        "clear_label_history": (lambda i, n: i >= n // 2 and i % 6 == 0)
    }


def _ss_labeler(problem, train_ixs, obs_labels, unlabeled_ixs, **kwargs):
    label_history = kwargs['label_history']
    label_change_rate = np.sum(label_history, axis=0)

    pts = problem['points']
    X_train = pts[train_ixs]

    #     clf = KNeighborsClassifier(n_neighbors=4)
    clf = SVC(probability=True)
    clf.fit(X_train, obs_labels)

    ixs = np.arange(len(problem['points']))
    train_mask = np.zeros(len(ixs), dtype=bool)
    train_mask[train_ixs] = True
    # only choose among points with label change rate = 0
    test_ixs = ixs[(~train_mask) & (label_change_rate == 0)]

    if len(test_ixs) == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)

    pred_labels = clf.predict(pts[test_ixs])
    pos_ixs = test_ixs[pred_labels == 1]
    neg_ixs = test_ixs[pred_labels == 0]

    if len(pos_ixs) > 0:
        pos_distances = clf.decision_function(pts[pos_ixs]).reshape(-1)
        #         log_probs = np.log(clf.predict_proba(pts[pos_ixs]))
        #         pos_distances = log_probs.max(axis=1) - log_probs.min(axis=1)
        median_pos_ix = pos_ixs[np.argsort(pos_distances)[-1]]  # [len(pos_ixs)//2]]
        pos_ixs = np.array([median_pos_ix], dtype=int)
    else:
        pos_ixs = np.array([], dtype=int)

    if len(neg_ixs) > 0:
        neg_distances = clf.decision_function(pts[neg_ixs]).reshape(-1)
        #         log_probs = np.log(clf.predict_proba(pts[neg_ixs]))
        #         neg_distances = log_probs.max(axis=1) - log_probs.min(axis=1)
        median_neg_ix = neg_ixs[np.argsort(neg_distances)[-1]]  # [len(neg_ixs)//2]]
        neg_ixs = np.array([median_neg_ix], dtype=int)
    else:
        neg_ixs = np.array([], dtype=int)

    return pos_ixs, neg_ixs


def ss_actively_learn(problem, train_ixs, obs_labels, oracle,
                      query_strat=uncertainty_sampling,
                      ss_labeler=_ss_labeler,
                      callback=None, **kwargs):
    """
    A variation on active learning that will use a Semi-supervised influenced approach to avoid
    oversampling the dense region.
    :param problem: dictionary that defines the problem, containing keys:
        * points:       an (n_samples, n_dim) matrix of points in the space
        * num_classes:  the number of different classes [0, num_classes)
        * batch_size:   number of points to query each iteration
        * num_queries:  the max number of queries we can make on the data
        * model:        the sk-learn model we are training
        Optional:

        - partition: list of np.arrays of indices into problem['points'] partitioning the space.
            This can restrict the batch to be from one partition!
        - is_ssl_round: `int` or callable.  If `int` then `ss_labeler` will be called every `is_ssl_round` iterations
            except for the first.  If callable, will be called as `is_ssl_round(query_num, num_queries)`.
        - clear_label_history: `int` or callable.  If `int` then label history will be cleared every x iterations
            except for the first.  If callable, will be called as `clear_label_history(query_num, num_queries)`.
    :param train_ixs: index into `points` of the training examples
    :param obs_labels: labels for the training examples
    :param oracle: gets the true labels for a set of points
    :param selector: gets the indexes of all available points to be tested
    :param query_strat: gets the indexes of points we wish to test next
    :param ss_labeler: semi-supervised labeler, returns (list of pos_ixs, list neg_ixs)
    :param callback: function or list of funcs to call at end of each iteration (retrain the model?)
    :return: None (yet)
    """

    problem['num_initial'] = len(train_ixs)
    num_queries = problem['num_queries']
    batch_size = problem['batch_size']

    # function to check whether ss_labeler should be queried in a given round
    is_ssl_round = _standardize_is_ssl_round(problem.get('is_ssl_round', 3))

    # function to check whether label_history should be cleared, default to every ssl_round
    clear_label_history = _standardize_is_ssl_round(problem.get('clear_label_history', 3))

    selector = kwargs.get("selector", identity_selector)

    # in order to track label change rate, store what each point would be labeled with
    label_history = []

    for i in range(num_queries):
        print(f"Query {i} / {num_queries}")

        # get the available points we might want to query
        unlabeled_ixs = selector(problem, train_ixs, obs_labels, **kwargs)
        logging.debug(f"{len(unlabeled_ixs)} available unlabeled points")

        # to track label change rate
        label_history.append(problem['model'].predict(problem['points']))

        if len(unlabeled_ixs) == 0:
            logging.debug("No available unlabeled points")
            return

        # ------------------
        # choose points from the available points
        selected_ixs = query_strat(problem, train_ixs, obs_labels, unlabeled_ixs, batch_size,
                                   budget=num_queries - i, **kwargs)

        # get the true labels from the oracle
        true_labels = oracle(problem, train_ixs, obs_labels, selected_ixs, **kwargs)

        # add the new labeled points to the training set
        train_ixs = np.concatenate([train_ixs, selected_ixs])
        obs_labels = np.concatenate([obs_labels, true_labels])
        # -------------------

        # -------------------
        if is_ssl_round(i, num_queries):
            # choose points from a model that should be targets
            pos_ixs, neg_ixs = ss_labeler(problem, train_ixs, obs_labels, unlabeled_ixs, label_history=label_history, **kwargs)
            proactive_ixs = np.concatenate([pos_ixs, neg_ixs])
            ss_labels = np.concatenate([
                np.ones(len(pos_ixs), dtype=int),
                np.zeros(len(neg_ixs), dtype=int)
            ])

            # just fake it till you make it and add these to the training set
            train_ixs = np.concatenate([train_ixs, proactive_ixs])
            obs_labels = np.concatenate([obs_labels, ss_labels])

        if clear_label_history(i, num_queries):
            label_history = []
        # -------------------

        # presumably the call back will update the model
        if callback is None:
            continue
        elif isinstance(callback, Iterable):
            for func in callback:
                func(problem, train_ixs, obs_labels, query=selected_ixs, ** kwargs)
        else:
            callback(problem, train_ixs, obs_labels, query=selected_ixs, **kwargs)
