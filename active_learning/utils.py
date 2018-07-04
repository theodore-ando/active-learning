import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from active_learning.selectors import identity_selector
from active_learning.ss_active_learning import ss_actively_learn, _default_ssl_config
from .active_learning import _actively_learn
from active_learning.query_strats.uncertainty_sampling import uncertainty_sampling


def chunks(l: list, n: int) -> list:
    """
    Chunks a list into parts
    :param l:
    :param n:
    """
    chunk_size = max(len(l) // n, 1)
    while len(l):
        yield l[:chunk_size]
        l = l[chunk_size:]


def make_training_set(estimator, y: np.ndarray, size: int = 5, n_targets: int = 1):
    """
    Produces a training set for active learning problems.

    :param estimator: estimator we are training
    :param y: array of labels for points in the problem space
    :param size: size of the training set to produce
    :param n_targets: number of targets (y=1) to include in training set
    :return: array of indices into ``y``
    """
    assert size >= 0 and 0 <= n_targets <= size

    train_ixs = np.random.randint(low=0, high=len(y), size=size)

    # if classification problem, we only want to include so many targets in the input
    if isinstance(estimator, ClassifierMixin):
        while sum(y[train_ixs]) != n_targets:
            train_ixs = np.random.randint(low=0, high=len(y), size=size)

    # otherwise we can just use random selection
    return train_ixs


# -----------------------------------------------------------------------------
#                       Useful callback functions
# -----------------------------------------------------------------------------


def retrain_model(problem, train_ixs, obs_labels):
    """
    Retrain the problem['model'] on train_ixs and obs_labels.
    :param problem:
    :param train_ixs:
    :param obs_labels:
    :return:
    """
    points = problem['points']
    model = problem['model']
    problem['model'] = model.fit(points[train_ixs], obs_labels)


def make_save_scores(scores, score_fn, X_test, y_test):
    """
    Callback to save the score of the model on a test set (i.e.: accuracy, F1, etc)
    :param X_test:
    :param y_test:
    :return:
    """
    def save_scores(problem, train_ixs, obs_labels, **kwargs):
        model = problem['model']
        pred_labels = model.predict(X_test)
        score = score_fn(y_test, pred_labels)
        scores.append(score)

    return save_scores


def make_save_queries(queries_list):
    def save_queries(problem, train_ixs, obs_labels, **kwargs):
        queries_list.append(kwargs['query'])

    return save_queries


def make_history_retrain(history):
    """
    Makes a `callback` function that can be passed to `actively_learn` that
    accumulates the number of targets seen so far in each iteration in a list.
    :return: (pointer to the list, callback function)
    """

    def plot_retrain_model(problem, train_ixs, obs_labels, **kwargs):
        history.append(np.sum(obs_labels))
        retrain_model(problem, train_ixs, obs_labels)

    return plot_retrain_model


def make_callback_retrain(func):
    def retrain_callback(problem, train_ixs, obs_labels, **kwargs):
        func(problem, train_ixs, obs_labels, **kwargs)
        retrain_model(problem, train_ixs, obs_labels)

    return retrain_callback


# -----------------------------------------------------------------------------
#                                 Oracles
# -----------------------------------------------------------------------------


def make_training_oracle(y_true: np.array):
    """
    Makes a training oracle:
        training_oracle(problem, train_ixs, obs_labels, selected_ixs)
    That returns y_true[selected_ixs].  Useful for training active learning
    models.
    :param y_true: np array of the true labels
    :return:
    """

    def training_oracle(problem, train_ixs, obs_labels, selected_ixs, **kwargs):
        return y_true[selected_ixs]

    return training_oracle


# -----------------------------------------------------------------------------
#                                 Testing Stuff
# -----------------------------------------------------------------------------


def _standardize_score_fns(score_fns) -> dict:
    if isinstance(score_fns, dict):
        return score_fns

    if callable(score_fns):
        return {
            "score": score_fns
        }

    if score_fns is None:
        return {
            "accuracy": accuracy_score
        }

    if isinstance(score_fns, list) or isinstance(score_fns, tuple):
        if callable(score_fns[0]):
            return {
                f"score{i}": score_fn
                for i, score_fn in enumerate(score_fns)
            }

        return dict(score_fns)

    raise TypeError("score_fns bad")


def perform_experiment(X, y, base_estimator=SVC(probability=True), n_queries=40, batch_size=1, semisupervised=False,
                       init_L_size=2, selector=identity_selector, query_strat=uncertainty_sampling,
                       score_fns=None, random_state=None):
    score_fns = _standardize_score_fns(score_fns)

    # These are the fields which can be filled in
    experiment_data = {
        "n_targets": [],
        "queries": [],
        "history": []
    }

    callbacks = []

    callbacks += [make_save_queries(experiment_data["queries"])]
    callbacks += [make_history_retrain(experiment_data["history"])]

    # if score_fn, then
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
    X, y = X_train, y_train

    for name, score_fn in score_fns.items():
        experiment_data[name] = []
        score_callback = make_save_scores(experiment_data[name], score_fn, X_test, y_test)
        callbacks.append(score_callback)


    L = make_training_set(base_estimator, y, size=init_L_size)
    oracle = make_training_oracle(y)

    problem = {
        "model": base_estimator,
        "num_queries": n_queries,
        "batch_size": batch_size,
        "points": X,
        "training_set_size": init_L_size
    }

    retrain_model(problem, L, y[L])

    if not semisupervised:
        _actively_learn(
            problem,
            L, y[L],
            oracle=oracle,
            selector=selector,
            query_strat=query_strat,
            callback=callbacks
        )
    else:
        # load some good defaults
        default_ssl_config = _default_ssl_config()
        ssl_config = default_ssl_config.copy()
        if isinstance(semisupervised, dict):
            ssl_config.update(semisupervised)
            ssl_config.update(problem)
            problem = ssl_config

        ss_actively_learn(
            problem,
            L, y[L],
            oracle=oracle,
            query_strat=query_strat,
            callback=callbacks
        )

    return {**experiment_data, **problem}
