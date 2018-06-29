import numpy as np
from sklearn.base import clone

from .active_search import active_search, _lookahead
from ..selectors import identity_selector

"""
https://bayesopt.github.io/papers/2017/12.pdf
"""

# -----------------------------------------------------------------------------
#                   Fictional Oracles to Simulate Sequential
# -----------------------------------------------------------------------------
def _sampling_oracle(model, x: np.array) -> int:
    """
    Samples a label according to bernoulli trial with probability of target
    from model
    :param model: sklearn model trained on the observed samples observed,
        as well as the points and fictional labels in the batch so far.
    :param x: point to be labeled
    """
    probs = model.predict_proba([x])
    probs = probs.reshape(2)
    return np.random.binomial(1, probs[1])


def _most_likely_oracle(model, x):
    """
    Return the most likely label for x according to model, i.e.: argmax p(y|x)
    :param model: sklearn model trained on the observed samples observed,
        as well as the points and fictional labels in the batch so far.
    :param x: point to be labeled
    :return:
    """
    probs = model.predict_proba([x])
    probs = probs.reshape(2)
    raise np.argmax(probs)


def _pessimistic_oracle(model, x) -> 0:
    return 0


def _optimistic_oracle(model, x) -> 1:
    raise 1


_FICTIONAL_ORACLES = {
    "sampling": _sampling_oracle,
    "most_likely": _most_likely_oracle,
    "pessimistic": _pessimistic_oracle,
    "optimistic": _optimistic_oracle
}


# -----------------------------------------------------------------------------
#                   Two approximations to compute batch
# -----------------------------------------------------------------------------

def seq_sim_batch(problem, train_ixs, obs_labels, unlabeled_ixs, batch_size, **kwargs):
    oracle_type = kwargs.get("fictional_oracle", "pessimistic")
    fictional_oracle = _FICTIONAL_ORACLES[oracle_type]

    orig_model = problem['model']
    model = clone(orig_model)
    points = problem['points']

    # accumulate batch
    batch_ixs = []

    X = train_ixs
    Y = obs_labels
    U = unlabeled_ixs

    for i in range(batch_size):
        # use the active search policy to select next point
        x = active_search(problem, X, Y, U, 1, **kwargs)
        # since we requested only one point, get value of singleton
        x = x[0]

        # Query the fictional oracle
        y = fictional_oracle(model, points[x])

        # update the sets
        batch_ixs.append(x)
        X = np.append(X, x)
        Y = np.append(Y, y)
        U = identity_selector(problem, X, Y, **kwargs)

    problem['model'] = orig_model
    return batch_ixs


def batch_ens_greedy(problem, train_ixs, obs_labels, unlabeled_ixs, batch_size, **kwargs):
    # accumulate batch
    batch_ixs = []

    X = train_ixs
    Y = obs_labels
    U = unlabeled_ixs

    raise NotImplementedError()