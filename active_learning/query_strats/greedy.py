from active_learning.query_strats import argmax
from active_learning.scoring import probability


def greedy(problem, train_ixs, obs_labels, unlabeled_ixs, npoints, **kwargs):
    """Greedily choose the most probably to be a target"""
    score_fn = probability
    return argmax(problem, train_ixs, obs_labels, unlabeled_ixs, score_fn, npoints)