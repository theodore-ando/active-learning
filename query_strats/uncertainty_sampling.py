import numpy as np

from . import argmax
from active_learning.scoring.marginal_entropy import marginal_entropy


def uncertainty_sampling(problem, train_ixs, obs_labels, unlabeled_ixs, npoints, **kwargs):
    score = marginal_entropy

    return argmax(problem, train_ixs, obs_labels, unlabeled_ixs, score, npoints)