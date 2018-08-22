import numpy as np


def random_sampling(problem, train_ixs, obs_labels, unlabeled_ixs, npoints, **kwargs):
    """Simple random sample of points from unlabeled indices"""
    rand_ixs = np.random.randint(0, len(unlabeled_ixs), size=npoints)

    return unlabeled_ixs[rand_ixs]