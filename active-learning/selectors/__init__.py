import numpy as np


"""
A selector is a function that takes in the problem dictionary (described
below), the training set, and the observed labels for the training set, and
outputs the indices into problem['points'] of the unlabeled set of points for
consideration in the next round of queries.

Selectors should be functions of the form:

    selector(problem, train_ixs, obs_labels)
    
Where:
    :param problem: dictionary that defines the problem, containing keys:
        * points:       an (n_samples, n_dim) matrix of points in the space
        * num_classes:  the number of different classes [0, num_classes)
        * batch_size:   number of points to query each iteration
        * num_queries:  the max number of queries we can make on the data
        * model:        the sk-learn model we are training 
    :param train_ixs: index into `points` of the training examples
    :param obs_labels: labels for the training examples
    :returns: np.array of ints indexing into problem['points']
"""


def identity_selector(problem, train_ixs, obs_labels, **kwargs):
    space = problem['points']
    num_points = space.shape[0]
    ixs = np.arange(num_points)

    train_mask = np.zeros(space.shape[0], dtype=bool)
    train_mask[train_ixs] = True

    # return every index not in the training mask
    return ixs[~train_mask]