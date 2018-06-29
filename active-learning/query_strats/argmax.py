import numpy as np


def _partition_ixs(partition, unlabeled_ixs):
    """
    Splits the unlabeled_ixs along the partitions given.
    :param partition: list of np.array of indices
    :param unlabeled_ixs: np.array of indices
    :return: mask of length unlabeled_ixs indicating which are in each partition
    """
    return [
        np.in1d(unlabeled_ixs, p_i)
        for p_i in partition
    ]


def argmax(problem, train_ixs, obs_labels, unlabeled_ixs, score, batch_size, **kwargs):
    """
    Generic arg-maximizer query strat that can be used with different scoring mechanisms.
    :param problem: dictionary that defines the problem, containing keys:
        * points:       an (n_samples, n_dim) matrix of points in the space
        * num_classes:  the number of different classes [0, num_classes)
        * batch_size:   number of points to query each iteration
        * num_queries:  the max number of queries we can make on the data
        * model:        the sk-learn model we are training
        Optional:
        - partition: list of sets of indices into problem['points'] partitioning the space.
            This can restrict the batch to be from one partition!
    :param train_ixs: index into `points` of the training examples
    :param obs_labels: labels for the training examples
    :param unlabeled_ixs: np.array of the indices of the unlabeled examples
    :param score: scoring function (see scoring API)
    :param batch_size: size of the batch to select
    :param kwargs: passed on to scoring function
    :return:
    """
    scores = score(problem, train_ixs, obs_labels, unlabeled_ixs, batch_size, **kwargs)

    # default to argmax over all unlabeled indices
    if "partition" not in problem:
        best_ixs = np.argsort(scores)[-batch_size:]
        # print(scores[best_ixs])
        # print(scores)
        return unlabeled_ixs[best_ixs]

    # user can specify partition (maybe only can take batch from one experiment at a time?)
    partition = problem['partition']
    part_unlabeled_ixs = _partition_ixs(partition, unlabeled_ixs)

    # find best batch for each partition, choose max across those
    max_batch_score = -np.inf
    max_batch = None
    for p_ixs in part_unlabeled_ixs:
        # mask the scores of the elements not in partition
        p_score = np.ma.masked_array(scores, mask=~p_ixs)

        # can only choose up to min(num things in partition, batch_size) items from this partition
        k = min(np.sum(p_ixs), batch_size)

        # indices of top k scores
        top_k_ixs = p_score.argsort(fill_value=-np.inf)[-k:]

        # score for the batch would be sum of them
        batch_score = np.sum(p_score[top_k_ixs])
        if batch_score > max_batch_score:
            max_batch_score = batch_score

            # batch_mask = np.zeros(p_ixs.shape, dtype=bool)
            # batch_mask[top_k_ixs] = True
            # max_batch = batch_mask
            max_batch = top_k_ixs

    return unlabeled_ixs[max_batch]