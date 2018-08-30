def probability(problem, train_ixs, obs_labels, selected_ixs, batch_size, **kwargs):
    """
    Score is simply the probability of being a target under current model.
    :param problem: dictionary that defines the problem, containing keys:
        * points:       an (n_samples, n_dim) matrix of points in the space
        * num_classes:  the number of different classes [0, num_classes)
        * batch_size:   number of points to query each iteration
        * num_queries:  the max number of queries we can make on the data
        * model:        the sk-learn model we are training
    :param train_ixs: index into `points` of the training examples
    :param obs_labels: labels for the training examples
    :param selected_ixs: indices into problem['points'] to score
    :param kwargs: unused
    :return: scores for each of selected_ixs
    """
    points = problem['points']
    model = problem['model']

    test_X = points[selected_ixs]

    p_x = model.predict_proba(test_X)

    return p_x[:,1].reshape(-1)