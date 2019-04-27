from unittest import TestCase

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from active_learning.query_strats.classification.greedy import GreedySearch
from active_learning.tests.test_problem import make_grid_problem


class TestGreedy(TestCase):

    def test_greedy(self):
        # Make the grid problem with a KNN
        model = KNeighborsClassifier(n_neighbors=3)
        problem = make_grid_problem()
        problem.positive_label = 1
        model.fit(*problem.get_labeled_points())

        # Compute the probabilities for each test point
        greedy = GreedySearch(model)
        inds, scores = greedy.score_all(problem)
        probs = model.predict_proba(problem.points[inds])[:, 1]
        self.assertTrue(np.isclose(probs, scores).all())
