from ..greedy import GreedySearch
from ...tests.test_problem import make_grid_problem
from sklearn.neighbors import KNeighborsClassifier
from unittest import TestCase
import numpy as np


class TestGreedy(TestCase):

    def test_greedy(self):
        # Make the grid problem with a KNN
        model = KNeighborsClassifier(n_neighbors=3)
        problem = make_grid_problem()
        problem.positive_label = 1
        problem.model = model
        problem.update_model()

        # Compute the probabilities for each test point
        greedy = GreedySearch()
        inds, scores = greedy.score_all(problem)
        probs = model.predict_proba(problem.points[inds])[:, 1]
        self.assertTrue(np.isclose(probs, scores).all())
