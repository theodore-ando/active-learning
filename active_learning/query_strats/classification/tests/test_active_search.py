from unittest import TestCase

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from active_learning.query_strats.classification.active_search import ActiveSearch
from active_learning.query_strats.classification.greedy import GreedySearch
from active_learning.tests.test_problem import make_grid_problem


class TestActiveSearch(TestCase):

    def test_active_search(self):
        # Make the grid problem with a KNN
        model = KNeighborsClassifier(n_neighbors=3)
        problem = make_grid_problem()
        problem.positive_label = 1
        problem.budget = 1
        model.fit(*problem.get_labeled_points())

        # For a budget of 1, active search should be equal to greedy
        active_search = ActiveSearch(model)
        greedy = GreedySearch(model)
        inds, score = active_search.score_all(problem)
        greedy_score = greedy._score_chunk(inds, problem)
        self.assertTrue(np.isclose(greedy_score, score).all())

        # Test with a larger lookahead
        #  Have not figured out a good case to solve by hand, so these test
        #   results are from a reference implement.
        problem.budget = 2
        score = active_search._score_chunk(sorted(inds), problem)
        self.assertTrue(np.isclose(score,
                                   [[4./3, 4./3, 4./3, 1]]).all())
