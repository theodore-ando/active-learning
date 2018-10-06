from ..active_search import ActiveSearch
from ..greedy import GreedySearch
from ...tests.test_problem import make_grid_problem
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from unittest import TestCase
import numpy as np


class TestActiveSearch(TestCase):

    def test_active_search(self):
        # Make the grid problem with a KNN
        model = KNeighborsClassifier(n_neighbors=3)
        problem = make_grid_problem()
        problem.model = model
        problem.positive_label = 1
        problem.budget = 1
        problem.update_model()

        # For a budget of 1, active search should be equal to greedy
        active_search = ActiveSearch()
        greedy = GreedySearch()
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

