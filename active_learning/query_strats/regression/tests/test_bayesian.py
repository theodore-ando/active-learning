from active_learning.query_strats.regression.bayesian import ExpectedImprovement
from active_learning.tests.test_problem import make_xsinx
from sklearn.linear_model import BayesianRidge
from unittest import TestCase


class TestBayesian(TestCase):

    def test_ei(self):
        # Make the problem
        problem = make_xsinx()
        model = BayesianRidge()

        # Run the sampling. For now, let's just test that it gives the correct number of samples
        uncert = ExpectedImprovement(model)
        selections = uncert.select_points(problem, 2)
        self.assertEqual(len(selections), 2)
