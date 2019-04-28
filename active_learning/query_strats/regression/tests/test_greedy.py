from unittest import TestCase
from sklearn.linear_model import LinearRegression
from active_learning.query_strats.regression.greedy import GreedySelection
from active_learning.tests.test_problem import make_xsinx


class TestGreedy(TestCase):

    def test_greedy(self):
        problem = make_xsinx()
        model = LinearRegression()

        # Should pick either the largest or the smallest index,
        # depending on the slope of the linear regression line
        greedy = GreedySelection(model)
        selection = greedy.select_points(problem, 1)[0]
        self.assertTrue(selection == 0 or selection == 15)
