from active_learning.query_strats.regression import UncertaintySampling
from active_learning.tests.test_problem import make_xsinx
from sklearn.linear_model import BayesianRidge
from unittest import TestCase
import numpy as np


class TestUncertainty(TestCase):

    def test_uncertainty(self):
        # Make the problem
        problem = make_xsinx()
        model = BayesianRidge()

        # Figure out the answer beforehand
        model.fit(*problem.get_labeled_points())
        _, y_std = model.predict(problem.get_unlabeled_points(), return_std=True)
        max_ix = problem.get_unlabeled_ixs()[np.argmax(y_std)]

        # Run the sampling
        uncert = UncertaintySampling(model)
        selections = uncert.select_points(problem, 1)
        self.assertEqual(selections, max_ix)
