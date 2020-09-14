from unittest import TestCase

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from active_learning.query_strats.classification.uncertainty_sampling import UncertaintySampling
from active_learning.tests.test_problem import make_grid_problem


class TestUncertainty(TestCase):

    def test_uncertainty(self):
        # Make the grid problem with a KNN
        model = KNeighborsClassifier(n_neighbors=2)
        problem = make_grid_problem()
        model.fit(*problem.get_labeled_points())

        # Compute the uncertainties
        sampler = UncertaintySampling(model)
        probs = model.predict_proba(problem.points[problem.get_unlabeled_ixs()])
        score = -1 * np.multiply(probs, np.log(probs)).sum(axis=1)
        self.assertTrue(np.isclose(score, sampler.score_all(problem)[1]).all())
