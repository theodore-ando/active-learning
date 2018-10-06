from ..uncertainty_sampling import UncertaintySampling
from ...tests.test_problem import make_grid_problem
from sklearn.neighbors import KNeighborsClassifier
from unittest import TestCase
import numpy as np


class TestUncertainty(TestCase):

    def test_uncertainty(self):
        # Make the grid problem with a KNN
        model = KNeighborsClassifier(n_neighbors=2)
        problem = make_grid_problem()
        problem.model = model
        problem.update_model()

        # Compute the uncertainties
        sampler = UncertaintySampling()
        probs = model.predict_proba(problem.points[problem.get_unlabeled_ixs()])
        score = -1 * np.multiply(probs, np.log(probs)).sum(axis=1)
        self.assertTrue(np.isclose(score,
                                   sampler._score_many(problem.get_unlabeled_ixs(), problem)).all())
