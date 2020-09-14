from unittest import TestCase

from sklearn.neighbors import KNeighborsClassifier

from active_learning.query_strats.classification.three_ds import ThreeDs
from active_learning.tests.test_problem import make_grid_problem


class TestUncertainty(TestCase):

    def test_uncertainty(self):
        # Make the grid problem with a KNN
        model = KNeighborsClassifier(n_neighbors=3)
        problem = make_grid_problem()

        # Run the selection
        d = ThreeDs(model)
        pts = d.select_points(problem, 4)
        self.assertEqual(4, len(set(pts)))

        d.dwc = 0
        pts = d.select_points(problem, 4)
        self.assertEqual(4, len(set(pts)))
