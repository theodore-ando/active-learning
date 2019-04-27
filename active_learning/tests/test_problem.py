from ..problem import ActiveLearningProblem
from unittest import TestCase
import numpy as np


def make_grid_problem():
    """Makes a test active learning problem based on a grid

        0   1   2
      -------------
    0 | x | ? | ? |
      -------------
    1 | ? | o | x |
      -------------
    2 | x | ? | o |
      -------------

    The grid is designed to make computing the utilities of each point easier.
    """

    # Print out the points
    x_known = [(0, 0), (1, 1), (1, 2), (2, 0), (2, 2)]
    x_labels = [1, 0, 1, 1, 0]
    x_unlabeled = [(0, 1), (0, 2), (1, 0), (2, 1)]

    # Make the active learning problem
    return ActiveLearningProblem.from_labeled_and_unlabled(x_known, x_labels, x_unlabeled,
                                                           target_label=1)


def make_xsinx():
    points = np.arange(0, 16)[:, None]
    y = np.squeeze(points * np.sin(points))
    selection = [1, 4, 11, 6]
    return ActiveLearningProblem(points, selection, y[selection])


class TestProblem(TestCase):

    def test_grid(self):
        prob = make_grid_problem()

        self.assertEqual((9, 2), prob.points.shape)
        self.assertListEqual(list(range(5)), prob.labeled_ixs)
        self.assertListEqual(list(range(5, 9)), prob.get_unlabeled_ixs())
        self.assertEqual(1, prob.target_label)
