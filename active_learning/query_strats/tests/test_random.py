from active_learning.tests.test_problem import make_grid_problem
from active_learning.query_strats.random_sampling import RandomQuery
from unittest import TestCase


class RandomTest(TestCase):

    def test_random(self):
        problem = make_grid_problem()

        # Make the selection tool
        query = RandomQuery()

        # Test with no threads
        output = query.select_points(problem, 2)
        self.assertEqual(2, len(output))

        # Test with 2 threads
        query.n_cpus = 2
        output = query.select_points(problem, 3)
        self.assertEqual(3, len(output))
