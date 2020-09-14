from active_learning.query_strats.classification.batch_active_search\
    import SequentialSimulatedBatchSearch
from active_learning.query_strats.random_sampling import RandomQuery
from active_learning.tests.test_problem import make_grid_problem
from unittest import TestCase

from sklearn.neighbors import KNeighborsClassifier


class TestBatch(TestCase):

    def test_seq_sim(self):
        problem = make_grid_problem()

        # Make sure the code works.
        # TODO: Figure out a batch active learning problem we can model analytically -lw
        model = KNeighborsClassifier()
        query_strat = RandomQuery()
        seq_sim = SequentialSimulatedBatchSearch(model, query_strat, 'pessimistic')
        points = seq_sim.select_points(problem, 3)
        self.assertEqual(3, len(points))
