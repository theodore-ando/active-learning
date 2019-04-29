from unittest import TestCase
from active_learning.query_strats.regression.mcal_regression import MCALSelection
from active_learning.tests.test_problem import make_xsinx


class TestMCAL(TestCase):

    def test_mcal(self):
        problem = make_xsinx()

        # Not sure how this one functions yet,
        #   so we're just going to make sure it does not crash
        mcal = MCALSelection()
        selection = mcal.select_points(problem, 4)
        self.assertEquals(4, len(selection))
