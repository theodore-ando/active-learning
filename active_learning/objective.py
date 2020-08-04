"""Objective functions used in defining an active learning problem"""

from typing import List
import numpy as np


class ObjectiveFunction:
    """Class that generates objective function scores for regression functions"""

    def score(self, y: List, y_uncert: List = None) -> List[float]:
        """Generate the objective function score

        Args:
            y (list): Values of a class for many entries
            y_uncert (list): Any kind of uncertainty values
        Returns:
            ([float]): Scores where minimal values are preferred
        """
        raise NotImplementedError


class Maximize(ObjectiveFunction):
    """Find the maximum scalar value"""

    def score(self, y: List, y_uncert: List = None) -> List[float]:
        return np.multiply(y, -1)


class Minimize(ObjectiveFunction):

    def score(self, y: List, y_uncert: List = None) -> List[float]:
        return y
