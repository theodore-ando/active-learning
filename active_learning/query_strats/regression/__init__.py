"""Query strategies specific to regression problems"""

from .greedy import GreedySelection
from .mcal_regression import MCALSelection
from .uncertainty import UncertaintySampling

__all__ = ['GreedySelection', 'MCALSelection', 'UncertaintySampling']
