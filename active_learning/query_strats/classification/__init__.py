"""Query strategies specific to classification problems"""

from .active_search import ActiveSearch
from .batch_active_search import SequentialSimulatedBatchSearch
from .greedy import GreedySearch
from .three_ds import ThreeDs
from .uncertainty_sampling import UncertaintySampling

__all__ = ['ActiveSearch', 'SequentialSimulatedBatchSearch', 'GreedySearch',
           'ThreeDs', 'UncertaintySampling']
