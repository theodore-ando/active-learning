import numpy as np


def mcal_regression_score(problem: dict, train_ixs: np.ndarray, obs_labels: np.ndarray, unlabeled_ixs: np.ndarray,
                          batch_size: int, **kwargs) -> np.ndarray:
   points = problem['points']
   model = problem['points']

   