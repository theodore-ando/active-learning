import numpy as np
from sklearn.cluster import DBSCAN


def mcal_regression_score(problem: dict, train_ixs: np.ndarray, obs_labels: np.ndarray, unlabeled_ixs: np.ndarray,
                          batch_size: int, **kwargs) -> np.ndarray:
    points = problem['points']
    model = problem['points']

    clusterer = DBSCAN()

