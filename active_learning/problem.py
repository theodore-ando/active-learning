from sklearn.base import BaseEstimator
from typing import List
import numpy as np


class ActiveLearningProblem:
    """Class for defining an active learning problem.

    The main point in defining an active learning problem is to define the total search space,
    which points in this space have already been labeled, and what those labels are.

    Optionally, you can define the budget of how many points are left to label.
    """

    def __init__(self, points, labeled_ixs, labels, budget=None, positive_label=1,
                 model: BaseEstimator = None):
        """Set up the active learning problem

        Args:
            points (ndarray): Coordinates of all points in the search space
            labeled_ixs ([int]): Indices of points that have been labeled
            labels (ndarray): Labels for the labeled points, in same order as labeled_ixs
            budget (int): How many entries are budgeted to be labeled (default: all of them)
            positive_label (int): Classification: Which entry is the desired class
            model (BaseEstimator): Machine learning model used to guide training.
        """

        self.points = points
        self.labeled_ixs = labeled_ixs
        self.labels = list(labels)
        self.positive_label = positive_label
        self.model = model

        # Set the budget
        self.budget = None
        if budget is None:
            self.budget = len(points) - len(labeled_ixs)

    @classmethod
    def from_labeled_and_unlabled(cls, labeled_points, labels, unlabled_points, **kwargs):
        """Construct an active learning problem from labeled and unlabled points

        Args:
            labeled_points (ndarray): Coordinates of points with labels
            labels (ndarray): Labels of those points
            unlabled_points (ndarray): Points that could possibly be labeled
        """

        points = np.vstack((labeled_points, unlabled_points))
        labeled_ixs = list(range(len(labeled_points)))

        return cls(points, labeled_ixs, labels, **kwargs)

    def get_unlabeled_ixs(self) -> List[int]:
        """Get a list of the unlabeled indices

        Returns:
            ([int]) Unlabelled indices
        """
        return list(
            set(range(len(self.points))).difference(self.labeled_ixs)
        )

    def update_model(self):
        """Update the machine learning model given the current labeled set"""

        self.model.fit(self.points[self.labeled_ixs],
                       self.labels)
