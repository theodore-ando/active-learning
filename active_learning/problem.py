"""Classes and methods related to defining an active learning problem"""

from .objective import ObjectiveFunction, Minimize
from typing import List, Tuple, Union
import numpy as np


class ActiveLearningProblem:
    """Class for defining an active learning problem.

    The main point in defining an active learning problem is to define the total search space,
    which points in this space have already been labeled, and what those labels are.

    Optionally, you can define the budget of how many points are left to label.
    """

    def __init__(self, points, labeled_ixs: List[int], labels,
                 budget=None, target_label=1, objective_fun: ObjectiveFunction = Minimize()):
        """Set up the active learning problem

        Args:
            points (ndarray): Coordinates of all points in the search space
            labeled_ixs ([int]): Indices of points that have been labeled
            labels (ndarray): Labels for the labeled points, in same order as labeled_ixs
            budget (int): How many entries are budgeted to be labeled (default: all of them)
            target_label (int): Index the desired class, used in classification problems
            objective_fun (ObjectiveFunction): Objective function, used in regression problems
        """

        # TODO: Add batch size and support for grouping points together -lw
        self.points = points
        self.labeled_ixs = labeled_ixs
        self.labels = list(labels)
        self.target_label = target_label
        self.objective_fun = objective_fun

        # Set the budget
        self.budget = budget
        if budget is None:
            self.budget = len(points) - len(labeled_ixs)

    @classmethod
    def from_labeled_and_unlabled(cls, labeled_points, labels, unlabeled_points, **kwargs):
        """Construct an active learning problem from labeled and unlabled points

        Args:
            labeled_points (ndarray): Coordinates of points with labels
            labels (ndarray): Labels of those points
            unlabeled_points (ndarray): Points that could possibly be labeled
        """

        points = np.vstack((labeled_points, unlabeled_points))
        labeled_ixs = list(range(len(labeled_points)))

        return cls(points, labeled_ixs, labels, **kwargs)

    def get_unlabeled_ixs(self) -> List[int]:
        """Get a list of the unlabeled indices

        Returns:
            ([int]) Unlabeled indices
        """
        return list(
            set(range(len(self.points))).difference(self.labeled_ixs)
        )

    def get_labeled_ixs(self) -> List[int]:
        """Get a list of the labeled indices

        Returns:
            ([int]): Labeled indices
        """
        return list(self.labeled_ixs)

    def add_label(self, ind: int, label: float):
        """Add a label to the labeled set

        Args:
            ind (int): Index of point to label
            label (float): Label of that point
        """

        if ind in self.labeled_ixs:
            raise AttributeError('Index already included in labeled set')
        self.labeled_ixs.append(ind)
        self.labels.append(label)

    def get_labeled_points(self) -> Tuple[np.ndarray, List[Union[float, int]]]:
        """Get the labeled points and their labels

        Returns:
            - (ndarray): Coordinates of all points with labels
            - (ndarray): Labels for all labeled points
        """
        return self.points[self.labeled_ixs], self.labels

    def get_unlabeled_points(self) -> np.ndarray:
        """Get the coordinates of all unlabeled points

        Returns:
            (list) Coordinates of all unlabeled points
        """
        return self.points[self.get_unlabeled_ixs()]
