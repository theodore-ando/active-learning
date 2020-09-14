Problem Definition
==================

``active-learning`` is built around a core class, :class:`active_learning.problem.ActiveLearningProblem`, that defines the active learning problem.

Problem definitions require two features to be defined:

#. *Search Space*: All possible experiments that could be performed
#. *Labels*: A set of experiments that have been perform, and their results (i.e., labels)

Some active learning algorithms also make use of further information about the active learning problem:

#. *Budget*: How many new experiments can be performed
#. *Objective*: Whether there are any desired experimental outcomes (e.g., successes, optimized properties)

The ``ActiveLearningProblem`` class stores all of this information in a single object to simplify defining and performing active learning:

.. autoclass:: active_learning.problem.ActiveLearningProblem
    :members:
    :noindex:
