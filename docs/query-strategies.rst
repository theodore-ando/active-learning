Query Strategies
================

This portion of the documentation details the query strategies that are available in ``active-learning``

Base API
--------

All of the query strategies are based on the ``BaseQueryStrategy``,
which provides a consistent API between all strategies.

.. autoclass:: active_learning.query_strats.base.BaseQueryStrategy
    :members:
    :noindex:

General Strategies
------------------

Several active learning strategies are agnostic to the type of problem being solved (e.g., classification, regression).

.. automodule:: active_learning.query_strats
    :members:
    :noindex:

Classification
--------------

``active-learning`` provides several algorithms for active learning in classification tasks.


.. automodule:: active_learning.query_strats.classification
    :members:
    :exclude-members: select_points
    :noindex:

Regression
----------

There are also many algorithms for supporting regression tasks.

.. automodule:: active_learning.query_strats.regression
    :members:
    :exclude-members: select_points
    :noindex:
