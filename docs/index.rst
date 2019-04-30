Welcome to active-learning's documentation!
===========================================

``active-learning`` is a library of different active learning methods.
Active learning methods are designed to identify what new data to capture,
and a variety of algorithms exist depending on what goals you have.
There are algorithms for when your objectives are to build the best machine-learning model.
There are also algorithms for when your objective is optimization (i.e., collecting the "best" data),
or even a combination of these two objectives.
Some algorithms are designed to identify a single experiment to run, and others optimize selecting a batch of new entries.
We created ``active-learning`` to make it easy to try out these different algorithms.

The key algorithms implemented in ``active-learning`` are known as "query strategies."
Query strategies are designed to identify which new experiments to perform provided a list of
experiments have been performed, the results of those experiments, and a list of valid new experiments.
Each query strategy is

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   problem-definition
   query-strategies
   api-docs



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
