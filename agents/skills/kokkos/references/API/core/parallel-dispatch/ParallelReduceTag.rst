``ParallelReduceTag``
=====================

.. role::cpp(code)
    :language: cpp

Header File: ``<Kokkos_ExecPolicy.hpp>``

.. _parallelReduce: ../parallel-dispatch/parallel_reduce.rst

.. |parallelReduce| replace:: :cpp:func:`parallel_reduce`

A tag used in team size calculation functions to indicate that the functor for which a team size is being requested is being used in a |parallelReduce|_

Usage
-----

.. code-block:: cpp

    using PolicyType = Kokkos::TeamPolicy<>; 
    PolicyType policy;
    int recommended_team_size = policy.team_size_recommended(
        Functor, Kokkos::ParallelReduceTag());

Synopsis
--------

.. code-block:: cpp

    struct ParallelReduceTag{};

Public Class Members
--------------------

None

Typedefs
~~~~~~~~

None

Constructors
~~~~~~~~~~~~

Default

Functions
~~~~~~~~~

None
