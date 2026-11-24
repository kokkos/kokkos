``ParallelForTag``
==================

.. role::cpp(code)
    :language: cpp

Header File: ``<Kokkos_ExecPolicy.hpp>``

.. _parallelFor: ../parallel-dispatch/parallel_for.rst

.. |parallelFor| replace:: :cpp:func:`parallel_for`

A tag used in team size calculation functions to indicate that the functor for which a team size is being requested is being used in a |parallelFor|_

Usage
-----

.. code-block:: cpp

    using PolicyType = Kokkos::TeamPolicy<>; 
    PolicyType policy;
    int recommended_team_size = policy.team_size_recommended(
        Functor, Kokkos::ParallelForTag());

Synopsis 
--------

.. code-block:: cpp

    struct ParallelForTag{};

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
