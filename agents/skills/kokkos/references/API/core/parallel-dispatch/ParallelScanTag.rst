``ParallelScanTag``
===================

.. role::cpp(code)
    :language: cpp

Header File: ``<Kokkos_ExecPolicy.hpp>``

.. _parallelScan: ../parallel-dispatch/parallel_scan.rst

.. |parallelScan| replace:: :cpp:func:`parallel_scan`

A tag used in team size calculation functions to indicate that the functor for which a team size is being requested is being used in a |parallelScan|_

Usage
-----

.. code-block:: cpp

    using PolicyType = Kokkos::TeamPolicy<>; 
    PolicyType policy;
    int recommended_team_size = policy.team_size_recommended(
        Functor, Kokkos::ParallelScanTag());

Synopsis 
--------

.. code-block:: cpp

    struct ParallelScanTag{};

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
