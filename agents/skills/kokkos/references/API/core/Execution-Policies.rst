Execution Policies
##################

Top Level Execution Policies
============================

`ExecutionPolicyConcept <policies/ExecutionPolicyConcept.rst>`__ is the fundamental abstraction to represent “how” the execution of a Kokkos parallel pattern takes place.

.. list-table::
    :widths: 35 65
    :header-rows: 1
    :align: left

    * - Policy
      - Description

    * * `RangePolicy <policies/RangePolicy.rst>`__
      * Each iterate is an integer in a contiguous range

    * * `MDRangePolicy <policies/MDRangePolicy.rst>`_
      * Each iterate for each rank is an integer in a contiguous range

    * * `TeamPolicy <policies/TeamPolicy.rst>`__
      * Assigns to each iterate in a contiguous range a team of threads

Nested Execution Policies
============================

Nested Execution Policies are used to dispatch parallel work inside of an already executing parallel region either dispatched with a `TeamPolicy <policies/TeamPolicy.rst>`__ or a task policy. `NestedPolicies <policies/NestedPolicies.rst>`__ summary.

.. list-table::
    :widths: 25 75
    :header-rows: 1
    :align: left

    * - Policy
      - Description

    * * `TeamThreadMDRange <policies/TeamThreadMDRange.rst>`__
      * Used inside of a TeamPolicy kernel to perform nested parallel loops over a multidimensional range split over threads of a team.

    * * `TeamThreadRange <policies/TeamThreadRange.rst>`__
      * Used inside of a TeamPolicy kernel to perform nested parallel loops split over threads of a team.

    * * `TeamVectorMDRange <policies/TeamVectorMDRange.rst>`__
      * Used inside of a TeamPolicy kernel to perform nested parallel loops over a multidimensional range split over threads of a team and their vector lanes.

    * * `TeamVectorRange <policies/TeamVectorRange.rst>`__
      * Used inside of a TeamPolicy kernel to perform nested parallel loops split over threads of a team and their vector lanes.

    * * `ThreadVectorMDRange <policies/ThreadVectorMDRange.rst>`__
      * Used inside of a TeamPolicy kernel to perform nested parallel loops over a multidimensional range with vector lanes of a thread.

    * * `ThreadVectorRange <policies/ThreadVectorRange.rst>`__
      * Used inside of a TeamPolicy kernel to perform nested parallel loops with vector lanes of a thread.

Common Arguments for all Execution Policies
===========================================

Execution Policies generally accept compile time arguments via template parameters and runtime parameters via constructor arguments or setter functions.

.. tip::

    Template arguments can be given in arbitrary order.

.. list-table::
    :widths: 30 30 40
    :header-rows: 1
    :align: left

    * - Argument
      - Options
      - Purpose

    * * ExecutionSpace
      * ``Serial``, ``OpenMP``, ``Threads``, ``Cuda``, ``HIP``, ``SYCL``, ``HPX``
      * Specify the Execution Space to execute the kernel in. Defaults to ``Kokkos::DefaultExecutionSpace``

    * * Schedule
      * ``Schedule<Dynamic>``, ``Schedule<Static>``
      * Specify scheduling policy for work items. ``Dynamic`` scheduling is implemented through a work stealing queue. Default is machine and backend specific.

    * * IndexType
      * e.g. ``IndexType<int>``
      * Specify integer type to be used for traversing the iteration space. Defaults to the ``size_type`` of `ExecutionSpaceConcept <execution_spaces.rst#typedefs>`__. Can affect the performance depending on the backend.

    * * LaunchBounds
      * ``LaunchBounds<MaxThreads, MinBlocks>``
      * Specifies hints to to the compiler about CUDA/HIP launch bounds.

    * * WorkTag
      * ``SomeClass``
      * Specify the work tag type used to call the functor operator. Can be any arbitrary tag type (i.e. an [empty](https://en.cppreference.com/w/cpp/types/is_empty) struct or class). Defaults to ``void``.


.. toctree::
   :hidden:
   :maxdepth: 1

   ./policies/ExecutionPolicyConcept
   ./policies/MDRangePolicy
   ./policies/NestedPolicies
   ./policies/RangePolicy
   ./policies/TeamHandleConcept
   ./policies/TeamPolicy
   ./policies/TeamThreadMDRange
   ./policies/TeamThreadRange
   ./policies/TeamVectorMDRange
   ./policies/TeamVectorRange
   ./policies/ThreadVectorMDRange
   ./policies/ThreadVectorRange
