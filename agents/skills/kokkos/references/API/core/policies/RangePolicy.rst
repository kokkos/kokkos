``RangePolicy``
===============

.. role::cpp(code)
    :language: cpp

Header File: ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

    Kokkos::RangePolicy<...>(begin, end)
    Kokkos::RangePolicy<...>(begin, end, chunk_size)
    Kokkos::RangePolicy<...>(exec, begin, end)
    Kokkos::RangePolicy<...>(exec, begin, end, chunk_size)

    // CTAD Constructors (since 4.3)
    Kokkos::RangePolicy(begin, end)
    Kokkos::RangePolicy(begin, end, chunk_size)
    Kokkos::RangePolicy(exec, begin, end)
    Kokkos::RangePolicy(exec, begin, end, chunk_size)

RangePolicy defines an execution policy for a 1D iteration space starting at ``begin`` and going to ``end`` with an open interval.

Synopsis
--------

.. code-block:: cpp

    struct Kokkos::ChunkSize {
        explicit ChunkSize(int value_);
    };

    template<class ... Args>
    struct Kokkos::RangePolicy {
        using execution_policy = RangePolicy;
        using member_type = PolicyTraits<Args...>::index_type;

        // Inherited from PolicyTraits<Args...>
        using execution_space   = PolicyTraits<Args...>::execution_space;
        using schedule_type     = PolicyTraits<Args...>::schedule_type;
        using work_tag          = PolicyTraits<Args...>::work_tag;
        using index_type        = PolicyTraits<Args...>::index_type;
        using iteration_pattern = PolicyTraits<Args...>::iteration_pattern;
        using launch_bounds     = PolicyTraits<Args...>::launch_bounds;

        // Constructors
        RangePolicy(const RangePolicy&) = default;
        RangePolicy(RangePolicy&&) = default;

        RangePolicy();

        RangePolicy( index_type work_begin
                   , index_type work_end );

        RangePolicy( index_type work_begin
                   , index_type work_end
                   , ChunkSize chunk_size );

        RangePolicy( const execution_space & work_space
                   , index_type work_begin
                   , index_type work_end );

        RangePolicy( const execution_space & work_space
                   , index_type work_begin
                   , index_type work_end
                   , ChunkSize chunk_size );

        // retrieve chunk_size
        index_type chunk_size() const;
        // set chunk_size to a discrete value
        RangePolicy& set_chunk_size(int chunk_size_);

        // return ExecSpace instance provided to the constructor
        KOKKOS_FUNCTION const execution_space & space() const;
        // return Range begin
        KOKKOS_FUNCTION member_type begin() const;
        // return Range end
        KOKKOS_FUNCTION member_type end()   const;
    };

Parameters
----------

General Template Arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~

Valid template arguments for ``RangePolicy`` are described `here <../Execution-Policies.rst#common-arguments-for-all-execution-policies>`_

Public Class Members
--------------------

Constructors
~~~~~~~~~~~~

.. cpp:function:: explicit ChunkSize(int value_)

   Provide a hint for optimal chunk-size to be used during scheduling.
   For the SYCL backend, the workgroup size used in a ``parallel_for`` kernel can be set via this passed to ``RangePolicy``.

   .. note:: ``ChunkSize`` constructor ``explicit`` since Kokkos 4.4

.. cpp:function:: RangePolicy()

   Default Constructor uninitialized policy.

.. cpp:function:: RangePolicy(int64_t begin, int64_t end)

   Provide a start and end index.

.. cpp:function:: RangePolicy(int64_t begin, int64_t end, ChunkSize chunk_size)

   Provide a start and end index as well as a ``ChunkSize``.

.. cpp:function:: RangePolicy(const ExecutionSpace& space, int64_t begin, int64_t end)

   Provide a start and end index and an ``ExecutionSpace`` instance to use as the execution resource.

.. cpp:function:: RangePolicy(const ExecutionSpace& space, int64_t begin, int64_t end, ChunkSize chunk_size)

   Provide a start and end index and an ``ExecutionSpace`` instance to use as the execution resource, as well as a ``ChunkSize``.

Preconditions:
^^^^^^^^^^^^^^

* The start index must not be greater than the end index.
* The actual constructors are templated so we can check that they are converted to ``index_type`` safely (see `#6754 <https://github.com/kokkos/kokkos/pull/6754>`_).

CTAD Constructors (since 4.3):
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

   int64_t work_begin = /* ... */; // conversions as well
   int64_t work_end   = /* ... */; // conversions as well
   ChunkSize cs       = /* ... */; // conversions as well
   DefaultExecutionSpace des;      // conversions as well
   SomeExecutionSpace ses;         // different from DefaultExecutionSpace

   // Deduces to RangePolicy<>
   RangePolicy rp0;
   RangePolicy rp1(work_begin, work_end);
   RangePolicy rp2(work_begin, work_end, cs);
   RangePolicy rp3(des, work_begin, work_end);
   RangePolicy rp4(des, work_begin, work_end, cs);

   // Deduces to RangePolicy<SomeExecutionSpace>
   RangePolicy rp5(ses, work_begin, work_end);
   RangePolicy rp6(ses, work_begin, work_end, cs);

Examples
--------

.. code-block:: cpp

    RangePolicy<> policy_1(0, N);
    RangePolicy<Cuda> policy_2(5,N-5);
    RangePolicy<Schedule<Dynamic>, OpenMP> policy_3(n,m);
    RangePolicy<IndexType<int>, Schedule<Dynamic>> policy_4(0, K);
    RangePolicy<> policy_6(-3,N+3, ChunkSize(8));
    RangePolicy<OpenMP> policy_7(OpenMP(), 0, N, ChunkSize(4));

Note: providing a single integer as a policy to a parallel pattern, implies a defaulted ``RangePolicy``

.. code-block:: cpp

    // These two calls are identical
    parallel_for("Loop", N, functor);
    parallel_for("Loop", RangePolicy<>(0, N), functor);
