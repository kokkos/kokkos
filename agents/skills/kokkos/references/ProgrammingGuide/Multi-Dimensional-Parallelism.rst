Multi-Dimensional Parallelism
=============================

.. _ParallelFor: ../API/core/parallel-dispatch/parallel_for.rst
.. |ParallelFor| replace:: ``parallel_for()``

.. _ParallelReduce: ../API/core/parallel-dispatch/parallel_reduce.rst
.. |ParallelReduce| replace:: ``parallel_reduce()``

.. _MDRangePolicy: ../API/core/policies/MDRangePolicy.rst
.. |MDRangePolicy| replace:: ``MDRangePolicy``

.. _RangePolicy: ../API/core/policies/RangePolicy.rst
.. |RangePolicy| replace:: ``RangePolicy``

This chapter explains how to leverage parallelism across more than one dimension.
If your problem exposes multiple dimensions of parallelism and does not require scratch memory,
you can use the |MDRangePolicy|_ to parallelize over several dimensions simultaneously.

``MDRangePolicy`` is best for tightly nested loops. For non-tightly-nested or loops requiring scratch memory,
prefer `TeamPolicy <../API/core/policies/TeamPolicy.rst>`_.

Use case example
----------------

This policy is particularly well-suited for operations on multidimensional arrays or tensor data.
A typical example arises when working on numerical methods for PDEs, such as finite element methods, where discretization of the domain results in ``C`` cells (elements), and basis
functions that are evaluated at ``P`` points, producing input and output whose rank and dimensions ``D`` depend on the field rank ``F`` of the basis function.

Problem formulation
~~~~~~~~~~~~~~~~~~~

**Input**:
  - ``inputData(C,P,D,D)`` - a rank 4 View
  - ``inputField(C,F,P,D)`` - a rank 4 View

**Return**:
  - ``outputField(C,F,P,D)`` - a rank 4 View

**Computation**:
  For each triple in ``C,F,P`` compute an output field from the two input views:

Serial implementation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

  for (int c = 0; c < C; ++c)
  for (int f = 0; f < F; ++f)
  for (int p = 0; p < P; ++p)
  {
    for (int i = 0; i < D; ++i) {
      double tmp(0);

      for (int j = 0; j < D; ++j)
        tmp += inputData(c, p, i, j) * inputField(c, f, p, j);  // compute the product

      outputField(c, f, p, i) = tmp;  // store the result
    }
  }

One dimension parallelization - ``RangePolicy``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The most straightforward way to parallelize the serial code above is to convert the outer ``for`` loop over cells with the sequential iteration pattern into a parallel for loop using a |RangePolicy|_

.. code-block:: cpp

   Kokkos::parallel_for("for_all_cells",
     Kokkos::RangePolicy<>(0,C),
      KOKKOS_LAMBDA (const int c) {
        for (int f = 0; f < F; ++f)
        for (int p = 0; p < P; ++p)
        {
         for (int i = 0; i < D; ++i) {

           double tmp(0);

           for (int j = 0; j < D; ++j)
             tmp += inputData(c, p, i, j) * inputField(c, f, p, j);

           outputField(c, f, p, i) = tmp;
         }
        }
     });


This works well if the number of cells is large enough to merit parallelization, that is, if the overhead for parallel dispatch plus computation time is less than total serial execution time, then this simple approach will already improve performance over the serial version.

However, there is more parallelism to exploit in the loops over fields ``F`` and points ``P``. This is especially important on device backends (GPU), which need a large number of concurrent work-items to hide memory latency.

One way to accomplish this would be to flatten the three iteration ranges into a single range of size ``C*F*P``, and perform a ``ParallelFor`` with ``RangePolicy`` over that product. But this would require extraction routines to map between the flat 1-D index (``C*F*P``) and the multidimensional ``(C,F,P)`` indices required by data structures. In addition, to be performance portable the mapping must be architecture-aware, akin to the notion of `LayoutLeft <../API/core/view/layoutLeft.rst>`_ and `LayoutRight <../API/core/view/layoutRight.rst>`_ used in Kokkos to establish data access patterns.

Multi-dimensional parallelization - ``MDRangePolicy``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The |MDRangePolicy|_ provides a natural way to accomplish the goal of parallelizing over all three iteration ranges without requiring manually computing the product of the iteration ranges and mapping between 1-D and 3-D multidimensional indices. The ``MDRangePolicy`` is suitable for use with tightly-nested for loops and provides a method to expose additional parallelism in computations beyond simply parallelizing in a single dimension, as was shown in the first implementation using the ``RangePolicy``.

.. code-block:: cpp

   Kokkos::parallel_for("mdr_for_all_cells",
     Kokkos::MDRangePolicy< Kokkos::Rank<3> > ({0,0,0}, {C,F,P}),
      KOKKOS_LAMBDA (const int c, const int f, const int p) {
       for (int i = 0; i < D; ++i) {

         double tmp(0);

         for (int j = 0; j < D; ++j)
           tmp += inputData(c, p, i, j) * inputField(c, f, p, j);

         outputField(c, f, p, i) = tmp;
       }
     });

MDRangePolicy usage
-------------------

The |MDRangePolicy|_ defines an execution policy for a multidimensional iteration space and can be used with both the |ParallelFor|_ and |ParallelReduce|_ patterns in Kokkos. The iteration space is defined by a tuple of "begin" indices and a tuple of "end" indices, which are provided as arguments to the |MDRangePolicy|_ constructor.

Number of dimensions (``Kokkos::Rank``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The |MDRangePolicy|_ accepts the same template parameters as the |RangePolicy|_, but also requires an additional type - the :cpp:class:`Kokkos::Rank` parameter, where ``R`` is the rank, that is the number of nested for-loops, and must be provided at compile-time.

Index arguments
~~~~~~~~~~~~~~~

The policy requires two arguments:

1. An initializer list, or :cpp:struct:`Kokkos::Array`, of "begin" indices
2. An initializer list, or :cpp:struct:`Kokkos::Array`, of "end" indices

.. code-block:: cpp

  // Using initializer lists
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> mdrange_policy({0, 0, 0}, {C, F, P});
  
  // Using Kokkos::Array
  Kokkos::Array<int64_t, 3> begin{0, 0, 0};
  Kokkos::Array<int64_t, 3> end{C, F, P};
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> mdrange_policy(begin, end);

The lambda (or functor's ``operator()``) must take one integer argument per rank of the policy.

.. code-block:: cpp

  // With rank 3, the lambda requires 3 arguments
  KOKKOS_LAMBDA(const int c, const int f, const int p) {
    // body of the lambda
  }

  // Agnostic rank functor example
  struct AddFunctor {
    // ...
    template<std::integral... Args>
    KOKKOS_INLINE_FUNCTION
    void operator()(Args... args) const {
      view_c(args...) = view_a(args...) + view_b(args...);
    }
  };


.. _MDRangePolicy-Iteration-order:

Specifying iteration order (``Kokkos::Iterate``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You may want to specify the iteration order of the |MDRangePolicy|_ to match the memory layout of your data for best performance.
The iteration order is specified with the :cpp:class:`Kokkos::Rank` template parameter, which accepts two optional template parameters, ``outer`` and ``inner`` of type :cpp:enum:`Kokkos::Iterate`.

.. code-block:: cpp

  // One mandatory template parameter is the rank of the iteration space
  Kokkos::Rank<3>;
  // Optionally, you can specify the iteration order
  Kokkos::Rank<3, Kokkos::Iterate::Default, Kokkos::Iterate::Default>;
  
  // Then use it inside the MDRangePolicy template parameters
  Kokkos::MDRangePolicy<Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Right>>;

By default, the iteration order depends on the default execution space. 

By default, the iteration order and the data layout should match.
For example, if you configured Kokkos with ``CUDA`` the default iteration order is ``Kokkos::Iterate::Left`` and the default data layout is ``LayoutLeft``.
If you configured Kokkos with ``OpenMP`` the default iteration order is ``Kokkos::Iterate::Right`` and the default data layout is ``LayoutRight``.

.. note:: Match your iteration pattern to your data layout for best cache performance on host backends, and to ensure coalesced memory access on device backends.

Tiling Strategy
~~~~~~~~~~~~~~~

Internally the |MDRangePolicy|_ uses tiling over the multidimensional iteration space. For customization an optional third argument may be passed to the policy - an initializer list of tile dimension sizes. This argument might become important when performance tuning, as simple default sizes can be problem-dependent and difficult to determine automatically.

.. code-block:: cpp

  // Tiling dimensions can be specified with an initializer list as the third argument to the MDRangePolicy constructor
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> mdrange_policy({0, 0, 0}, {C, F, P}, {16, 4, 4});

Playing with tile size is a common way to optimize performance when using ``MDRangePolicy``.
The optimal tile size is dependent on the kernel being run, the data being accessed, and the hardware it is running on.
You can query the default tile sizes with the member function :cpp:func:`tile_size_recommended`.

* **Device backends** (``CUDA``, ``HIP``, ``SYCL``): tile sizes are used to determine the number of work-items per work-group.
  Their size is limited by the underlying hardware; you can query the upper limit on the total tile size with :cpp:func:`max_total_tile_size`.

* **Host backends** (``Serial``, ``OpenMP``, ``Threads``): there are no limitations on tile sizes.

References
----------

The use case that this example is based on comes from the `Intrepid2 <https://trilinos.github.io/docs/intrepid2/index.html>`_ package of Trilinos. For more examples, check out code in Trilinos in files at: ``Trilinos/packages/intrepid2/src/Shared/Intrepid2_ArrayToolsDef*.hpp``.
