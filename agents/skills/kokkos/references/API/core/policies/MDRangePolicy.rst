``MDRangePolicy``
=================

.. role:: cpp(code)
    :language: cpp

Header File: ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

    Kokkos::MDRangePolicy<..., Rank<N>, ...>(begin, end)
    Kokkos::MDRangePolicy<..., Rank<N>, ...>(Space, begin, end)
    Kokkos::MDRangePolicy<..., Rank<N>, ...>(begin, end, tiling)
    Kokkos::MDRangePolicy<..., Rank<N>, ...>(Space, begin, end, tiling)

``MDRangePolicy`` defines an execution policy for a multidimensional iteration space starting at a ``begin`` tuple and going to ``end`` with an open interval. The iteration space will be tiled, and the user can optionally provide tiling sizes.

Interface
---------

.. code-block:: cpp

    template<class ... Args>
    class Kokkos::MDRangePolicy;

Parameters
----------

General Template Arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~

Valid template arguments for ``MDRangePolicy`` are described `here <../Execution-Policies.rst#common-arguments-for-all-execution-policies>`_.

Required Argument Specific to MDRangePolicy - ``Kokkos::Rank``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Interface
^^^^^^^^^
.. code-block:: cpp

    template<int N,
             Kokkos::Iterate outer = Kokkos::Iterate::Default,
             Kokkos::Iterate inner = Kokkos::Iterate::Default>
    class Kokkos::Rank;

``Kokkos::Rank`` is a required template argument unique to ``MDRangePolicy``. It specifies the rank of the iteration space and, optionally, the iteration order over and within tiles.

``outer`` and ``inner`` default to ``Kokkos::Iterate::Default`` and can be set to ``Kokkos::Iterate::Left`` or ``Kokkos::Iterate::Right``.

Template Arguments
^^^^^^^^^^^^^^^^^^

.. cpp:class:: template<int N, Kokkos::Iterate outer, Kokkos::Iterate inner> Kokkos::Rank;

   :tparam N: Rank of the iteration space (1 to 6).

   .. note:: Rank 1 is supported since Kokkos 5.2.

   :tparam outer: Iteration order over tiles (optional).
   :tparam inner: Iteration order within each tile (optional).

.. cpp:enum-class:: Kokkos::Iterate

   .. cpp:enumerator:: Kokkos::Iterate::Default

      Use the natural iteration order for the execution space.

   .. cpp:enumerator:: Kokkos::Iterate::Left

      Column-major: leftmost index varies fastest.

   .. cpp:enumerator:: Kokkos::Iterate::Right

      Row-major: rightmost index varies fastest.

.. note::

   For best performance, match the iteration order to your View's memory layout. See :ref:`Iteration Order <MDRangePolicy-Iteration-order>` in the Programming Guide.

Public Class Members
--------------------

Constructors
~~~~~~~~~~~~

.. cpp:function:: MDRangePolicy()

    * Default Constructor uninitialized policy.

.. cpp:function:: MDRangePolicy(const Kokkos::Array<int64_t,rank>& begin, const Kokkos::Array<int64_t,rank>& end)

    * Provide a start and end index.

.. cpp:function:: MDRangePolicy(const Kokkos::Array<int64_t,rank>& begin, const Kokkos::Array<int64_t,rank>& end,  const Kokkos::Array<int64_t,rank>& tiling)

    * Provide a start and end index as well as the tiling dimensions.

.. cpp:function:: template<class OT, class IT> MDRangePolicy(const std::initializer_list<OT>& begin, const std::initializer_list<IT>& end)

    * Provide a start and end index. The length of the lists must match the rank of the policy.

.. cpp:function:: template<class OT, class IT, class TT> MDRangePolicy(const std::initializer_list<OT>& begin, const std::initializer_list<IT>& end, const std::initializer_list<TT>& tiling)

    * Provide a start and end index as well as the tiling dimensions. The length of the lists must match the rank of the policy.

CTAD Constructors (since 4.3)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

   DefaultExecutionSpace des;
   SomeExecutionSpace ses; // different from DefaultExecutionSpace

   // Deduces to MDRangePolicy<Rank<3>>
   MDRangePolicy pl0({0, 0, 0}, {4, 5, 10});
   MDRangePolicy pl1({0, 0, 0}, {4, 5, 10}, {3, 3, 3});

   // Deduces to MDRangePolicy<SomeExecutionSpace, Rank<3>>
   MDRangePolicy pl2(ses, {0, 0, 0}, {4, 5, 10});
   MDRangePolicy pl3(ses, {0, 0, 0}, {4, 5, 10}, {3, 3, 3});

   int cbegin[3];
   int cend[3];
   int64_t ctiling[3];

   // Deduces to MDRangePolicy<Rank<3>>
   MDRangePolicy pc0(cbegin, cend);
   MDRangePolicy pc1(cbegin, cend, ctiling);
   MDRangePolicy pc2(des, cbegin, cend);
   MDRangePolicy pc3(des, cbegin, cend, ctiling);

   // Deduces to MDRangePolicy<SomeExecutionSpace, Rank<3>>
   MDRangePolicy pc4(ses, cbegin, cend);
   MDRangePolicy pc5(ses, cbegin, cend, ctiling);

   Array<int, 2> abegin;
   Array<int, 2> aend;
   Array<int, 2> atiling;

   // Deduces to MDRangePolicy<Rank<2>>
   MDRangePolicy pa0(abegin, aend);
   MDRangePolicy pa1(abegin, aend, atiling);
   MDRangePolicy pa2(des, abegin, aend);
   MDRangePolicy pa3(des, abegin, aend, atiling);

   // Deduces to MDRangePolicy<SomeExecutionSpace, Rank<2>>
   MDRangePolicy pa4(ses, abegin, aend);
   MDRangePolicy pa5(ses, abegin, aend, atiling);

Member Functions
~~~~~~~~~~~~~~~~
.. cpp:function:: tile_type tile_size_recommended() const

    * Returns a ``Kokkos::Array<array_index_type, rank>`` type containing per-rank tile sizes that ``MDRangePolicy`` internally uses by default. The default tile sizes are static and are set based on the specified backend.

    .. note:: ``tile_size_recommended()`` available since Kokkos 4.5

.. cpp:function:: int max_total_tile_size() const

    * Returns a value that represents the upper limit for the product of all tile sizes.

    .. note:: ``max_total_tile_size()`` available since Kokkos 4.5

Notes
~~~~~

* The start index must not be greater than the matching end index for all ranks.
* The begin and end array ranks must match.
* The tiling array rank must be less than or equal to the begin/end array rank.

Examples
--------

.. code-block:: cpp

    MDRangePolicy<Rank<3>> policy_1({0,0,0},{N0,N1,N2});
    MDRangePolicy<Cuda,Rank<3,Iterate::Right,Iterate::Left>> policy_2({5,5,5},{N0-5,N1-5,N2-5},{T0,T1,T2});
