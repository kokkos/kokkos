``MinMaxFirstLastLoc``
======================

.. role:: cpp(code)
    :language: cpp

Specific implementation of `ReducerConcept <ReducerConcept.rst>`_ storing both the minimum and maximum values with corresponding first and last indices

Header File: ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

   MinMaxFirstLastLoc<T,I,S>::value_type result;
   parallel_reduce(N,Functor,MinMaxFirstLastLoc<T,I,S>(result));

Synopsis
--------

.. code-block:: cpp

   template<class Scalar, class Index, class Space>
   class MinMaxFirstLastLoc {
     public:
       using reducer = MinMaxFirstLastLoc;
       using value_type = MinMaxFirstLastLocScalar<typename std::remove_cv<Scalar>::type,
                                        typename std::remove_cv<Index>::type>;

       using result_view_type = Kokkos::View<value_type, Space>;

       KOKKOS_INLINE_FUNCTION
       void join(value_type& dest, const value_type& src) const;

       KOKKOS_INLINE_FUNCTION
       void init(value_type& val) const;

       KOKKOS_INLINE_FUNCTION
       value_type& reference() const;

       KOKKOS_INLINE_FUNCTION
       result_view_type view() const;

       KOKKOS_INLINE_FUNCTION
       MinMaxFirstLastLoc(value_type& value_);

       KOKKOS_INLINE_FUNCTION
       MinMaxFirstLastLoc(const result_view_type& value_);
   };

Interface
---------

.. cpp:class:: template<class Scalar, class Index, class Space> MinMaxFirstLastLoc

   .. rubric:: Public Types

   .. cpp:type:: reducer

      The self type

   .. cpp:type:: value_type

      The reduction scalar type (specialization of `MinMaxLocScalar <MinMaxLocScalar.rst>`_)

   .. cpp:type:: result_view_type

      A ``Kokkos::View`` referencing the reduction result

   .. rubric:: Constructors

   .. cpp:function:: KOKKOS_INLINE_FUNCTION MinMaxFirstLastLoc(value_type& value_);

      Constructs a reducer which references a local variable as its result location.

   .. cpp:function:: KOKKOS_INLINE_FUNCTION MinMaxFirstLastLoc(const result_view_type& value_);

      Constructs a reducer which references a specific view as its result location.

   .. rubric:: Public Member Functions

   .. cpp:function:: KOKKOS_INLINE_FUNCTION void join(value_type& dest, const value_type& src) const;

      Store minimum with the first location of ``src`` and ``dest`` into ``dest``.
      Store maximum with the last location of ``src`` and ``dest`` into ``dest``.

   .. cpp:function:: KOKKOS_INLINE_FUNCTION void init( value_type& val) const;

      Initialize ``val.min_val`` using the ``Kokkos::reduction_identity<Scalar>::min()`` method. The default implementation sets ``val=<TYPE>_MAX``.

      Initialize ``val.max_val`` using the ``Kokkos::reduction_identity<Index>::max()`` method. The default implementation sets ``val=<TYPE>_MIN``.

      Initialize ``val.min_loc`` using the ``Kokkos::reduction_identity<Scalar>::min()`` method. The default implementation sets ``val=<TYPE>_MAX``.

      Initialize ``val.max_loc`` using the ``Kokkos::reduction_identity<Index>::max()`` method. The default implementation sets ``val=<TYPE>_MAX``.

   .. cpp:function:: KOKKOS_INLINE_FUNCTION value_type& reference() const;

      Returns a reference to the result provided in the class constructor.

   .. cpp:function:: KOKKOS_INLINE_FUNCTION result_view_type view() const;

      Returns a view of the result provided in the class constructor.

Additional Information
^^^^^^^^^^^^^^^^^^^^^^

* ``MinMaxFirstLastLoc<T,I,S>::result_view_type`` is ``Kokkos::View<T,S,Kokkos::MemoryTraits<Kokkos::Unmanaged>>``. Note that the S (memory space) must be the same as the space where the result resides.

* Requires: ``Scalar`` has ``operator =``, ``operator <`` and ``operator >`` defined. ``Kokkos::reduction_identity<Scalar>::min()`` and ``Kokkos::reduction_identity<Scalar>::max()`` are a valid expressions.

* Requires: ``Index`` has ``operator =`` defined. ``Kokkos::reduction_identity<Index>::min()`` is a valid expressions.

* In order to use ``MinMaxFirstLastLoc`` with a custom type of either ``Scalar`` or ``Index``, a template specialization of ``Kokkos::reduction_identity<CustomType>`` must be defined. See `Built-In Reducers with Custom Scalar Types <../../../ProgrammingGuide/Custom-Reductions-Built-In-Reducers-with-Custom-Scalar-Types.rst>`_ for details.
