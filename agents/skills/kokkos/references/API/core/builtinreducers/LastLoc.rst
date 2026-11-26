``LastLoc``
===========

.. role:: cpp(code)
    :language: cpp

Specific implementation of `ReducerConcept <ReducerConcept.rst>`_ storing the last index satisfying a condition.

Header File: ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

   LastLoc<I,S>::value_type result;
   parallel_reduce(N,Functor,LastLoc<I,S>(result));

Synopsis
--------

.. code-block:: cpp

   template<class Index, class Space>
   class LastLoc{
     public:
       using reducer = LastLoc;
       using value_type = LastLocScalar<typename std::remove_cv<Index>::type>;
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
       LastLoc(value_type& value_);

       KOKKOS_INLINE_FUNCTION
       LastLoc(const result_view_type& value_);
   };

Interface
---------

.. cpp:class:: template<class Index, class Space> LastLoc

   .. rubric:: Public Types

   .. cpp:type:: reducer

      The self type.

   .. cpp:type:: value_type

      The reduction scalar type (specialization of `LastLocScalar <LastLocScalar.rst>`_)

   .. cpp:type:: result_view_type

      A ``Kokkos::View`` referencing the reduction result

   .. rubric:: Constructors

   .. cpp:function:: KOKKOS_INLINE_FUNCTION LastLoc(value_type& value_);

      Constructs a reducer which references a local variable as its result location.

   .. cpp:function:: KOKKOS_INLINE_FUNCTION LastLoc(const result_view_type& value_);

      Constructs a reducer which references a specific view as its result location.

   .. rubric:: Public Member Functions

   .. cpp:function:: KOKKOS_INLINE_FUNCTION void join(value_type& dest, const value_type& src) const;

      Store maximum with the last index of ``src`` and ``dest`` into ``dest``: ``dest = (src.max_loc_true > dest.max_loc_true) ? src :dest;``.

   .. cpp:function:: KOKKOS_INLINE_FUNCTION void init(value_type& val) const;

      Initialize ``val.max_loc_true`` using the ``Kokkos::reduction_identity<Index>::max()`` method. The default implementation sets ``val=<TYPE>_MIN``.

   .. cpp:function:: KOKKOS_INLINE_FUNCTION value_type& reference() const;

      Returns a reference to the result provided in the class constructor.

   .. cpp:function:: KOKKOS_INLINE_FUNCTION result_view_type view() const;

      Returns a view of the result provided in the class constructor.

Additional Information
^^^^^^^^^^^^^^^^^^^^^^

* ``LastLoc<I,S>::result_view_type`` is ``Kokkos::View<I,S,Kokkos::MemoryTraits<Kokkos::Unmanaged>>``. Note that the S (memory space) must be the same as the space where the result resides.

* Requires: ``Index`` has ``operator =`` defined. ``Kokkos::reduction_identity<Index>::max()`` is a valid expression.

* In order to use ``LastLoc`` with a custom type of ``Index``, a template specialization of ``Kokkos::reduction_identity<CustomType>`` must be defined. See `Built-In Reducers with Custom Scalar Types <../../../ProgrammingGuide/Custom-Reductions-Built-In-Reducers-with-Custom-Scalar-Types.rst>`_ for details
