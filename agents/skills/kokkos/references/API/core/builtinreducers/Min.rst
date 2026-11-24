``Min``
=======

.. role:: cpp(code)
    :language: cpp

Specific implementation of `ReducerConcept <ReducerConcept.rst>`_ storing the minimum value

Header File: ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

   T result;
   parallel_reduce(N,Functor,Min<T,S>(result));

Synopsis
--------

.. code-block:: cpp

   template<class Scalar, class Space>
   class Min{
     public:
       typedef Min reducer;
       typedef typename std::remove_cv<Scalar>::type value_type;
       typedef Kokkos::View<value_type, Space> result_view_type;

       KOKKOS_INLINE_FUNCTION
       void join(value_type& dest, const value_type& src) const;

       KOKKOS_INLINE_FUNCTION
       void init(value_type& val) const;

       KOKKOS_INLINE_FUNCTION
       value_type& reference() const;

       KOKKOS_INLINE_FUNCTION
       result_view_type view() const;

       KOKKOS_INLINE_FUNCTION
       Min(value_type& value_);

       KOKKOS_INLINE_FUNCTION
       Min(const result_view_type& value_);
   };

Interface
---------

.. cpp:class:: template<class Scalar, class Space> Min

   .. rubric:: Public Types

   .. cpp:type:: reducer

      The self type

   .. cpp:type:: value_type

      The reduction scalar type.

   .. cpp:type:: result_view_type

      A ``Kokkos::View`` referencing the reduction result

   .. rubric:: Constructors

   .. cpp:function:: KOKKOS_INLINE_FUNCTION Min(value_type& value_);

      Constructs a reducer which references a local variable as its result location.

   .. cpp:function:: KOKKOS_INLINE_FUNCTION Min(const result_view_type& value_);

      Constructs a reducer which references a specific view as its result location.

   .. rubric:: Public Member Functions

   .. cpp:function:: KOKKOS_INLINE_FUNCTION void join(value_type& dest, const value_type& src) const;

      Store minimum of ``src`` and ``dest`` into ``dest``:  ``dest = (src < dest) ? src :dest;``.

   .. cpp:function:: KOKKOS_INLINE_FUNCTION void init(value_type& val) const;

      Initialize ``val`` using the ``Kokkos::reduction_identity<Scalar>::min()`` method. The default implementation sets ``val=<TYPE>_MAX``.

   .. cpp:function:: KOKKOS_INLINE_FUNCTION value_type& reference() const;

      Returns a reference to the result provided in class constructor.

   .. cpp:function:: KOKKOS_INLINE_FUNCTION result_view_type view() const;

      Returns a view of the result place provided in class constructor.

Additional Information
^^^^^^^^^^^^^^^^^^^^^^

* ``Min<T,S>::value_type`` is non-const ``T``

* ``Min<T,S>::result_view_type`` is ``Kokkos::View<T,S,Kokkos::MemoryTraits<Kokkos::Unmanaged>>``. Note that the S (memory space) must be the same as the space where the result resides.

* Requires: ``Scalar`` has ``operator =`` and ``operator <`` defined. ``Kokkos::reduction_identity<Scalar>::min()`` is a valid expression.

* In order to use Min with a custom type, a template specialization of ``Kokkos::reduction_identity<CustomType>`` must be defined. See `Built-In Reducers with Custom Scalar Types <../../../ProgrammingGuide/Custom-Reductions-Built-In-Reducers-with-Custom-Scalar-Types.rst>`_ for details
