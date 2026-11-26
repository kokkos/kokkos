``BOr``
=======

.. role:: cpp(code)
    :language: cpp

Specific implementation of `ReducerConcept <ReducerConcept.rst>`_ performing bitwise ``OR`` operation

Header File: ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

   T result;
   parallel_reduce(N, Functor, BOr<T, S>(result));

Synopsis
--------

.. code-block:: cpp

   template<class Scalar, class Space>
   class BOr {
     public:
       using reducer = BOr<Scalar, Space>;
       using value_type = typename std::remove_cv<Scalar>::type;

       KOKKOS_INLINE_FUNCTION
       void join(value_type& dest, const value_type& src) const {
         dest = dest | src;
       }

       KOKKOS_INLINE_FUNCTION
       void init(value_type& val) const {
         val = Kokkos::reduction_identity<value_type>::bor();
       }

       // other members to fulfill the ReducerConcept
   };

Interface
---------

All the public types, constructors and methods from `ReducerConcept <ReducerConcept.rst>`_ are available. The following types and methods are overridden by this reducer:

.. cpp:class:: template<class Scalar, class Space> BOr

   .. rubric:: Public Types

   .. cpp:type:: reducer

      The self type.

   .. cpp:type:: value_type

      The ``Scalar`` template parameter stripped of its potential ``const`` and/or ``volatile`` qualifier.

   .. rubric:: Public Member Functions

   .. cpp:function:: KOKKOS_INLINE_FUNCTION void join(value_type& dest, const value_type& src) const;

      Store logical ``or`` of ``src`` and ``dest`` into ``dest``:  ``dest = src | dest``.

   .. cpp:function:: KOKKOS_INLINE_FUNCTION void init(value_type& val) const;

      Initialize ``val`` using the ``Kokkos::reduction_identity<value_type>::bor()`` method. The default implementation sets ``val=0x0``.

Additional Information
^^^^^^^^^^^^^^^^^^^^^^

* Requires: ``value_type`` has ``operator =`` and ``operator |`` defined. ``Kokkos::reduction_identity<value_type>::bor()`` is a valid expression.

* In order to use ``BOr`` with a custom type, a template specialization of ``Kokkos::reduction_identity<CustomType>`` must be defined. See `Built-In Reducers with Custom Scalar Types <../../../ProgrammingGuide/Custom-Reductions-Built-In-Reducers-with-Custom-Scalar-Types.rst>`_ for details
