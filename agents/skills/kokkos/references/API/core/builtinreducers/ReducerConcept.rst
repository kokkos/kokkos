``ReducerConcept``
==================

.. role:: cpp(code)
    :language: cpp

The concept of a Reducer is the abstraction that defines the "how" a "Reduction" is performed during the parallel reduce execution pattern. The abstraction of "what" is given as a template parameter and corresponds to the "what" that is being reduced in the `parallel_reduce <../parallel-dispatch/parallel_reduce.rst>`_ operation. This page describes the definitions and functions expected from a Reducer with a hypothetical 'Reducer' class definition. A brief description of built-in reducers is also included.

Header File: ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

    T result;
    parallel_reduce(N,Functor,ReducerConcept<T>(result));


Synopsis
--------

.. code-block:: cpp

    class Reducer {
        public:
            //Required for Concept
            typedef Reducer reducer;
            typedef ... value_type;
            typedef Kokkos::View<value_type, ... > result_view_type;

            KOKKOS_INLINE_FUNCTION
            void join(value_type& dest, const value_type& src) const;

            KOKKOS_INLINE_FUNCTION
            value_type& reference() const;

            KOKKOS_INLINE_FUNCTION
            result_view_type view() const;

            //Optional
            KOKKOS_INLINE_FUNCTION
            void init(value_type& val) const;

            KOKKOS_INLINE_FUNCTION
            void final(value_type& val) const;

            //Part of Build-In reducers for Kokkos
            KOKKOS_INLINE_FUNCTION
            Reducer(value_type& value_);

            KOKKOS_INLINE_FUNCTION
            Reducer(const result_view_type& value_);
    };

Public Class Members
--------------------

Typedefs
~~~~~~~~

* ``reducer``: The self type.
* ``value_type``: The reduction scalar type.
* ``result_view_type``: A ``Kokkos::View`` referencing where the reduction result should be placed. Can be an unmanaged view of a scalar or complex datatype (class or struct). Unmanaged views must specify the same memory space where the referenced scalar (or complex datatype) resides.

Constructors
~~~~~~~~~~~~

Constructors are not part of the concept. A custom reducer can have complex custom constructors. All Build-In reducers in Kokkos have the following two constructors:

.. cpp:function:: KOKKOS_INLINE_FUNCTION Reducer(value_type& value_);

    * Constructs a reducer which references a local variable as its result location.

.. cpp:function:: KOKKOS_INLINE_FUNCTION Reducer(const result_view_type& value_);

    * Constructs a reducer which references a specific view as its result location.

Functions
~~~~~~~~~

.. cpp:function:: KOKKOS_INLINE_FUNCTION void join(value_type& dest, const value_type& src) const;

    * Combine ``src`` into ``dest``. For example, ``Add`` performs ``dest+=src;``.

.. cpp:function:: KOKKOS_INLINE_FUNCTION void init(value_type& val) const;

    * Optional callback initializing ``val`` with appropriate initial value. For example, 'Add' assigns ``val = 0;``, but Prod assigns ``val = 1;``.
      Defaults to calling the default constructor.

.. cpp:function:: KOKKOS_INLINE_FUNCTION void final(value_type& val) const;

    * Optional callback modifying the result ``val``. Defaults to a no-op.

.. cpp:function:: KOKKOS_INLINE_FUNCTION value_type& reference() const;

    * Returns a reference to the result place.

.. cpp:function:: KOKKOS_INLINE_FUNCTION result_view_type view() const;

    * Returns a view of the result place.

Requirements
~~~~~~~~~~~~

The reducer is assumed to define a commutative monoid with respect to the value type it is used with, i.e., the binary operation

.. code-block:: cpp

    value_type op(const value_type& val1, const value_type& val2) {
      value_type result = val1;
      reducer.join(result, val2);
      return result;
    }

is commutative and associative with identity element that can be set by calling ``reducer.init(el)``.


Built-In Reducers
~~~~~~~~~~~~~~~~~

Kokkos provides a number of built-in reducers that automatically work with the intrinsic C++ types as well as ``Kokkos::complex``. In order to use a Built-in reducer with a custom type, a template specialization of ``Kokkos::reduction_identity<CustomType>`` must be defined. A simple example is shown below and more information can be found under `Custom Reductions <../../../ProgrammingGuide/Custom-Reductions.rst>`_.

* `Kokkos::BAnd <BAnd.rst>`_
* `Kokkos::BOr <BOr.rst>`_
* `Kokkos::LAnd <LAnd.rst>`_
* `Kokkos::LOr <LOr.rst>`_
* `Kokkos::Max <Max.rst>`_
* `Kokkos::MaxLoc <MaxLoc.rst>`_
* `Kokkos::Min <Min.rst>`_
* `Kokkos::MinLoc <MinLoc.rst>`_
* `Kokkos::MinMax <MinMax.rst>`_
* `Kokkos::MinMaxLoc <MinMaxLoc.rst>`_
* `Kokkos::Prod <Prod.rst>`_
* `Kokkos::Sum <Sum.rst>`_

Examples
--------

.. code-block:: cpp

    #include<Kokkos_Core.hpp>

    int main(int argc, char* argv[]) {

        long N = argc>1 ? atoi(argv[1]):100;
        long result;
        Kokkos::parallel_reduce("ReduceSum: ", N, KOKKOS_LAMBDA (const int i, long& lval) {
            lval += i;
        }, Sum<long>(result));

        printf("Result: %l Expected: %l\n",result,N*(N-1)/2);
    }
