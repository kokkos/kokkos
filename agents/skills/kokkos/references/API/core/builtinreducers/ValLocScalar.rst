``ValLocScalar``
================

.. role::cpp(code)
    :language: cpp

:cpp:struct:`ValLocScalar` is a class template that stores a **value** and its
corresponding **location** (index) as a single, convenient unit. It is
primarily designed to hold the result of :cpp:func:`parallel_reduce` operations
using the :cpp:class:`MinLoc` and :cpp:class:`MaxLoc` builtin reducers.

It is generally recommended to get this type by using the reducer's
``::value_type`` member (e.g., ``MaxLoc<Scalar,Index,Space>::value_type``) to
ensure the correct template parameters are used.

Header File: ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

    MaxLoc<T,I,S>::value_type result;
    parallel_reduce(N,Functor,MaxLoc<T,I,S>(result));
    T resultValue = result.val;
    I resultIndex = result.loc;

Interface
---------

.. cpp:struct::  template<class Scalar, class Index> ValLocScalar

   :tparam Scalar: The data type of the value being reduced (e.g., ``double``, ``int``).

   :tparam Index: The data type of the location or iteration index (e.g., ``int``, ``long long``).

   .. rubric:: Data members

   .. cpp:var:: Scalar val

      The reduced value.

   .. cpp:var:: Index loc

      The location (iteration index) of the reduced value
