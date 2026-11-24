``FirstLocScalar``
==================

.. role::cpp(code)
    :language: cpp

The :cpp:struct:`FirstLocScalar` is a class template that stores a **location** (index) of the first occurrence satisfying a condition as a single, convenient unit.
It's designed to hold the result of :cpp:func:`parallel_reduce` operations using a :cpp:class:`FirstLoc` builtin reducer.

It is generally recommended to get this type by using the reducer's
``::value_type`` member (e.g., ``FirstLoc<Index,Space>::value_type``)
to ensure the correct template parameters are used.

Header File: ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

   FirstLocScalar<Index>::value_type result;
   parallel_reduce(N,Functor,FirstLocScalar<Index>(result));
   I firstLoc = result.min_loc_true;

Interface
---------

.. cpp:struct::  template<class Index> FirstLocScalar

   :tparam Index: The data type of the locations (indices) of the values.

   .. rubric:: Data members

   .. cpp:var:: Index min_loc_true

      The location (iteration index) of the minimum value
