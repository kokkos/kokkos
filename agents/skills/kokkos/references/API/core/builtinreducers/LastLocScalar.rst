``LastLocScalar``
=================

.. role::cpp(code)
    :language: cpp

The :cpp:struct:`LastLocScalar` is a class template that stores a **location** (index) of the last occurrence satisfying a condition as a single, convenient unit.
It's designed to hold the result of :cpp:func:`parallel_reduce` operations using a :cpp:class:`LastLoc` builtin reducer.

It is generally recommended to get this type by using the reducer's
``::value_type`` member (e.g., ``LastLoc<Index,Space>::value_type``)
to ensure the correct template parameters are used.

Header File: ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

   LastLocScalar<Index>::value_type result;
   parallel_reduce(N,Functor,LastLocScalar<Index>(result));
   I lastLoc = result.max_loc_true;

Interface
---------

.. cpp:struct::  template<class Index> LastLocScalar

   :tparam Index: The data type of the locations (indices) of the values.

   .. rubric:: Data members

   .. cpp:var:: Index max_loc_true

      The location (iteration index) of the maximum value
