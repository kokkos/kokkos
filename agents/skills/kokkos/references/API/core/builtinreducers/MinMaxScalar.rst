``MinMaxScalar``
================

.. role::cpp(code)
    :language: cpp

:cpp:struct:`MinMaxScalar` is a class template that stores both a minimum and a
maximum value as a single, convenient unit. It is primarily designed to hold
the result of :cpp:func:`parallel_reduce` operations using the
:cpp:class:`MinMax` builtin reducer.

It is generally recommended to get this type by using the reducer's
``::value_type`` member (e.g., ``MinMax<Scalar,Space>::value_type``) to ensure
the correct template parameters are used.

Header File: ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

   MinMax<T,S>::value_type result;
   parallel_reduce(N,Functor,MinMax<T,S>(result));
   T minValue = result.min_val;
   T maxValue = result.max_val;

Interface
---------

.. cpp:struct::  template<class Scalar> MinMaxScalar

   :tparam Scalar: The data type of the value being reduced (e.g., ``double``, ``int``).

   .. rubric:: Data members

   .. cpp:var:: Scalar min_val

      The reduced minimum value.

   .. cpp:var:: Scalar max_val

      The reduced maximum value.
