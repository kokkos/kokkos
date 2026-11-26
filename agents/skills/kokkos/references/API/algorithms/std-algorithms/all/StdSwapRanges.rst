
``swap_ranges``
===============

Header: ``<Kokkos_StdAlgorithms.hpp>``

Description
-----------

Swaps the elements between two ranges or two rank-1 ``View``

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.


Overload set accepting execution space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <class ExecutionSpace, class IteratorType1, class IteratorType2>
   IteratorType2 swap_ranges(const ExecutionSpace& ex, IteratorType1 first1,          (1)
                             IteratorType1 last1, IteratorType2 first2);

   template <class ExecutionSpace, class DataType1, class... Properties1,
             class DataType2, class... Properties2>
   auto swap_ranges(const ExecutionSpace& ex,                                         (2)
                    const ::Kokkos::View<DataType1, Properties1...>& source,
                    ::Kokkos::View<DataType2, Properties2...>& dest);

   template <class ExecutionSpace, class IteratorType1, class IteratorType2>
   IteratorType2 swap_ranges(const std::string& label, const ExecutionSpace& ex,      (3)
                             IteratorType1 first1, IteratorType1 last1,
                             IteratorType2 first2);

   template <class ExecutionSpace, class DataType1, class... Properties1,
             class DataType2, class... Properties2>
   auto swap_ranges(const std::string& label, const ExecutionSpace& ex,               (4)
                    const ::Kokkos::View<DataType1, Properties1...>& source,
                    ::Kokkos::View<DataType2, Properties2...>& dest);

Overload set accepting a team handle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 4.2

.. code-block:: cpp

   template <class TeamHandleType, class IteratorType1, class IteratorType2>
   KOKKOS_FUNCTION
   IteratorType2 swap_ranges(const TeamHandleType& teamHandle, IteratorType1 first1,  (5)
                             IteratorType1 last1, IteratorType2 first2);

   template <class TeamHandleType, class DataType1, class... Properties1,
             class DataType2, class... Properties2>
   KOKKOS_FUNCTION
   auto swap_ranges(const TeamHandleType& teamHandle,                                 (6)
                    const ::Kokkos::View<DataType1, Properties1...>& source,
                    ::Kokkos::View<DataType2, Properties2...>& dest);


Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``exespace``: execution space instance

- ``teamHandle``: team handle instance given inside a parallel region when using a TeamPolicy

- ``label``: string forwarded to internal parallel kernels for debugging purposes

  - for 1, the default string is: "Kokkos::swap_ranges_iterator_api_default"

  - for 2, the default string is: "Kokkos::swap_ranges_view_api_default"

  - NOTE: overloads accepting a team handle do not use a label internally

- ``first1``, ``last1``, ``first2``: iterators defining the ranges to swap

  - must be *random access iterator*

  - must represent a valid range, i.e., ``last1 >= first1``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``source``, ``dest``: views to swap

  - must be rank-1, and have ``LayoutLeft``, ``LayoutRight``, or ``LayoutStride``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle


Return Value
~~~~~~~~~~~~

- 1,3,5: an iterator equal to ``first2 + Kokkos::Experimental::distance(first1, last1)``

- 2,4,6: an iterator equal to
  ``Kokkos::Experimental::begin(dest) +
  Kokkos::Experimental:distance(Kokkos::Experimental::begin(source), Kokkos::Experimental::end(source))``
