``is_sorted_until``
===================

Header: ``Kokkos_StdAlgorithms.hpp``

Description
-----------

Finds the largest range beginning at ``first`` or at ``Kokkos::Experimental::begin(view)`` in which the elements are sorted in non-descending order. Comparison between elements is done via ``operator<`` or the binary functor ``comp``.

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.

Overload set accepting execution space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <class ExecutionSpace, class IteratorType>
   IteratorType is_sorted_until(const ExecutionSpace& exespace,                     (1)
                                IteratorType first, IteratorType last);

   template <class ExecutionSpace, class IteratorType>
   IteratorType is_sorted_until(const std::string& label,                           (2)
                                const ExecutionSpace& exespace,
                                IteratorType first, IteratorType last);

   template <class ExecutionSpace, class IteratorType, class ComparatorType>
   IteratorType is_sorted_until(const ExecutionSpace& exespace,                     (3)
                                IteratorType first, IteratorType last,
                                ComparatorType comp);

   template <class ExecutionSpace, class IteratorType, class ComparatorType>
   IteratorType is_sorted_until(const std::string& label,                           (4)
                                const ExecutionSpace& exespace,
                                IteratorType first, IteratorType last,
                                ComparatorType comp);

   template <class ExecutionSpace, class DataType, class... Properties>
   auto is_sorted_until(const ExecutionSpace& exespace,                             (5)
                        const Kokkos::View<DataType, Properties...>& view);

   template <class ExecutionSpace, class DataType, class... Properties>
   auto is_sorted_until(const std::string& label,                                   (6)
                        const ExecutionSpace& exespace,
                        const Kokkos::View<DataType, Properties...>& view);

   template <
      class ExecutionSpace,
      class DataType, class... Properties, class ComparatorType>
   auto is_sorted_until(const ExecutionSpace& exespace,                             (7)
                        const Kokkos::View<DataType, Properties...>& view,
                        ComparatorType comp);

   template <
      class ExecutionSpace,
      class DataType, class... Properties, class ComparatorType>
   auto is_sorted_until(const std::string& label, const ExecutionSpace& exespace,   (8)
                        const Kokkos::View<DataType, Properties...>& view,
                        ComparatorType comp);

Overload set accepting a team handle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 4.2

.. code-block:: cpp

   template <class TeamHandleType, class IteratorType>
   KOKKOS_FUNCTION
   IteratorType is_sorted_until(const TeamHandleType& teamHandle,                   (9)
                                IteratorType first, IteratorType last);

   template <class TeamHandleType, class DataType, class... Properties>
   KOKKOS_FUNCTION
   auto is_sorted_until(const TeamHandleType& teamHandle,                           (10)
                        const Kokkos::View<DataType, Properties...>& view);

   template <class TeamHandleType, class IteratorType, class ComparatorType>
   KOKKOS_FUNCTION
   IteratorType is_sorted_until(const TeamHandleType& teamHandle,                   (11)
                                IteratorType first, IteratorType last,
                                ComparatorType comp);

   template <
      class TeamHandleType,
      class DataType, class... Properties, class ComparatorType>
   KOKKOS_FUNCTION
   auto is_sorted_until(const TeamHandleType& teamHandle,                           (12)
                        const Kokkos::View<DataType, Properties...>& view,
                        ComparatorType comp);

Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. |IsSorted| replace:: ``is_sorted``
.. _IsSorted: ./StdIsSorted.rst

- ``exespace``, ``teamHandle``, ``first``, ``last``, ``view``, ``comp``: same as in |IsSorted|_

- ``label``: string forwarded to internal parallel kernels for debugging purposes

  - must be rank-1, and have ``LayoutLeft``, ``LayoutRight``, or ``LayoutStride``

  - must be accessible from ``exespace``

  - 1 & 3: The default string is "Kokkos::is_sorted_until_iterator_api_default"

  - 5 & 7: The default string is "Kokkos::is_sorted_until_view_api_default"

  - NOTE: overloads accepting a team handle do not use a label internally

Return Value
~~~~~~~~~~~~

- The last iterator ``it`` for which range ``[first, it)`` is sorted and where the following is true: ``std::is_same_v<decltype(it), IteratorType>``, or for which range ``[Kokkos::Experimental::begin(view), it)`` is sorted. For this second case, note that ``it`` is computed as: ``Kokkos::Experimental::begin(view) + increment`` where ``increment`` is found in the algorithm.
