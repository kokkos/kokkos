``rotate``
==========

Header: ``Kokkos_StdAlgorithms.hpp``

Description
-----------

Swaps the elements in the range or in ``view`` in such a way that
the element ``n_first`` or ``view(n_location)`` becomes the first element
of the new range and ``n_first - 1`` becomes the last element.

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.

Overload set accepting execution space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <class ExecutionSpace, class IteratorType>
   IteratorType rotate(const ExecutionSpace& exespace,                            (1)
                       IteratorType first,
                       IteratorType n_first,
                       IteratorType last);

   template <class ExecutionSpace, class IteratorType>
   IteratorType rotate(const std::string& label, const ExecutionSpace& exespace,  (2)
                       IteratorType first,
                       IteratorType n_first,
                       IteratorType last);

   template <class ExecutionSpace, class DataType, class... Properties>
   auto rotate(const ExecutionSpace& exespace,                                    (3)
               const Kokkos::View<DataType, Properties...>& view,
               std::size_t n_location);

   template <class ExecutionSpace, class DataType, class... Properties>
   auto rotate(const std::string& label, const ExecutionSpace& exespace,          (4)
               const Kokkos::View<DataType, Properties...>& view,
               std::size_t n_location);

Overload set accepting a team handle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 4.2

.. code-block:: cpp

   template <class TeamHandleType, class IteratorType>
   KOKKOS_FUNCTION
   IteratorType rotate(const TeamHandleType& teamHandle,                          (5)
                       IteratorType first,
                       IteratorType n_first,
                       IteratorType last);

   template <class TeamHandleType, class DataType, class... Properties>
   KOKKOS_FUNCTION
   auto rotate(const TeamHandleType& teamHandle,                                  (6)
               const Kokkos::View<DataType, Properties...>& view,
               std::size_t n_location);

Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``exespace``: execution space instance

- ``teamHandle``: team handle instance given inside a parallel region when using a TeamPolicy

- ``label``: string forwarded to internal parallel kernels for debugging purposes

  - 1: The default string is "Kokkos::rotate_iterator_api_default".

  - 3: The default string is "Kokkos::rotate_view_api_default".

  - NOTE: overloads accepting a team handle do not use a label internally

- ``first, last``: range of elements to modify

  - must be *random access iterators*, e.g., returned from ``Kokkos::Experimental::(c)begin/(c)end``

  - must represent a valid range, i.e., ``last >= first``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``n_first``: iterator to element that should be the first of the rotated range

  - must be *random access iterators*, e.g., returned from ``Kokkos::Experimental::(c)begin/(c)end``

  - must be such that ``[first, n_first)`` and ``[n_first, last)`` are valid ranges.

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``view``:

  - must be rank-1, and have ``LayoutLeft``, ``LayoutRight``, or ``LayoutStride``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``n_location``: integer value identifying the element to rotate about

Return Value
~~~~~~~~~~~~

- 1, 2, 5: Returns the iterator computed as ``first + (last - n_first)``

- 3, 4, 6: Returns ``Kokkos::begin(view) + (Kokkos::end(view) - n_location)``
