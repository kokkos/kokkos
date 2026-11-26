``rotate_copy``
===============

Header: ``Kokkos_StdAlgorithms.hpp``

Description
-----------

Copies the elements from the range ``[first_from, last_from)`` to the range starting at ``first_to`` or from ``view_from`` to ``view_dest`` in such a way that the element ``n_first`` or ``view(n_location)`` becomes the first element of the new range and ``n_first - 1`` becomes the last element.

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.

Overload set accepting execution space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <class ExecutionSpace, class InputIterator, class OutputIterator>
   OutputIterator rotate_copy(const ExecutionSpace& exespace,                   (1)
                              InputIterator first_from,
                              InputIterator n_first,
                              InputIterator last_from,
                              OutputIterator first_to);

   template <class ExecutionSpace, class InputIterator, class OutputIterator>
   OutputIterator rotate_copy(const std::string& label,                         (2)
                              const ExecutionSpace& exespace,
                              InputIterator first_from,
                              InputIterator n_first,
                              InputIterator last_from,
                              OutputIterator first_to);

   template <
     class ExecutionSpace,
     class DataType1, class... Properties1,
     class DataType2, class... Properties2>
   auto rotate_copy(const ExecutionSpace& exespace,                             (3)
                    const Kokkos::View<DataType1, Properties1...>& source,
                    std::size_t n_location,
                    const Kokkos::View<DataType2, Properties2...>& dest);

   template <
     class ExecutionSpace,
     class DataType1, class... Properties1,
     class DataType2, class... Properties2>
   auto rotate_copy(const std::string& label,                                   (4)
                    const ExecutionSpace& exespace,
                    const Kokkos::View<DataType1, Properties1...>& source,
                    std::size_t n_location,
                    const Kokkos::View<DataType2, Properties2...>& dest);

Overload set accepting a team handle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 4.2

.. code-block:: cpp

   template <class TeamHandleType, class InputIterator, class OutputIterator>
   KOKKOS_FUNCTION
   OutputIterator rotate_copy(const TeamHandleType& teamHandle,                 (5)
                              InputIterator first_from,
                              InputIterator n_first,
                              InputIterator last_from,
                              OutputIterator first_to);

   template <
     class TeamHandleType,
     class DataType1, class... Properties1,
     class DataType2, class... Properties2>
   KOKKOS_FUNCTION
   auto rotate_copy(const TeamHandleType& teamHandle,                           (6)
                    const Kokkos::View<DataType1, Properties1...>& source,
                    std::size_t n_location,
                    const Kokkos::View<DataType2, Properties2...>& dest);

Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``exespace``: execution space instance

- ``teamHandle``: team handle instance given inside a parallel region when using a TeamPolicy

- ``label``: string forwarded to internal parallel kernels for debugging purposes

  - 1: The default string is "Kokkos::rotate_copy_iterator_api_default".

  - 3: The default string is "Kokkos::rotate_copy_view_api_default".

  - NOTE: overloads accepting a team handle do not use a label internally

- ``first_from, last_from``: range of elements to copy from

  - must be *random access iterators*, e.g., returned from ``Kokkos::Experimental::(c)begin/(c)end``

  - must represent a valid range, i.e., ``last >= first``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``first_to``: beginning of the range to copy to

  - must be *random access iterators*, e.g., returned from ``Kokkos::Experimental::(c)begin/(c)end``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``n_first``: iterator to element that should be the first of the rotated range

  - must be *random access iterators*, e.g., returned from ``Kokkos::Experimental::(c)begin/(c)end``

  - must be such that ``[first_from, n_first)`` and ``[n_first, last_from)`` are valid ranges.

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``source, dest``:

  - must be rank-1, and have ``LayoutLeft``, ``LayoutRight``, or ``LayoutStride``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``n_location``: integer value identifying the element to rotate about

Return Value
~~~~~~~~~~~~

Iterator to the element *after* the last element copied.
