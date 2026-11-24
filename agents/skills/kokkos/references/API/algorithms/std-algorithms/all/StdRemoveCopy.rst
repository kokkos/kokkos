``remove_copy``
===============

Header: ``Kokkos_StdAlgorithms.hpp``

Description
-----------

Copies the elements from a range to a new range starting at ``first_to`` or from ``view_from``
to ``view_dest`` omitting those that are equal to ``value``.

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.

Overload set accepting execution space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <
     class ExecutionSpace,
     class InputIterator, class OutputIterator,
     class ValueType>
   OutputIterator remove_copy(const ExecutionSpace& exespace,                   (1)
                              InputIterator first_from,
                              InputIterator last_from,
                              OutputIterator first_to,
                              const ValueType& value);

   template <
     class ExecutionSpace,
     class InputIterator, class OutputIterator,
     class ValueType>
   OutputIterator remove_copy(const std::string& label,                         (2)
                              const ExecutionSpace& exespace,
                              InputIterator first_from,
                              InputIterator last_from,
                              OutputIterator first_to,
                              const ValueType& value);

   template <
     class ExecutionSpace,
     class DataType1, class... Properties1,
     class DataType2, class... Properties2,
     class ValueType>
   auto remove_copy(const ExecutionSpace& exespace,                             (3)
                    const Kokkos::View<DataType1, Properties1...>& view_from,
                    const Kokkos::View<DataType2, Properties2...>& view_dest,
                    const ValueType& value);

   template <
     class ExecutionSpace,
     class DataType1, class... Properties1,
     class DataType2, class... Properties2,
     class ValueType>
   auto remove_copy(const std::string& label,                                   (4)
                    const ExecutionSpace& exespace,
                    const Kokkos::View<DataType1, Properties1...>& view_from,
                    const Kokkos::View<DataType2, Properties2...>& view_dest,
                    const ValueType& value);


Overload set accepting a team handle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 4.2

.. code-block:: cpp

   template <
     class TeamHandleType,
     class InputIterator, class OutputIterator,
     class ValueType>
   KOKKOS_FUNCTION
   OutputIterator remove_copy(const TeamHandleType& teamHandle,                 (5)
                              InputIterator first_from,
                              InputIterator last_from,
                              OutputIterator first_to,
                              const ValueType& value);

   template <
     class TeamHandleType,
     class DataType1, class... Properties1,
     class DataType2, class... Properties2,
     class ValueType>
   KOKKOS_FUNCTION
   auto remove_copy(const TeamHandleType& teamHandle,                           (6)
                    const Kokkos::View<DataType1, Properties1...>& view_from,
                    const Kokkos::View<DataType2, Properties2...>& view_dest,
                    const ValueType& value);

Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``exespace``: execution space instance

- ``teamHandle``: team handle instance given inside a parallel region when using a TeamPolicy

- ``label``: string forwarded to internal parallel kernels for debugging purposes

  - 1: The default string is "Kokkos::remove_copy_iterator_api_default".

  - 3: The default string is "Kokkos::remove_copy_view_api_default".

  - NOTE: overloads accepting a team handle do not use a label internally

- ``first_from, last_from``: range of elements to copy from

  - must be *random access iterators*, e.g., returned from ``Kokkos::Experimental::(c)begin/(c)end``

  - must represent a valid range, i.e., ``last >= first``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``first_to``: beginning of the range to copy to

  - must be *random access iterators*, e.g., returned from ``Kokkos::Experimental::(c)begin/(c)end``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``view_from``, ``view_dest``: source and destination views

  - must be rank-1, and have ``LayoutLeft``, ``LayoutRight``, or ``LayoutStride``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``value``: target value to omit

Return Value
~~~~~~~~~~~~

Iterator to the element after the last element copied.
