``remove``
==========

Header: ``Kokkos_StdAlgorithms.hpp``

Description
-----------

Removes all elements equal to ``value`` by shifting via move assignment the elements in a range or in ``View`` such that the elements not to be removed appear in the beginning of the range or in the beginning of ``View``. Relative order of the elements that remain is preserved and the physical size of the container is unchanged.

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.

Overload set accepting execution space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <class ExecutionSpace, class Iterator, class ValueType>
   Iterator remove(const ExecutionSpace& exespace,                       (1)
                   Iterator first, Iterator last,
                   const ValueType& value);

   template <class ExecutionSpace, class Iterator, class ValueType>
   Iterator remove(const std::string& label,                             (2)
                   const ExecutionSpace& exespace,
                   Iterator first, Iterator last,
                   const ValueType& value);

   template <
     class ExecutionSpace,
     class DataType, class... Properties,
     class ValueType>
   auto remove(const ExecutionSpace& exespace,                           (3)
               const Kokkos::View<DataType, Properties...>& view,
               const ValueType& value);

   template <
     class ExecutionSpace,
     class DataType, class... Properties,
     class ValueType>
   auto remove(const std::string& label,                                 (4)
               const ExecutionSpace& exespace,
               const Kokkos::View<DataType, Properties...>& view,
               const ValueType& value);

Overload set accepting a team handle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 4.2

.. code-block:: cpp

   template <class TeamHandleType, class Iterator, class ValueType>
   KOKKOS_FUNCTION
   Iterator remove(const TeamHandleType& teamHandle,                     (5)
                   Iterator first, Iterator last,
                   const ValueType& value);

   template <
     class TeamHandleType,
     class DataType, class... Properties,
     class ValueType>
   KOKKOS_FUNCTION
   auto remove(const TeamHandleType& teamHandle,                         (6)
               const Kokkos::View<DataType, Properties...>& view,
               const ValueType& value);

Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``exespace``: execution space instance

- ``teamHandle``: team handle instance given inside a parallel region when using a TeamPolicy

- ``label``: string forwarded to internal parallel kernels for debugging purposes

  - 1: The default string is "Kokkos::remove_iterator_api_default".

  - 3: The default string is "Kokkos::remove_view_api_default".

  - NOTE: overloads accepting a team handle do not use a label internally

- ``first, last``: range of elements to modify

  - must be *random access iterators*, e.g., returned from ``Kokkos::Experimental::(c)begin/(c)end``

  - must represent a valid range, i.e., ``last >= first``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``value``: target value to remove

- ``view``: view of elements to modify

  - must be rank-1, and have ``LayoutLeft``, ``LayoutRight``, or ``LayoutStride``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

Return Value
~~~~~~~~~~~~

Iterator to the element *after* the new logical end.
