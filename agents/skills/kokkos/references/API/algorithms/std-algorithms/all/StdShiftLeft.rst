``shift_left``
==============

Header: ``Kokkos_StdAlgorithms.hpp``

Description
-----------

Shifts the elements in a range or in ``view`` by ``n`` positions towards the *beginning*.

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.

Overload set accepting execution space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <class ExecutionSpace, class IteratorType>
   IteratorType shift_left(const ExecutionSpace& exespace,                 (1)
                           IteratorType first, IteratorType last,
                           typename IteratorType::difference_type n);

   template <class ExecutionSpace, class IteratorType>
   IteratorType shift_left(const std::string& label,                       (2)
                           const ExecutionSpace& exespace,
                           IteratorType first, IteratorType last,
                           typename IteratorType::difference_type n);

   template <class ExecutionSpace, class DataType, class... Properties>
   auto shift_left(const ExecutionSpace& exespace,                         (3)
                  const Kokkos::View<DataType, Properties...>& view,
                  typename decltype(begin(view))::difference_type n);

   template <class ExecutionSpace, class DataType, class... Properties>
   auto shift_left(const std::string& label,                               (4)
                   const ExecutionSpace& exespace,
                   const Kokkos::View<DataType, Properties...>& view,
                  typename decltype(begin(view))::difference_type n);


Overload set accepting a team handle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 4.2

.. code-block:: cpp

   template <class TeamHandleType, class IteratorType>
   KOKKOS_FUNCTION
   IteratorType shift_left(const TeamHandleType& teamHandle,               (5)
                           IteratorType first, IteratorType last,
                           typename IteratorType::difference_type n);

   template <class TeamHandleType, class DataType, class... Properties>
   KOKKOS_FUNCTION
   auto shift_left(const TeamHandleType& teamHandle,                       (6)
                   const Kokkos::View<DataType, Properties...>& view,
                   typename decltype(begin(view))::difference_type n);

Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``exespace``: execution space instance

- ``teamHandle``: team handle instance given inside a parallel region when using a TeamPolicy

- ``label``: string forwarded to internal parallel kernels for debugging purposes

  - 1: The default string is "Kokkos::shift_left_iterator_api_default".

  - 3: The default string is "Kokkos::shift_left_view_api_default".

  - NOTE: overloads accepting a team handle do not use a label internally

- ``first, last``: range of elements to shift

  - must be *random access iterators*, e.g., returned from ``Kokkos::Experimental::(c)begin/(c)end``

  - must represent a valid range, i.e., ``last >= first``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``view``: view to modify

  - must be rank-1, and have ``LayoutLeft``, ``LayoutRight``, or ``LayoutStride``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``n``: the number of positions to shift

  - must be non-negative

Return Value
~~~~~~~~~~~~

The end of the resulting range. If ``n`` is less than ``last - first``, returns ``first + (last - first - n)``. Otherwise, returns ``first``.
