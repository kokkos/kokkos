
``move``
========

Header: ``<Kokkos_StdAlgorithms.hpp>``

Description
-----------

Moves the elements from the range or from a rank-1 ``View``
to the range beginning at ``d_first`` or to a target rank-1 ``View``.

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.


Overload set accepting execution space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <class ExecutionSpace, class InputIterator, class OutputIterator>
   OutputIterator move(const ExecutionSpace& ex, InputIterator first,          (1)
                       InputIterator last, OutputIterator d_first);

   template <class ExecutionSpace, class InputIterator, class OutputIterator>
   OutputIterator move(const std::string& label, const ExecutionSpace& ex,     (2)
                       InputIterator first, InputIterator last,
                       OutputIterator d_first);

   template <class ExecutionSpace, class DataType1, class... Properties1,
             class DataType2, class... Properties2>
   auto move(const ExecutionSpace& ex,                                         (3)
             const ::Kokkos::View<DataType1, Properties1...>& source,
             ::Kokkos::View<DataType2, Properties2...>& dest);

   template <class ExecutionSpace, class DataType1, class... Properties1,
             class DataType2, class... Properties2>
   auto move(const std::string& label, const ExecutionSpace& ex,               (4)
             const ::Kokkos::View<DataType1, Properties1...>& source,
             ::Kokkos::View<DataType2, Properties2...>& dest);

Overload set accepting a team handle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 4.2

.. code-block:: cpp

   template <class TeamHandleType, class InputIterator, class OutputIterator>
   KOKKOS_FUNCTION
   OutputIterator move(const TeamHandleType& teamHandle, InputIterator first,  (5)
                       InputIterator last, OutputIterator d_first);

   template <class TeamHandleType, class DataType1, class... Properties1,
             class DataType2, class... Properties2>
   KOKKOS_FUNCTION
   auto move(const TeamHandleType& teamHandle,                                 (6)
             const ::Kokkos::View<DataType1, Properties1...>& source,
             ::Kokkos::View<DataType2, Properties2...>& dest);

Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``exespace``: execution space instance

- ``teamHandle``: team handle instance given inside a parallel region when using a TeamPolicy

- ``label``: string forwarded to internal parallel kernels for debugging purposes

  - for 1, the default string is: "Kokkos::move_iterator_api_default"

  - for 3, the default string is: "Kokkos::move_view_api_default"

  - NOTE: overloads accepting a team handle do not use a label internally

- ``first``, ``last``, ``d_first``: range of elements to move from and to

  - must be *random access iterator*

  - must represent a valid range, i.e., ``last >= first``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``source``, ``dest``: views to move from and to

  - must be rank-1, and have ``LayoutLeft``, ``LayoutRight``, or ``LayoutStride``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle


Return Value
~~~~~~~~~~~~

- 1,2,5: an iterator equal to ``d_first + Kokkos::Experimental::distance(first, last)``

- 3,4,6: an iterator equal to
  ``Kokkos::Experimental::begin(dest) +
  Kokkos::Experimental:distance(Kokkos::Experimental::begin(source), Kokkos::Experimental::end(source))``
