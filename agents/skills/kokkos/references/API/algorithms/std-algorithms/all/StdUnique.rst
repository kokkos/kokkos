``unique``
==========

Header: ``Kokkos_StdAlgorithms.hpp``

Description
-----------

Eliminates all except the last element from every consecutive group of equivalent elements in a range or in a ``View`` and returns an iterator to the element *after* the new logical end of the range. Equivalence is checked using ``operator==`` and the binary predicate ``pred``.

.. note:: This behavior deviates from that of ``std::unique``. Specifically, ``std::unique`` eliminates all except the **first** element from every consecutive group of equivalent elements.

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.

Overload set accepting execution space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <class ExecutionSpace, class IteratorType>
   IteratorType unique(const ExecutionSpace& exespace,                       (1)
                       IteratorType first, IteratorType last);

   template <class ExecutionSpace, class IteratorType>
   IteratorType unique(const std::string& label,                             (2)
                       const ExecutionSpace& exespace,
                       IteratorType first, IteratorType last);

   template <class ExecutionSpace, class DataType, class... Properties>
   auto unique(const ExecutionSpace& exespace,                               (3)
               const Kokkos::View<DataType, Properties...>& view);

   template <class ExecutionSpace, class DataType, class... Properties>
   auto unique(const std::string& label, const ExecutionSpace& exespace,     (4)
               const Kokkos::View<DataType, Properties...>& view);

   template <class ExecutionSpace, class IteratorType, class BinaryPredicate>
   IteratorType unique(const ExecutionSpace& exespace,                       (5)
                       IteratorType first, IteratorType last,
                       BinaryPredicate pred);

   template <class ExecutionSpace, class IteratorType, class BinaryPredicate>
   IteratorType unique(const std::string& label,                             (6)
                       const ExecutionSpace& exespace,
                       IteratorType first, IteratorType last,
                       BinaryPredicate pred);

   template <
     class ExecutionSpace,
     class DataType, class... Properties,
     class BinaryPredicate>
   auto unique(const ExecutionSpace& exespace,                               (7)
               const Kokkos::View<DataType, Properties...>& view,
               BinaryPredicate pred);

   template <
     class ExecutionSpace,
     class DataType, class... Properties,
     class BinaryPredicate>
   auto unique(const std::string& label,                                     (8)
               const ExecutionSpace& exespace,
               const Kokkos::View<DataType, Properties...>& view,
               BinaryPredicate pred);

Overload set accepting a team handle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 4.2

.. code-block:: cpp

   template <class TeamHandleType, class IteratorType>
   KOKKOS_FUNCTION
   IteratorType unique(const TeamHandleType& teamHandle,                     (9)
                       IteratorType first, IteratorType last);

   template <class TeamHandleType, class DataType, class... Properties>
   KOKKOS_FUNCTION
   auto unique(const TeamHandleType& teamHandle,                             (10)
               const Kokkos::View<DataType, Properties...>& view);

   template <class TeamHandleType, class IteratorType, class BinaryPredicate>
   KOKKOS_FUNCTION
   IteratorType unique(const TeamHandleType& teamHandle,                     (11)
                       IteratorType first, IteratorType last,
                       BinaryPredicate pred);

   template <
       class TeamHandleType,
       class DataType, class... Properties,
       class BinaryPredicate>
   KOKKOS_FUNCTION
   auto unique(const TeamHandleType& teamHandle,                             (12)
               const Kokkos::View<DataType, Properties...>& view,
               BinaryPredicate pred);

Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``exespace``: execution space instance

- ``teamHandle``: team handle instance given inside a parallel region when using a TeamPolicy

- ``label``: string forwarded to internal parallel kernels for debugging purposes

  - 1, 5: The default string is "Kokkos::unique_iterator_api_default".

  - 3, 7: The default string is "Kokkos::unique_view_api_default".

  - NOTE: overloads accepting a team handle do not use a label internally

- ``first, last``: range of elements to search in

  - must be *random access iterators*, e.g., returned from ``Kokkos::Experimental::(c)begin/(c)end``

  - must represent a valid range, i.e., ``last >= first``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``view``:

  - must be rank-1, and have ``LayoutLeft``, ``LayoutRight``, or ``LayoutStride``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``pred``:

  - *unary* predicate returning ``true`` for the required element to replace; ``pred(v)`` must be valid to be called from the execution space passed, and convertible to bool for every argument ``v`` of type (possible const) ``value_type``, where ``value_type`` is the value type of ``IteratorType`` (for 1,2,5,6,9,11) or the value type of ``view`` (for 3,4,7,8,10,12), and must not modify ``v``.

  - must conform to:

  .. code-block:: cpp

     struct Predicate
     {
       KOKKOS_INLINE_FUNCTION
       bool operator()(const value_type & v) const { return /* ... */; }

       // or, also valid

       KOKKOS_INLINE_FUNCTION
       bool operator()(value_type v) const { return /* ... */; }
     };

Return Value
~~~~~~~~~~~~

Iterator to the element *after* the new logical end of the range.
