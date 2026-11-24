``partition_copy``
==================

Header: ``Kokkos_StdAlgorithms.hpp``

Description
-----------

Copies from a range or a rank-1 ``View`` the elements that satisfy the predicate ``pred`` to ``to_first_true`` or ``view_true``, while the others are copied to ``to_first_false`` or ``view_false``.

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.

Overload set accepting execution space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <class ExecutionSpace,
             class InputIteratorType,
             class OutputIteratorTrueType,
             class OutputIteratorFalseType,
             class PredicateType>
   ::Kokkos::pair<OutputIteratorTrueType, OutputIteratorFalseType>
   partition_copy(const ExecutionSpace& exespace,                                  (1)
                  InputIteratorType from_first,
                  InputIteratorType from_last,
                  OutputIteratorTrueType to_first_true,
                  OutputIteratorFalseType to_first_false,
                  PredicateType pred);

   template <class ExecutionSpace,
             class InputIteratorType,
             class OutputIteratorTrueType,
             class OutputIteratorFalseType,
             class PredicateType>
   ::Kokkos::pair<OutputIteratorTrueType, OutputIteratorFalseType>
   partition_copy(const std::string& label,                                        (2)
                  const ExecutionSpace& exespace,
                  InputIteratorType from_first,
                  InputIteratorType from_last,
                  OutputIteratorTrueType to_first_true,
                  OutputIteratorFalseType to_first_false,
                  PredicateType pred);

   template <class ExecutionSpace,
             class DataType1, class... Properties1,
             class DataType2, class... Properties2,
             class DataType3, class... Properties3,
             class PredicateType>
   auto partition_copy(const ExecutionSpace& exespace,                             (3)
                       const ::Kokkos::View<DataType1, Properties1...>& view_from,
                       const ::Kokkos::View<DataType2, Properties2...>& view_dest_true,
                       const ::Kokkos::View<DataType3, Properties3...>& view_dest_false,
                       PredicateType pred);

   template <class ExecutionSpace,
             class DataType1, class... Properties1,
             class DataType2, class... Properties2,
             class DataType3, class... Properties3,
             class PredicateType>
   auto partition_copy(const std::string& label,                                   (4)
                       const ExecutionSpace& exespace,
                       const ::Kokkos::View<DataType1, Properties1...>& view_from,
                       const ::Kokkos::View<DataType2, Properties2...>& view_dest_true,
                       const ::Kokkos::View<DataType3, Properties3...>& view_dest_false,
                       PredicateType pred);

Overload set accepting a team handle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 4.2

.. code-block:: cpp

   template <class TeamHandleType, class InputIteratorType,
             class OutputIteratorTrueType, class OutputIteratorFalseType,
             class PredicateType>
   KOKKOS_FUNCTION
   ::Kokkos::pair<OutputIteratorTrueType, OutputIteratorFalseType>
   partition_copy(const TeamHandleType& teamHandle, InputIteratorType from_first,  (5)
               InputIteratorType from_last,
               OutputIteratorTrueType to_first_true,
               OutputIteratorFalseType to_first_false, PredicateType pred);

   template <class TeamHandleType, class DataType1, class... Properties1,
       class DataType2, class... Properties2, class DataType3,
       class... Properties3, class PredicateType>
   KOKKOS_FUNCTION
   auto partition_copy(const TeamHandleType& teamHandle,                           (6)
       const ::Kokkos::View<DataType1, Properties1...>& view_from,
       const ::Kokkos::View<DataType2, Properties2...>& view_dest_true,
       const ::Kokkos::View<DataType3, Properties3...>& view_dest_false,
       PredicateType pred);

Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``exespace``: execution space instance

- ``teamHandle``: team handle instance given inside a parallel region when using a TeamPolicy

- ``label``: string forwarded to internal parallel kernels for debugging purposes

  - 1: The default string is "Kokkos::partition_copy_iterator_api_default".

  - 3: The default string is "Kokkos::partition_copy_view_api_default".

  - NOTE: overloads accepting a team handle do not use a label internally

- ``from_first, from_last``: range of elements to copy from

  - must be *random access iterators*, e.g., returned from ``Kokkos::Experimental::(c)begin/(c)end``

  - must represent a valid range, i.e., ``last >= first``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``to_first_true``: beginning of the range to copy the elements that satisfy ``pred`` to

  - must be *random access iterators*, e.g., returned from ``Kokkos::Experimental::(c)begin/(c)end``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``to_first_false``: beginning of the range to copy the elements that do NOT satisfy ``pred`` to

  - must be *random access iterators*, e.g., returned from ``Kokkos::Experimental::(c)begin/(c)end``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``view_from``: source view of elements to copy from

  - must be rank-1, and have ``LayoutLeft``, ``LayoutRight``, or ``LayoutStride``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``view_dest_true``: destination view to copy the elements that satisfy ``pred`` to
  - must be rank-1, and have ``LayoutLeft``, ``LayoutRight``, or ``LayoutStride``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``view_dest_false``: destination view to copy the elements that do NOT satisfy ``pred`` to

  - must be rank-1, and have ``LayoutLeft``, ``LayoutRight``, or ``LayoutStride``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``pred``:

  - *unary* predicate returning ``true`` for the required element to replace; ``pred(v)``
    must be valid to be called from the execution space passed, and convertible to bool for every
    argument ``v`` of type (possible const) ``value_type``, where ``value_type``
    is the value type of ``InputIteratorType`` (for 1,2) or the value type of ``view_from`` (for 3,4),
    and must not modify ``v``.

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

Returns a ``Kokkos::pair`` containing the iterators to the end of two destination ranges (or views)
