``is_partitioned``
==================

Header: ``<Kokkos_StdAlgorithms.hpp>``

Description
-----------

Returns ``true`` if all elements in a range or in a rank-1 ``View`` satisfying
the predicate ``pred`` appear *before* all elements that don't.
If the range or the ``view`` is empty, returns ``true``.

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.

Overload set accepting execution space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <class ExecutionSpace, class InputIterator, class PredicateType>
   bool is_partitioned(const ExecutionSpace& exespace,                              (1)
                       InputIterator first, InputIterator last,
                       PredicateType pred);

   template <class ExecutionSpace, class InputIterator, class PredicateType>
   bool is_partitioned(const std::string& label, const ExecutionSpace& exespace,    (2)
                       InputIterator first, InputIterator last,
                       PredicateType pred);

   template <class ExecutionSpace, class DataType, class... Properties, class PredicateType>
   auto is_partitioned(const ExecutionSpace& exespace,
                       const ::Kokkos::View<DataType, Properties...>& view,         (3)
                       PredicateType pred);

   template <class ExecutionSpace, class DataType, class... Properties, class PredicateType>
   auto is_partitioned(const std::string& label, const ExecutionSpace& exespace,
                       const ::Kokkos::View<DataType, Properties...>& view,         (4)
                       PredicateType pred);


Overload set accepting a team handle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 4.2

.. code-block:: cpp

   template <class TeamHandleType, class IteratorType, class PredicateType>
   KOKKOS_FUNCTION
   bool is_partitioned(const TeamHandleType& teamHandle, IteratorType first,        (5)
                       IteratorType last, PredicateType pred);

   template <class TeamHandleType, class PredicateType, class DataType,
             class... Properties>
   KOKKOS_FUNCTION
   bool is_partitioned(const TeamHandleType& teamHandle,                            (6)
                       const ::Kokkos::View<DataType, Properties...>& view,
                       PredicateType pred);

Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``exespace``: execution space instance

- ``teamHandle``: team handle instance given inside a parallel region when using a TeamPolicy

- ``label``: string forwarded to internal parallel kernels for debugging purposes

  - 1: The default string is "Kokkos::is_partitioned_iterator_api_default".

  - 3: The default string is "Kokkos::is_partitioned_view_api_default".

  - NOTE: overloads accepting a team handle do not use a label internally

- ``first, last``: range of elements to search in

  - must be *random access iterators*, e.g., returned from ``Kokkos::Experimental::(c)begin/(c)end``

  - must represent a valid range, i.e., ``last >= first``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``view``:

  - must be rank-1, and have ``LayoutLeft``, ``LayoutRight``, or ``LayoutStride``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``pred``:

  - *unary* predicate returning ``true`` for the required element to replace; ``pred(v)``
    must be valid to be called from the execution space passed, and convertible to bool for every
    argument ``v`` of type (possible const) ``value_type``, where ``value_type``
    is the value type of ``IteratorType`` (for 1,2) or the value type of ``view`` (for 3,4),
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

- ``true``: if range is partitioned according to ``pred`` or if range is empty
- ``false``: otherwise

Example
~~~~~~~

.. code-block:: cpp

   namespace KE = Kokkos::Experimental;

   template<class ValueType>
   struct IsNegative
   {
     KOKKOS_INLINE_FUNCTION
     bool operator()(const ValueType & operand) const {
       constexpr auto zero = static_cast<ValueType>(0);
       return (operand < zero);
     }
   };

   using view_type = Kokkos::View<int*>;
   view_type a("a", 15);
   // fill a somehow

   auto exespace  = Kokkos::DefaultExecutionSpace;
   const auto res = KE::is_partitioned(exespace, KE::cbegin(a), KE::cend(a), IsNegative<int>());
