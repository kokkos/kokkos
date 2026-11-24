``unique_copy``
===============

Header: ``Kokkos_StdAlgorithms.hpp``

Description
-----------

Copies the elements from a range or a ``source`` view to a range starting at ``first_to`` or a ``dest`` view such that there are no consecutive equal elements. It returns an iterator to the element *after* the last element copied in the destination or destination view. Equivalence is checked using ``operator==`` or the binary predicate ``pred``.

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.

Overload set accepting execution space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <class ExecutionSpace, class InputIterator, class OutputIterator>
   OutputIterator unique_copy(const ExecutionSpace& exespace,                 (1)
                              InputIterator first_from, InputIterator last_from,
                              OutputIterator first_to);

   template <class ExecutionSpace, class InputIterator, class OutputIterator>
   OutputIterator unique_copy(const std::string& label,                       (2)
                              const ExecutionSpace& exespace,
                              InputIterator first_from, InputIterator last_from,
                              OutputIterator first_to);

   template <
     class ExecutionSpace,
     class DataType1, class... Properties1,
     class DataType2, class... Properties2>
   auto unique_copy(const ExecutionSpace& exespace,                           (3)
                    const Kokkos::View<DataType1, Properties1...>& source,
                    const Kokkos::View<DataType2, Properties2...>& dest);

   template <
     class ExecutionSpace,
     class DataType1, class... Properties1,
     class DataType2, class... Properties2>
   auto unique_copy(const std::string& label,                                 (4)
                    const ExecutionSpace& exespace,
                    const Kokkos::View<DataType1, Properties1...>& source,
                    const Kokkos::View<DataType2, Properties2...>& dest);

   template <
     class ExecutionSpace,
     class InputIterator, class OutputIterator,
     class BinaryPredicate>
   OutputIterator unique_copy(const ExecutionSpace& exespace,                 (5)
                              InputIterator first_from, InputIterator last_from,
                              OutputIterator first_to,
                              BinaryPredicate pred);

   template <
     class ExecutionSpace,
     class InputIterator, class OutputIterator,
     class BinaryPredicate>
   OutputIterator unique_copy(const std::string& label,                       (6)
                              const ExecutionSpace& exespace,
                              InputIterator first_from, InputIterator last_from,
                              OutputIterator first_to,
                              BinaryPredicate pred);

   template <
     class ExecutionSpace,
     class DataType1, class... Properties1,
     class DataType2, class... Properties2,
     class BinaryPredicate>
   auto unique_copy(const ExecutionSpace& exespace,                           (7)
                    const Kokkos::View<DataType1, Properties1...>& source,
                    const Kokkos::View<DataType2, Properties2...>& dest,
                    BinaryPredicate pred);

   template <
     class ExecutionSpace,
     class DataType1, class... Properties1,
     class DataType2, class... Properties2,
     class BinaryPredicate>
   auto unique_copy(const std::string& label,                                 (8)
                    const ExecutionSpace& exespace,
                    const Kokkos::View<DataType1, Properties1...>& source,
                    const Kokkos::View<DataType2, Properties2...>& dest,
                    BinaryPredicate pred);

Overload set accepting a team handle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 4.2

.. code-block:: cpp

   template <class TeamHandleType, class InputIterator, class OutputIterator>
   KOKKOS_FUNCTION
   OutputIterator unique_copy(const TeamHandleType& teamHandle,               (9)
                              InputIterator first_from, InputIterator last_from,
                              OutputIterator first_to);

   template <
     class TeamHandleType,
     class DataType1, class... Properties1,
     class DataType2, class... Properties2>
   KOKKOS_FUNCTION
   auto unique_copy(const TeamHandleType& teamHandle,                         (10)
                    const Kokkos::View<DataType1, Properties1...>& source,
                    const Kokkos::View<DataType2, Properties2...>& dest);

   template <
     class TeamHandleType,
     class InputIterator, class OutputIterator,
     class BinaryPredicate>
   KOKKOS_FUNCTION
   OutputIterator unique_copy(const TeamHandleType& teamHandle,               (11)
                              InputIterator first_from, InputIterator last_from,
                              OutputIterator first_to,
                              BinaryPredicate pred);

   template <
     class TeamHandleType,
     class DataType1, class... Properties1,
     class DataType2, class... Properties2,
     class BinaryPredicate>
   KOKKOS_FUNCTION
   auto unique_copy(const TeamHandleType& teamHandle,                         (12)
                    const Kokkos::View<DataType1, Properties1...>& source,
                    const Kokkos::View<DataType2, Properties2...>& dest,
                    BinaryPredicate pred);

Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``exespace``: execution space instance

- ``teamHandle``: team handle instance given inside a parallel region when using a TeamPolicy

- ``label``: string forwarded to internal parallel kernels for debugging purposes

  - 1, 5: The default string is "Kokkos::unique_copy_iterator_api_default".

  - 3, 7: The default string is "Kokkos::unique_copy_view_api_default".

  - NOTE: overloads accepting a team handle do not use a label internally

- ``first_from, last_from``, ``first_to``: iterators to source range ``{first,last}_from``
  and destination range ``first_to``

  - must be *random access iterators*, e.g., returned from ``Kokkos::Experimental::(c)begin/(c)end``

  - must represent a valid range, i.e., ``last >= first``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``source``, ``dest``:

  - must be rank-1, and have ``LayoutLeft``, ``LayoutRight``, or ``LayoutStride``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``pred``:

  - *unary* predicate returning ``true`` for the required element to replace; ``pred(v)`` must be valid to be called from the execution space passed, and convertible to bool for every argument ``v`` of type (possible const) ``value_type``, where ``value_type`` is the value type of ``InputIterator`` (for 1,2,5,6,9,11) or the value type of ``view`` (for 3,4,7,8,10,12), and must not modify ``v``.

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

Iterator to the element *after* the last element copied in the destination range or view.
