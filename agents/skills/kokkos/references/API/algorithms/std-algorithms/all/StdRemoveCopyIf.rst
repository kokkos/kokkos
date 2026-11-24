``remove_copy_if``
==================

Header: ``Kokkos_StdAlgorithms.hpp``

Description
-----------

Copies the elements from a range to a new range starting at ``first_to`` or from ``view_from``
to ``view_dest`` omitting those for which ``pred`` returns ``true``.

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.

Overload set accepting execution space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <
     class ExecutionSpace,
     class InputIterator, class OutputIterator,
     class UnaryPredicate>
   OutputIterator remove_copy_if(const ExecutionSpace& exespace,                   (1)
                                 InputIterator first_from,
                                 InputIterator last_from,
                                 OutputIterator first_to,
                                 const UnaryPredicate& pred);

   template <
     class ExecutionSpace,
     class InputIterator, class OutputIterator,
     class UnaryPredicate>
   OutputIterator remove_copy_if(const std::string& label,                         (2)
                                 const ExecutionSpace& exespace,
                                 InputIterator first_from,
                                 InputIterator last_from,
                                 OutputIterator first_to,
                                 const UnaryPredicate& pred);

   template <
     class ExecutionSpace,
     class DataType1, class... Properties1,
     class DataType2, class... Properties2,
     class UnaryPredicate>
   auto remove_copy_if(const ExecutionSpace& exespace,                             (3)
                     const Kokkos::View<DataType1, Properties1...>& view_from,
                     const Kokkos::View<DataType2, Properties2...>& view_dest,
                     const UnaryPredicate& pred);

   template <
     class ExecutionSpace,
     class DataType1, class... Properties1,
     class DataType2, class... Properties2,
     class UnaryPredicate>
   auto remove_copy_if(const std::string& label,                                   (4)
                       const ExecutionSpace& exespace,
                       const Kokkos::View<DataType1, Properties1...>& view_from,
                       const Kokkos::View<DataType2, Properties2...>& view_dest,
                       const UnaryPredicate& pred);

Overload set accepting a team handle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 4.2

.. code-block:: cpp

   template <
     class TeamHandleType,
     class InputIterator, class OutputIterator,
     class UnaryPredicate>
   KOKKOS_FUNCTION
   OutputIterator remove_copy_if(const TeamHandleType& teamHandle,                 (5)
                                 InputIterator first_from,
                                 InputIterator last_from,
                                 OutputIterator first_to,
                                 const UnaryPredicate& pred);

   template <
     class TeamHandleType,
     class DataType1, class... Properties1,
     class DataType2, class... Properties2,
     class UnaryPredicate>
   KOKKOS_FUNCTION
   auto remove_copy_if(const TeamHandleType& teamHandle,                           (6)
                       const Kokkos::View<DataType1, Properties1...>& view_from,
                       const Kokkos::View<DataType2, Properties2...>& view_dest,
                       const UnaryPredicate& pred);

Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. |RemoveCopy| replace:: ``remove_copy``
.. _RemoveCopy: ./StdRemoveCopy.rst

- ``exespace``, ``teamHandle``, ``first_from, last_from``, ``first_to``, ``view_from``, ``view_dest``: same as in |RemoveCopy|_

- ``label``: string forwarded to internal parallel kernels for debugging purposes

  - 1: The default string is "Kokkos::remove_copy_if_iterator_api_default".

  - 3: The default string is "Kokkos::remove_copy_if_view_api_default".

  - NOTE: overloads accepting a team handle do not use a label internally

- ``pred``:

  - *unary* predicate returning ``true`` for the required element to replace; ``pred(v)`` must be valid to be called from the execution space passed, and convertible to bool for every argument ``v`` of type (possible const) ``value_type``, where ``value_type`` is the value type of ``InputIterator`` (for 1,2,5) or the value type of ``view`` (for 3,4,6), and must not modify ``v``.

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

Iterator to the element after the last element copied.
