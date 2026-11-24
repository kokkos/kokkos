
``copy_if``
===========

Header: ``<Kokkos_StdAlgorithms.hpp>``

Description
-----------

Copies the elements for which a predicate returns ``true`` from source range or ``View`` to
another range or ``View``

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.

Overload set accepting execution space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

  template <
    class ExecutionSpace, class InputIteratorType,
    class OutputIteratorType, class UnaryPredicateType>
  OutputIteratorType copy_if(const ExecutionSpace& exespace,                   (1)
                             InputIteratorType first_from,
                             InputIteratorType last_from,
                             OutputIteratorType first_to,
                             UnaryPredicateType pred);

  template <
    class ExecutionSpace, class InputIteratorType,
    class OutputIteratorType, class UnaryPredicateType>
  OutputIteratorType copy_if(const std::string& label,
                             const ExecutionSpace& exespace,                   (2)
                             InputIteratorType first_from,
                             InputIteratorType last_from,
                             OutputIteratorType first_to,
                             UnaryPredicateType pred);

  template <
    class ExecutionSpace,
    class DataType1, class... Properties1,
    class DataType2, class... Properties2,
    class UnaryPredicateType
  >
  auto copy_if(const ExecutionSpace& exespace,                                 (3)
               const Kokkos::View<DataType1, Properties1...>& view_from,
               const Kokkos::View<DataType2, Properties2...>& view_to,
               UnaryPredicateType pred);

  template <
    class ExecutionSpace,
    class DataType1, class... Properties1,
    class DataType2, class... Properties2,
    class UnaryPredicateType
  >
  auto copy_if(const std::string& label, const ExecutionSpace& exespace,       (4)
               const Kokkos::View<DataType1, Properties1...>& view_from,
               const Kokkos::View<DataType2, Properties2...>& view_to,
               UnaryPredicateType pred);


Overload set accepting a team handle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 4.2

.. code-block:: cpp

  template <
    class TeamHandleType, class InputIteratorType,
    class OutputIteratorType, class UnaryPredicateType>
  KOKKOS_FUNCTION
  OutputIteratorType copy_if(const TeamHandleType& teamHandle,                 (5)
                             InputIteratorType first_from,
                             InputIteratorType last_from,
                             OutputIteratorType first_to,
                             UnaryPredicateType pred);

  template <
    class TeamHandleType,
    class DataType1, class... Properties1,
    class DataType2, class... Properties2,
    class UnaryPredicateType>
  KOKKOS_FUNCTION
  auto copy_if(const TeamHandleType& teamHandle,                              (6)
               const Kokkos::View<DataType1, Properties1...>& view_from,
               const Kokkos::View<DataType2, Properties2...>& view_to,
               UnaryPredicateType pred);


Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. |copy| replace:: ``copy``
.. _copy: ./StdCopy.rst

- ``exespace``, ``teamHandle``, ``first_from``, ``last_from``, ``first_to``, ``view_from``, ``view_to``: same as in |copy|_

- ``label``:

  - for 1, the default string is: "Kokkos::copy_if_iterator_api_default"

  - for 3, the default string is: "Kokkos::copy_if_view_api_default"

- ``pred``: unary predicate which returns ``true`` for the required element to copy

  - ``pred(v)`` must be valid to be called from the execution space passed or the execution
    space associated with the team handle, and convertible to bool for every
    argument ``v`` of type (possible const) ``value_type``, where ``value_type``
    is the value type of ``InputIteratorType`` or of ``view_from``, and must not modify ``v``.

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

Iterator to the destination element *after* the last element copied.
