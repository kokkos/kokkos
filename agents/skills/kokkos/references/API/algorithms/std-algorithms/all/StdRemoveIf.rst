``remove_if``
=============

Header: ``Kokkos_StdAlgorithms.hpp``

Description
-----------

Removes all elements for which ``pred`` returns ``true``, by shifting via move assignment the elements in a range or in ``View`` such that the elements not to be removed appear in the beginning of the range or in the beginning of ``View``. Relative order of the elements that remain is preserved and the physical size of the container is unchanged.

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.

Overload set accepting execution space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <class ExecutionSpace, class Iterator, class UnaryPredicate>
   Iterator remove_if(const ExecutionSpace& exespace,                           (1)
                      Iterator first, Iterator last,
                      UnaryPredicate pred);

   template <class ExecutionSpace, class Iterator, class UnaryPredicate>
   Iterator remove_if(const std::string& label,                                 (2)
                      const ExecutionSpace& exespace,
                      Iterator first, Iterator last,
                      UnaryPredicate pred);

   template <
     class ExecutionSpace,
     class DataType, class... Properties,
     class UnaryPredicate>
   auto remove_if(const ExecutionSpace& exespace,                               (3)
                  const Kokkos::View<DataType, Properties...>& view,
                  UnaryPredicate pred);

   template <
     class ExecutionSpace,
     class DataType, class... Properties,
     class UnaryPredicate>
   auto remove_if(const std::string& label,                                     (4)
                  const ExecutionSpace& exespace,
                  const Kokkos::View<DataType, Properties...>& view,
                  UnaryPredicate pred);

Overload set accepting a team handle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 4.2

.. code-block:: cpp

   template <class TeamHandleType, class Iterator, class UnaryPredicate>
   KOKKOS_FUNCTION
   Iterator remove_if(const TeamHandleType& teamHandle,                         (5)
                      Iterator first, Iterator last,
                      UnaryPredicate pred);

   template <
     class TeamHandleType,
     class DataType, class... Properties,
     class UnaryPredicate>
   KOKKOS_FUNCTION
   auto remove_if(const TeamHandleType& teamHandle,                             (6)
                  const Kokkos::View<DataType, Properties...>& view,
                  UnaryPredicate pred);

Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. |remove| replace:: ``remove``
.. _remove: ./StdRemove.rst

- ``exespace``, ``first``, ``last``, ``view``: same as in |remove|_

- ``teamHandle``: team handle instance given inside a parallel region when using a TeamPolicy

- ``label``: string forwarded to internal parallel kernels for debugging purposes

  - 1: The default string is "Kokkos::remove_if_iterator_api_default".

  - 3: The default string is "Kokkos::remove_if_view_api_default".

  - NOTE: overloads accepting a team handle do not use a label internally

- ``pred``:

  - *unary* predicate returning ``true`` for the required element to replace; ``pred(v)``
    must be valid to be called from the execution space passed or the execution space
    associated with the team handle, and convertible to bool for every argument ``v``
    of type (possible const) ``value_type``, where ``value_type`` is the value type
    of ``Iterator`` (for 1,2,5) or the value type of ``view`` (for 3,4,6),
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

Iterator to the element *after* the new logical end.
