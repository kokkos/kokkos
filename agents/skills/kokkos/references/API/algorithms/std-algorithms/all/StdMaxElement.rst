``max_element``
===============

Header: ``<Kokkos_StdAlgorithms.hpp>``

Description
-----------

Finds the largest element in a range or in a rank-1 ``View`` using either ``operator<`` to compare two elements or a user-provided comparison operator.

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.


Overload set accepting execution space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <class ExecutionSpace, class IteratorType>
   auto max_element(const ExecutionSpace& exespace,                        (1)
                    IteratorType first, IteratorType last);

   template <class ExecutionSpace, class IteratorType>
   auto max_element(const std::string& label,                              (2)
                    const ExecutionSpace& exespace,
                    IteratorType first, IteratorType last);

   template <class ExecutionSpace, class IteratorType, class ComparatorType>
   auto max_element(const ExecutionSpace& exespace,                        (3)
                    IteratorType first, IteratorType last,
                    ComparatorType comp);

   template <class ExecutionSpace, class IteratorType, class ComparatorType>
   auto max_element(const std::string& label,                              (4)
                    const ExecutionSpace& exespace,
                    IteratorType first, IteratorType last,
                    ComparatorType comp);

   template <class ExecutionSpace, class DataType, class... Properties>
   auto max_element(const ExecutionSpace& exespace,                        (5)
                    const ::Kokkos::View<DataType, Properties...>& view);

   template <class ExecutionSpace, class DataType, class... Properties>
   auto max_element(const std::string& label,                              (6)
                    const ExecutionSpace& exespace,
                    const ::Kokkos::View<DataType, Properties...>& view);

   template <class ExecutionSpace, class DataType, class ComparatorType, class... Properties>
   auto max_element(const ExecutionSpace& exespace,                        (7)
                    const ::Kokkos::View<DataType, Properties...>& view,
                    ComparatorType comp);

   template <class ExecutionSpace, class DataType, class ComparatorType, class... Properties>
   auto max_element(const std::string& label,                              (8)
                    const ExecutionSpace& exespace,
                    const ::Kokkos::View<DataType, Properties...>& view,
                    ComparatorType comp);

Overload set accepting a team handle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 4.2

.. code-block:: cpp

   template <class TeamHandleType, class IteratorType>
   KOKKOS_FUNCTION
   auto max_element(const TeamHandleType& teamHandle,                      (9)
                    IteratorType first, IteratorType last);

   template <class TeamHandleType, class DataType, class... Properties>
   KOKKOS_FUNCTION
   auto max_element(const TeamHandleType& teamHandle,                      (10)
                    const ::Kokkos::View<DataType, Properties...>& view);

   template <class TeamHandleType, class IteratorType, class ComparatorType>
   KOKKOS_FUNCTION
   auto max_element(const TeamHandleType& teamHandle,                      (11)
                    IteratorType first, IteratorType last,
                    ComparatorType comp);

   template <class TeamHandleType, class DataType, class ComparatorType,
             class... Properties>
   KOKKOS_FUNCTION
   auto max_element(const TeamHandleType& teamHandle,                      (12)
                    const ::Kokkos::View<DataType, Properties...>& view,
                    ComparatorType comp);

Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _min_element_link: ./StdMinElement.rst

.. |min_element_link| replace:: ``min_element``

- ``exespace``, ``first``, ``last``, ``view``, ``comp``: same as in |min_element_link|_

- ``teamHandle``: team handle instance given inside a parallel region when using a TeamPolicy

- ``label``: string forwarded to internal parallel kernels for debugging purposes

  - 1 and 3: The default string is "Kokkos::max_element_iterator_api_default".

  - 5 and 7: the default string is "Kokkos::max_element_view_api_default".

  - NOTE: overloads accepting a team handle do not use a label internally

Return Value
~~~~~~~~~~~~

Iterator to the largest element.

The following special cases apply:

- if several elements are equivalent to the largest element, returns the iterator to the *first* such element.

- if the range ``[first, last)`` is empty it returns ``last``.

- if ``view`` is empty, it returns ``Kokkos::Experimental::end(view)``.

Example
~~~~~~~

.. code-block:: cpp

   namespace KE = Kokkos::Experimental;
   Kokkos::View<double*> a("a", 13);
   // fill a somehow

   auto res = KE::max_element(Kokkos::DefaultExecutionSpace(), KE::begin(a), KE::end(a));

   // passing the view directly
   auto res = KE::max_element(Kokkos::DefaultExecutionSpace(), a);


   // using a custom comparator
   template <class ValueType1, class ValueType2 = ValueType1>
   struct CustomLessThanComparator {
     KOKKOS_INLINE_FUNCTION
     bool operator()(const ValueType1& a,
                     const ValueType2& b) const {
       // here we use < but one can put any custom logic to return true if a is less than b
       return a < b;
     }

     KOKKOS_INLINE_FUNCTION
     CustomLessThanComparator() {}
   };

   // passing the view directly
   auto res = KE::max_element(Kokkos::DefaultExecutionSpace(), a, CustomLessThanComparator<double>());
