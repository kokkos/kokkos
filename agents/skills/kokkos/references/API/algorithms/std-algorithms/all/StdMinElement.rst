``min_element``
===============

Header: ``<Kokkos_StdAlgorithms.hpp>``

Description
-----------

Finds the smallest element in a range or in a rank-1 ``View`` using either ``operator<`` to compare two elements or a user-provided comparison operator.

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.

Overload set accepting execution space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <class ExecutionSpace, class IteratorType>
   auto min_element(const ExecutionSpace& exespace,                        (1)
                    IteratorType first, IteratorType last);

   template <class ExecutionSpace, class IteratorType>
   auto min_element(const std::string& label,                              (2)
                    const ExecutionSpace& exespace,
                    IteratorType first, IteratorType last);

   template <class ExecutionSpace, class IteratorType, class ComparatorType>
   auto min_element(const ExecutionSpace& exespace,                        (3)
                    IteratorType first, IteratorType last,
                    ComparatorType comp);

   template <class ExecutionSpace, class IteratorType, class ComparatorType>
   auto min_element(const std::string& label,                              (4)
                    const ExecutionSpace& exespace,
                    IteratorType first, IteratorType last,
                    ComparatorType comp);

   template <class ExecutionSpace, class DataType, class... Properties>
   auto min_element(const ExecutionSpace& exespace,                        (5)
                    const ::Kokkos::View<DataType, Properties...>& view);

   template <class ExecutionSpace, class DataType, class... Properties>
   auto min_element(const std::string& label,                              (6)
                    const ExecutionSpace& exespace,
                    const ::Kokkos::View<DataType, Properties...>& view);

   template <class ExecutionSpace, class DataType, class ComparatorType, class... Properties>
   auto min_element(const ExecutionSpace& exespace,                        (7)
                    const ::Kokkos::View<DataType, Properties...>& view,
                    ComparatorType comp);

   template <class ExecutionSpace, class DataType, class ComparatorType, class... Properties>
   auto min_element(const std::string& label,                              (8)
                    const ExecutionSpace& exespace,
                    const ::Kokkos::View<DataType, Properties...>& view,
                    ComparatorType comp);

Overload set accepting a team handle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 4.2

.. code-block:: cpp

   template <class TeamHandleType, class IteratorType>
   KOKKOS_FUNCTION
   auto min_element(const TeamHandleType& teamHandle,                      (9)
                    IteratorType first, IteratorType last);

   template <class TeamHandleType, class DataType, class... Properties>
   KOKKOS_FUNCTION
   auto min_element(const TeamHandleType& teamHandle,                      (10)
                    const ::Kokkos::View<DataType, Properties...>& view);

   template <class TeamHandleType, class IteratorType, class ComparatorType>
   KOKKOS_FUNCTION
   auto min_element(const TeamHandleType& teamHandle,                      (11)
                    IteratorType first, IteratorType last,
                    ComparatorType comp);

   template <class TeamHandleType, class DataType, class ComparatorType,
             class... Properties>
   KOKKOS_FUNCTION
   auto min_element(const TeamHandleType& teamHandle,                      (12)
                    const ::Kokkos::View<DataType, Properties...>& view,
                    ComparatorType comp);

Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``exespace``: execution space instance

- ``teamHandle``: team handle instance given inside a parallel region when using a TeamPolicy

- ``label``: string forwarded to internal parallel kernels for debugging purposes

  - 1 and 3: The default string is "Kokkos::min_element_iterator_api_default".

  - 5 and 7: The default string is "Kokkos::min_element_view_api_default".

  - NOTE: overloads accepting a team handle do not use a label internally

- ``first``, ``last``: range of elements to examine

  - must be *random access iterators*, e.g., returned from ``Kokkos::Experimental::(c)begin/(c)end``

  - must represent a valid range, i.e., ``last >= first`` (checked in debug mode)

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``view``: Kokkos view to examine

  - must be rank-1, and have ``LayoutLeft``, ``LayoutRight``, or ``LayoutStride``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``comp``:

  - *binary* functor returning ``true`` if the first argument is *less than* the second argument;
    ``comp(a,b)`` must be valid to be called from the execution space passed,
    and convertible to bool for every pair of arguments ``a,b`` of type
    ``value_type``, where ``value_type`` is the value type of ``IteratorType`` (for 1,2,3,4)
    or the value type of ``view`` (for 5,6,7,8) and must not modify ``a,b``.

  - must conform to:

  .. code-block:: cpp

     struct Comparator
     {
       KOKKOS_INLINE_FUNCTION
       bool operator()(const value_type & a, const value_type & b) const {
         return /* true if a is less than b, based on your logic of "less than" */;
       }
     };

Return Value
~~~~~~~~~~~~

Returns iterator to the smallest element.

The following special cases apply:

- if several elements are equivalent to the smallest element, it returns the iterator to the *first* such element.

- if the range ``[first, last)`` is empty it returns ``last``.

- if ``view`` is empty, it returns ``Kokkos::Experimental::end(view)``.

Example
~~~~~~~

.. code-block:: cpp

   namespace KE = Kokkos::Experimental;
   Kokkos::View<double*> a("a", 13);
   // fill a somehow

   auto res = KE::min_element(Kokkos::DefaultExecutionSpace(), KE::begin(a), KE::end(a));

   // passing the view directly
   auto res = KE::min_element(Kokkos::DefaultExecutionSpace(), a);


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
   auto res = KE::min_element(Kokkos::DefaultExecutionSpace(), a, CustomLessThanComparator<double>());
