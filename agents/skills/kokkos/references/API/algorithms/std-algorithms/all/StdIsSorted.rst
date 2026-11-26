
``is_sorted``
=============

Header: ``<Kokkos_StdAlgorithms.hpp>``

Description
-----------

Checks if the elements in a range or in a rank-1 ``View`` are sorted in non-descending order using either ``operator<`` to compare two elements or a user-provided comparison operator.

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.

Overload set accepting execution space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <class ExecutionSpace, class IteratorType>
   bool is_sorted(const ExecutionSpace& exespace,                              (1)
                  IteratorType first, IteratorType last);

   template <class ExecutionSpace, class IteratorType>
   bool is_sorted(const std::string& label,                                    (2)
                  const ExecutionSpace& exespace,
                  IteratorType first, IteratorType last);

   template <class ExecutionSpace, class DataType, class... Properties>
   bool is_sorted(const ExecutionSpace& exespace,                              (3)
                  const ::Kokkos::View<DataType, Properties...>& view);

   template <class ExecutionSpace, class DataType, class... Properties>
   bool is_sorted(const std::string& label, const ExecutionSpace& exespace,    (4)
                  const ::Kokkos::View<DataType, Properties...>& view);

   template <class ExecutionSpace, class IteratorType, class ComparatorType>
   bool is_sorted(const ExecutionSpace& exespace,                              (5)
                  IteratorType first, IteratorType last,
                  ComparatorType comp);

   template <class ExecutionSpace, class IteratorType, class ComparatorType>
   bool is_sorted(const std::string& label, const ExecutionSpace& exespace,    (6)
                  IteratorType first, IteratorType last,
                  ComparatorType comp);

   template <class ExecutionSpace, class DataType, class... Properties,
             class ComparatorType>
   bool is_sorted(const ExecutionSpace& exespace,                              (7)
                  const ::Kokkos::View<DataType, Properties...>& view,
                  ComparatorType comp);

   template <class ExecutionSpace, class DataType, class... Properties,
             class ComparatorType>
   bool is_sorted(const std::string& label, const ExecutionSpace& exespace,    (8)
                  const ::Kokkos::View<DataType, Properties...>& view,
                  ComparatorType comp);

Overload set accepting a team handle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 4.2

.. code-block:: cpp

   template <class TeamHandleType, class IteratorType>
   KOKKOS_FUNCTION
   bool is_sorted(const TeamHandleType& teamHandle,                            (9)
                  IteratorType first, IteratorType last);

   template <class TeamHandleType, class DataType, class... Properties>
   KOKKOS_FUNCTION
   bool is_sorted(const TeamHandleType& teamHandle,                            (10)
                  const ::Kokkos::View<DataType, Properties...>& view);

   template <class TeamHandleType, class IteratorType, class ComparatorType>
   KOKKOS_FUNCTION
   bool is_sorted(const TeamHandleType& teamHandle,                            (11)
                  IteratorType first, IteratorType last,
                  ComparatorType comp);

   template <class TeamHandleType, class DataType, class... Properties,
             class ComparatorType>
   KOKKOS_FUNCTION
   bool is_sorted(const TeamHandleType& teamHandle,                            (12)
                  const ::Kokkos::View<DataType, Properties...>& view,
                  ComparatorType comp);


Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``exespace``: execution space instance

- ``teamHandle``: team handle instance given inside a parallel region when using a TeamPolicy

- ``label``: string forwarded to internal parallel kernels for debugging purposes

  - 1: The default string is "Kokkos::is_sorted_iterator_api_default".

  - 3: The default string is "Kokkos::is_sorted_view_api_default".

  - 5: The default string is "Kokkos::is_sorted_iterator_api_default".

  - 7: The default string is "Kokkos::is_sorted_view_api_default".

  - NOTE: overloads accepting a team handle do not use a label internally

- ``first, last``: range of elements to search in

  - must be *random access iterators*, e.g., returned from ``Kokkos::Experimental::(c)begin/(c)end``

  - must represent a valid range, i.e., ``last >= first``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``view``:

  - must be rank-1, and have ``LayoutLeft``, ``LayoutRight``, or ``LayoutStride``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``comp``:

  - *binary* functor returning ``true`` if the first argument is *less than* the second argument;
    ``comp(a,b)`` must be valid to be called from the execution space passed,
    and convertible to bool for every pair of arguments ``a,b`` of type ``value_type``,
    where ``value_type`` is the value type of ``IteratorType`` (for 1,2,5,6)
    or the value type of ``view`` (for 3,4,7,8) and must not modify ``a,b``.

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

Returns ``true`` if the elements are sorted in descending order.
