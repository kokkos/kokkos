
``none_of``
===========

Header: ``<Kokkos_StdAlgorithms.hpp>``

Description
-----------

Returns ``true`` if no element in a range or rank-1 ``View`` satisfies a target unary predicate.

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.

Overload set accepting execution space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <class ExecutionSpace, class IteratorType, class Predicate>
   bool none_of(const ExecutionSpace& exespace,                              (1)
		IteratorType first, IteratorType last,
		Predicate predicate);

   template <class ExecutionSpace, class IteratorType, class Predicate>
   bool none_of(const std::string& label, const ExecutionSpace& exespace,    (2)
		IteratorType first, IteratorType last,
		Predicate predicate);

   template <class ExecutionSpace, class DataType, class... Properties,      (3)
	     class Predicate>
   bool none_of(const ExecutionSpace& exespace,
		const ::Kokkos::View<DataType, Properties...>& v,
		Predicate predicate);

   template <class ExecutionSpace, class DataType, class... Properties,
	     class Predicate>
   bool none_of(const std::string& label, const ExecutionSpace& exespace,    (4)
		const ::Kokkos::View<DataType, Properties...>& v,
		Predicate predicate);

Overload set accepting a team handle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 4.2

.. code-block:: cpp

   template <class TeamHandleType, class IteratorType, class Predicate>
   KOKKOS_FUNCTION
   bool none_of(const TeamHandleType& teamHandle,                            (5)
		IteratorType first, IteratorType last,
		Predicate predicate);

   template <class TeamHandleType, class DataType, class... Properties,
	     class Predicate>
   KOKKOS_FUNCTION
   bool none_of(const TeamHandleType& teamHandle,                           (6)
		const ::Kokkos::View<DataType, Properties...>& v,
		Predicate predicate);


Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``exespace``: execution space instance

- ``teamHandle``: team handle instance given inside a parallel region when using a TeamPolicy

- ``label``: string forwarded to internal parallel kernels for debugging purposes

  - 1: The default string is "Kokkos::none_of_iterator_api_default".

  - 3: The default string is "Kokkos::none_of_view_api_default"

  - NOTE: overloads accepting a team handle do not use a label internally

- ``first, last``: range of elements to search in

  - must be *random access iterators*, e.g., returned from ``Kokkos::Experimental::(c)begin/(c)end``

  - must represent a valid range, i.e., ``last >= first``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``view``:

  - must be rank-1, and have ``LayoutLeft``, ``LayoutRight``, or ``LayoutStride``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``pred``: *unary* functor such that ``pred(v)`` must be valid to be called from the execution space passed,
  or the execution space associated with the team handle, and convertible to bool for every argument ``v``
  of type ``value_type``, where ``value_type`` is the value type of ``IteratorType`` or ``view``
  and must not modify ``v``.

  - must conform to:

  .. code-block:: cpp

     struct CustomPredicate{
       KOKKOS_INLINE_FUNCTION
       bool operator()(const value_type & v) const;
     };

Return Value
~~~~~~~~~~~~

Returns ``true`` if no elements in the range or ``view`` satisfy the unary predicate,
or if the range or ``view`` are empty. Returns ``false`` otherwise.
