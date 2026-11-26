
``count_if``
============

Header: ``<Kokkos_StdAlgorithms.hpp>``

Description
-----------

Returns the number of elements in a range or in rank-1 ``View`` that satisfy a given unary prediate.

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.

Overload set accepting execution space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <class ExecutionSpace, class IteratorType, class Predicate>
   typename IteratorType::difference_type count_if(const ExecutionSpace& exespace,
						   IteratorType first,
						   IteratorType last,                   (1)
						   Predicate pred);


   template <class ExecutionSpace, class IteratorType, class Predicate>
   typename IteratorType::difference_type count_if(const std::string& label,
						   const ExecutionSpace& exespace,
						   IteratorType first,                  (2)
						   IteratorType last,
						   Predicate pred);

   template <class ExecutionSpace, class DataType, class... Properties,
	     class Predicate>
   auto count_if(const ExecutionSpace& exespace,
		 const ::Kokkos::View<DataType, Properties...>& view,                   (3)
		 Predicate pred);

   template <class ExecutionSpace, class DataType, class... Properties,
	     class Predicate>
   auto count_if(const std::string& label, const ExecutionSpace& exespace,
		 const ::Kokkos::View<DataType, Properties...>& view,                   (4)
		 Predicate pred);


Overload set accepting a team handle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 4.2

.. code-block:: cpp

   template <class TeamHandleType, class IteratorType, class Predicate>
   KOKKOS_FUNCTION
   typename IteratorType::difference_type count_if(const TeamHandleType& teamHandle,
						   IteratorType first,
						   IteratorType last,                   (5)
						   Predicate pred);

   template <class TeamHandleType, class DataType, class... Properties,
	     class Predicate>
   KOKKOS_FUNCTION
   auto count_if(const TeamHandleType& teamHandle,
		 const ::Kokkos::View<DataType, Properties...>& view,                   (6)
		 Predicate pred);

Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``exespace``: execution space instance

- ``teamHandle``: team handle instance given inside a parallel region when using a TeamPolicy

- ``label``: string forwarded to internal parallel kernels for debugging purposes

  - 1: The default string is "Kokkos::count_if_iterator_api_default".

  - 3: The default string is "Kokkos::count_if_view_api_default".

  - NOTE: overloads accepting a team handle do not use a label internally

- ``first, last``: range of elements to search in

  - must be *random access iterators*, e.g., returned from ``Kokkos::Experimental::(c)begin/(c)end``

  - must represent a valid range, i.e., ``last >= first``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``view``:

  - must be rank-1, and have ``LayoutLeft``, ``LayoutRight``, or ``LayoutStride``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``pred``: *unary* functor returning ``true`` if an argument satisfies the desired condition.

  ``pred(v)`` must be valid to be called from the execution space passed, or the execution space
  associated with the team handle, and convertible to bool for every argument ``v``
  of type ``value_type``, where ``value_type`` is the value type of ``IteratorType`` or ``view``
  and must not modify ``v``.

  - must conform to:

  .. code-block:: cpp

     struct CustomPredicate{
       KOKKOS_INLINE_FUNCTION
       bool operator()(const value_type & v) const {
         return /* true if v satisfies your desired condition */;
       }
     };

Return Value
~~~~~~~~~~~~

Returns the number of elements in the range ``first,last`` or in ``view`` for which the predicate is true.
