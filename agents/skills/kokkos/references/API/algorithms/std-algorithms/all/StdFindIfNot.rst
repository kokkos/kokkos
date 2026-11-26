
``find_if_not``
===============

Header: ``<Kokkos_StdAlgorithms.hpp>``

Description
-----------

Returns an iterator to the *first* element in a range or a View for
which a custom predicate returns ``false``.

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.

Overload set accepting execution space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <class ExecutionSpace, class InputIterator, class PredicateType>
   InputIterator find_if_not(const ExecutionSpace& exespace,                            (1)
		             InputIterator first, InputIterator last,
			     PredicateType pred);

   template <class ExecutionSpace, class InputIterator, class PredicateType>
   InputIterator find_if_not(const std::string& label, const ExecutionSpace& exespace,  (2)
			     InputIterator first, InputIterator last,
			     PredicateType pred);

   template <class ExecutionSpace, class DataType, class... Properties, class PredicateType>
   auto find_if_not(const ExecutionSpace& exespace,                                     (3)
		    const Kokkos::View<DataType, Properties...>& view,
		    PredicateType pred);

   template <class ExecutionSpace, class DataType, class... Properties, class PredicateType>
   auto find_if_not(const std::string& label, const ExecutionSpace& exespace,           (4)
		    const Kokkos::View<DataType, Properties...>& view,
		    PredicateType pred);

Overload set accepting a team handle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 4.2

.. code-block:: cpp

   template <class TeamHandleType, class InputIterator, class PredicateType>
   KOKKOS_FUNCTION
   InputIterator find_if_not(const TeamHandleType& teamHandle,                          (5)
		             InputIterator first, InputIterator last,
			     PredicateType pred);

   template <class TeamHandleType, class DataType, class... Properties, class PredicateType>
   KOKKOS_FUNCTION
   auto find_if_not(const TeamHandleType& teamHandle,                                   (6)
		    const Kokkos::View<DataType, Properties...>& view,
		    PredicateType pred);

Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``exespace``: execution space instance

- ``teamHandle``: team handle instance given inside a parallel region when using a TeamPolicy

- ``label``: string forwarded to internal parallel kernels for debugging purposes

  - for 1, the default string is: "Kokkos::find_if_not_iterator_api_default"

  - for 3, the default string is: "Kokkos::find_if_not_view_api_default"

  - NOTE: overloads accepting a team handle do not use a label internally

- ``first, last``: range of elements to search in

  - must be *random access iterators*, e.g., returned from ``Kokkos::Experimental::(c)begin/(c)end``

  - must represent a valid range, i.e., ``last >= first``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``view``: view to search in

  - must be rank-1, and have ``LayoutLeft``, ``LayoutRight``, or ``LayoutStride``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``pred``: unary predicate which returns ``false`` for the required element;

  ``pred(a)`` must be valid to be called from the execution space passed, or
  the execution space associated with the team handle, and convertible to bool for every
  argument ``a`` of type (possible const) ``value_type``, where ``value_type`` is the value
  type of ``InputIterator`` or ``view``, and must not modify ``a``.

  - must conform to:

    .. code-block:: cpp

       struct Predicate
       {
	  KOKKOS_INLINE_FUNCTION
	  bool operator()(const /*type needed */ & operand) const { return /* ... */; }

	  // or, also valid

	  KOKKOS_INLINE_FUNCTION
	  bool operator()(/*type needed */ operand) const { return /* ... */; }
       };


Return Value
~~~~~~~~~~~~

- (1,2,5): ``InputIterator`` instance pointing to the first element
  where the predicate evaluates to ``false``, or ``last`` if no such element is found

- (3,4,6): iterator to the first element where the predicate evaluates to ``false``,
  or ``Kokkos::Experimental::end(view)`` if no such element is found
