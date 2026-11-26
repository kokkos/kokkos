
``find_end``
============

Header: ``<Kokkos_StdAlgorithms.hpp>``

Description
-----------

Searches a given range or rank-1 ``View`` for the *last* occurrence
of a target sequence or ``View`` of values.

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.

Overload set accepting execution space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <class ExecutionSpace, class IteratorType1, class IteratorType2>
   IteratorType1 find_end(const ExecutionSpace& exespace,                                (1)
                          IteratorType1 first, IteratorType1 last,
			  IteratorType2 s_first, IteratorType2 s_last);

   template <class ExecutionSpace, class IteratorType1, class IteratorType2>
   IteratorType1 find_end(const std::string& label, const ExecutionSpace& exespace,
			  IteratorType1 first, IteratorType1 last,                       (2)
			  IteratorType2 s_first, IteratorType2 s_last);

   template <class ExecutionSpace, class DataType1, class... Properties1,
	     class DataType2, class... Properties2>
   auto find_end(const ExecutionSpace& exespace,
		 const ::Kokkos::View<DataType1, Properties1...>& view,                  (3)
		 const ::Kokkos::View<DataType2, Properties2...>& s_view);

   template <class ExecutionSpace, class DataType1, class... Properties1,
	     class DataType2, class... Properties2>
   auto find_end(const std::string& label, const ExecutionSpace& exespace,
		 const ::Kokkos::View<DataType1, Properties1...>& view,                  (4)
		 const ::Kokkos::View<DataType2, Properties2...>& s_view);

   template <class ExecutionSpace, class IteratorType1, class IteratorType2,
	     class BinaryPredicateType>
   IteratorType1 find_end(const ExecutionSpace& exespace,                                (5)
                          IteratorType1 first, IteratorType1 last,
			  IteratorType2 s_first, IteratorType2 s_last,
			  const BinaryPredicateType& pred);

   template <class ExecutionSpace, class IteratorType1, class IteratorType2,
	     class BinaryPredicateType>
   IteratorType1 find_end(const std::string& label, const ExecutionSpace& exespace,      (6)
			  IteratorType1 first, IteratorType1 last,
			  IteratorType2 s_first, IteratorType2 s_last,
			  const BinaryPredicateType& pred);

   template <class ExecutionSpace, class DataType1, class... Properties1,
	     class DataType2, class... Properties2, class BinaryPredicateType>
   auto find_end(const ExecutionSpace& exespace,                                         (7)
		 const ::Kokkos::View<DataType1, Properties1...>& view,
		 const ::Kokkos::View<DataType2, Properties2...>& s_view,
		 const BinaryPredicateType& pred);

   template <class ExecutionSpace, class DataType1, class... Properties1,
	     class DataType2, class... Properties2, class BinaryPredicateType>
   auto find_end(const std::string& label, const ExecutionSpace& exespace,               (8)
		 const ::Kokkos::View<DataType1, Properties1...>& view,
		 const ::Kokkos::View<DataType2, Properties2...>& s_view,
		 const BinaryPredicateType& pred);

Overload set accepting a team handle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 4.2

.. code-block:: cpp

   template <class TeamHandleType, class IteratorType1, class IteratorType2>
   KOKKOS_FUNCTION
   IteratorType1 find_end(const TeamHandleType& teamHandle,                              (9)
                          IteratorType1 first, IteratorType1 last,
			  IteratorType2 s_first, IteratorType2 s_last);

   template <class TeamHandleType, class DataType1, class... Properties1,
	     class DataType2, class... Properties2>
   KOKKOS_FUNCTION
   auto find_end(const TeamHandleType& teamHandle,                                      (10)
		 const ::Kokkos::View<DataType1, Properties1...>& view,
		 const ::Kokkos::View<DataType2, Properties2...>& s_view);

   template <class TeamHandleType, class IteratorType1, class IteratorType2,
	     class BinaryPredicateType>
   KOKKOS_FUNCTION
   IteratorType1 find_end(const TeamHandleType& teamHandle,                             (11)
                          IteratorType1 first, IteratorType1 last,
			  IteratorType2 s_first, IteratorType2 s_last,
			  const BinaryPredicateType& pred);

   template <class TeamHandleType, class DataType1, class... Properties1,
	     class DataType2, class... Properties2, class BinaryPredicateType>
   KOKKOS_FUNCTION
   auto find_end(const TeamHandleType& teamHandle,                                      (12)
		 const ::Kokkos::View<DataType1, Properties1...>& view,
		 const ::Kokkos::View<DataType2, Properties2...>& s_view,
		 const BinaryPredicateType& pred);

Overload Set Detailed Description
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- 1,2,5,6: searches for the last occurrence of the sequence ``[s_first, s_last)``
  in the range ``[first, last)`` comparing elements via ``operator ==`` (1,2) or via ``pred`` (5,6)

- 3,4,7,8: searches for the last occurrence of the ``s_view`` in ``view``
  comparing elements via ``operator ==`` (3,4 or via ``pred`` (7,8)

Parameters and Requirements
---------------------------

- ``exespace``: execution space instance

- ``teamHandle``: team handle instance given inside a parallel region when using a TeamPolicy

- ``label``: string forwarded to internal parallel kernels for debugging purposes

  - 1,5: The default string is "Kokkos::find_end_iterator_api_default".

  - 3,7: The default string is "Kokkos::find_end_view_api_default".

  - NOTE: overloads accepting a team handle do not use a label internally

- ``first, last``: range of elements to search in

  - must be *random access iterators*, e.g., returned from ``Kokkos::Experimental::(c)begin/(c)end``

  - must represent a valid range, i.e., ``last >= first``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``s_first, s_last``: range of elements that you want to search for

  - same requirements as ``first, last``

- ``view``, ``s_view``: views to search in and for, respectively

  - must be rank-1, and have ``LayoutLeft``, ``LayoutRight``, or ``LayoutStride``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``pred``: *binary* functor returning ``true`` if two arguments should be considered "equal".

  ``pred(a,b)`` must be valid to be called from the execution space passed, or
  the execution space associated with the team handle, and convertible to bool
  for every pair of arguments ``a,b`` of type ``ValueType1`` and ``ValueType2``,
  respectively, where ``ValueType1`` and ``ValueType{1,2}`` are the value types of
  ``IteratorType{1,2}`` or ``(s_)view``, and must not modify ``a,b``.

  - must conform to:

  .. code-block:: cpp

     template <class ValueType1, class ValueType2 = ValueType1>
     struct IsEqualFunctor {
      KOKKOS_INLINE_FUNCTION
      bool operator()(const ValueType1& a, const ValueType2& b) const {
        return (a == b);
      }
     };


Return Value
~~~~~~~~~~~~

Iterator to the beginning of the last occurrence of the sequence ``[s_first, s_last)``
in range ``[first, last)``, or the last occurrence of ``s_view`` in ``view``.

If ``[s_first, s_last)`` or ``[first, last)`` is empty, ``last`` is returned.

If ``view`` or ``s_view`` is empty, ``Kokkos::Experimental::end(view)`` is returned.
