
``equal``
=========

Header: ``<Kokkos_StdAlgorithms.hpp>``

Description
-----------

Returns true if two ranges or two rank-1 ``View`` s are equal.

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.


Overload set accepting execution space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <class ExecutionSpace, class IteratorType1, class IteratorType2>
   bool equal(const ExecutionSpace& exespace,                                        (1)
              IteratorType1 first1, IteratorType1 last1,
	      IteratorType2 first2);

   template <class ExecutionSpace, class IteratorType1, class IteratorType2>
   bool equal(const std::string& label, const ExecutionSpace& exespace,              (2)
	      IteratorType1 first1, IteratorType1 last1,
	      IteratorType2 first2);

   template <class ExecutionSpace, class IteratorType1, class IteratorType2,
	     class BinaryPredicateType>
   bool equal(const ExecutionSpace& exespace,                                        (3)
              IteratorType1 first1, IteratorType1 last1,
	      IteratorType2 first2,
	      BinaryPredicateType predicate);

   template <class ExecutionSpace, class IteratorType1, class IteratorType2,
	     class BinaryPredicateType>
   bool equal(const std::string& label, const ExecutionSpace& exespace,              (4)
	      IteratorType1 first1, IteratorType1 last1,
	      IteratorType2 first2,
	      BinaryPredicateType predicate);

   template <class ExecutionSpace, class IteratorType1, class IteratorType2>
   bool equal(const ExecutionSpace& exespace, IteratorType1 first1,                  (5)
              IteratorType1 last1, IteratorType2 first2,
	      IteratorType2 last2);

   template <class ExecutionSpace, class IteratorType1, class IteratorType2>
   bool equal(const std::string& label, const ExecutionSpace& exespace,              (6)
	      IteratorType1 first1, IteratorType1 last1,
	      IteratorType2 first2, IteratorType2 last2);

   template <class ExecutionSpace, class IteratorType1, class IteratorType2,
	     class BinaryPredicateType>
   bool equal(const ExecutionSpace& exespace,                                        (7)
	      IteratorType1 first1, IteratorType1 last1,
	      IteratorType2 first2, IteratorType2 last2,
	      BinaryPredicateType predicate);

   template <class ExecutionSpace, class IteratorType1, class IteratorType2,
	     class BinaryPredicateType>
   bool equal(const std::string& label, const ExecutionSpace& exespace,              (8)
	      IteratorType1 first1, IteratorType1 last1,
	      IteratorType2 first2, IteratorType2 last2,
	      BinaryPredicateType predicate);

   template <class ExecutionSpace, class DataType1, class... Properties1,
	     class DataType2, class... Properties2>
   bool equal(const ExecutionSpace& exespace,                                        (9)
	      const Kokkos::View<DataType1, Properties1...>& view1,
              const Kokkos::View<DataType2, Properties2...>& view2);

   template <class ExecutionSpace, class DataType1, class... Properties1,
	     class DataType2, class... Properties2>
   bool equal(const std::string& label, const ExecutionSpace& exespace,             (10)
	      const Kokkos::View<DataType1, Properties1...>& view1,
	      const Kokkos::View<DataType2, Properties2...>& view2);

   template <class ExecutionSpace, class DataType1, class... Properties1,
	     class DataType2, class... Properties2, class BinaryPredicate>
   bool equal(const ExecutionSpace& exespace,                                       (11)
	      const Kokkos::View<DataType1, Properties1...>& view1,
	      const Kokkos::View<DataType2, Properties2...>& view2,
	      BinaryPredicate pred);

   template <class ExecutionSpace, class DataType1, class... Properties1,
	     class DataType2, class... Properties2, class BinaryPredicate>
   bool equal(const std::string& label, const ExecutionSpace& exespace,             (12)
	      const Kokkos::View<DataType1, Properties1...>& view1,
	      const Kokkos::View<DataType2, Properties2...>& view2,
	      BinaryPredicate pred);

Overload set accepting a team handle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 4.2

.. code-block:: cpp

   template <class TeamHandleType, class IteratorType1, class IteratorType2>
   KOKKOS_FUNCTION
   bool equal(const TeamHandleType& teamHandle,                                     (13)
              IteratorType1 first1, IteratorType1 last1,
	      IteratorType2 first2);

   template <class TeamHandleType, class IteratorType1, class IteratorType2,
	     class BinaryPredicateType>
   KOKKOS_FUNCTION
   bool equal(const TeamHandleType& teamHandle,                                     (14)
              IteratorType1 first1, IteratorType1 last1,
	      IteratorType2 first2,
	      BinaryPredicateType predicate);

   template <class TeamHandleType, class IteratorType1, class IteratorType2>
   KOKKOS_FUNCTION
   bool equal(const TeamHandleType& teamHandle,                                     (15)
              IteratorType1 first1, IteratorType1 last1,
	      IteratorType2 first2, IteratorType2 last2);

   template <class TeamHandleType, class IteratorType1, class IteratorType2,
	     class BinaryPredicateType>
   KOKKOS_FUNCTION
   bool equal(const TeamHandleType& teamHandle,                                     (16)
              IteratorType1 first1, IteratorType1 last1,
	      IteratorType2 first2, IteratorType2 last2,
	      BinaryPredicateType predicate);

   template <class TeamHandleType, class DataType1, class... Properties1,
	     class DataType2, class... Properties2>
   KOKKOS_FUNCTION
   bool equal(const TeamHandleType& teamHandle,                                     (17)
	      const Kokkos::View<DataType1, Properties1...>& view1,
	      const Kokkos::View<DataType2, Properties2...>& view2);

   template <class TeamHandleType, class DataType1, class... Properties1,
	     class DataType2, class... Properties2, class BinaryPredicate>
   KOKKOS_FUNCTION
   bool equal(const TeamHandleType& teamHandle,                                     (18)
	      const Kokkos::View<DataType1, Properties1...>& view1,
	      const Kokkos::View<DataType2, Properties2...>& view2,
	      BinaryPredicate pred);


Overload Set Detailed Description
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- (1,2,3,4,13,14): returns true if the range ``[first1, last1)`` is equal to the
  range ``[first2, first2 + (last1 - first1))``, and false otherwise

- (5,6,7,8,15,16): returns true if the range ``[first1, last1)`` is equal
  to the range ``[first2, last2)``, and false otherwise

- (9,10,11,12,17,18): returns true if ``view1`` and ``view2`` are equal and false otherwise

- where applicable, the binary predicate ``pred`` is used to check equality between
  two elements, otherwise ``operator ==`` is used

Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``exespace``: execution space instance

- ``teamHandle``: team handle instance given inside a parallel region when using a TeamPolicy

- ``label``: string forwarded to internal parallel kernels for debugging purposes

  - (1,3,5,7): The default string is "Kokkos::equal_iterator_api_default"

  - (9,11): The default string is "Kokkos::equal_view_api_default"

  - NOTE: overloads accepting a team handle do not use a label internally

- ``first1``, ``last1``, ``first2``, ``last2``: iterators defining the ranges to read and compare

  - must be *random access iterators*, e.g., returned from ``Kokkos::Experimental::(c)begin/(c)end``

  - must represent a valid range, i.e., ``last1 >= first1``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``view1``, ``view2``: views to compare

  - must be rank-1, and have ``LayoutLeft``, ``LayoutRight``, or ``LayoutStride``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``pred``: *binary* functor returning ``true`` if two arguments should be considered "equal".

  ``pred(a,b)`` must be valid to be called from the execution space passed, or
  the execution space associated with the team handle, and convertible to bool
  for every pair of arguments ``a,b`` of type ``ValueType1`` and ``ValueType2``,
  respectively, ``ValueType1`` and ``ValueType{1,2}`` are the value types of
  ``IteratorType{1,2}`` or ``view{1,2}``, and must not modify ``a,b``.

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

If the elements of the two ranges or Views are equal, returns ``true``, otherwise ``false``.

Corner cases when ``false`` is returned:

- if ``view1.extent(0) != view2.extent(1)`` for all overloads accepting Views

- if the length of the range ``[first1, last)`` is not equal to length of ``[first2,last2)``


Example
-------

.. code-block:: cpp

   namespace KE = Kokkos::Experimental;

   template <class ValueType1, class ValueType2 = ValueType1>
   struct IsEqualFunctor {
     KOKKOS_INLINE_FUNCTION
     bool operator()(const ValueType1& a, const ValueType2& b) const {
       return (a == b);
     }
   };

   auto exespace = Kokkos::DefaultExecutionSpace;
   using view_type = Kokkos::View<exespace, int*>;
   view_type a("a", 15);
   view_type b("b", 15);
   // fill a,b somehow

   // create functor
   IsEqualFunctor<int,int> p();

   bool isEqual = KE::equal(exespace, KE::begin(a), KE::end(a),
                            KE::begin(b), KE::end(b) p);

   // To run explicitly on the host (assuming a and b are accessible on Host)
   bool isEqual = KE::equal(Kokkos::DefaultHostExecutionSpace(), KE::begin(a), KE::end(a),
                            KE::begin(b), KE::end(b), p);
