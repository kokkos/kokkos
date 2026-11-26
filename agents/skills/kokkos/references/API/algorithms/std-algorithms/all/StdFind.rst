
``find``
========

Header: ``<Kokkos_StdAlgorithms.hpp>``

Description
-----------

Returns an iterator to the *first* element in a range or rank-1 ``View``
that equals a target value. Equality is checked using ``operator==``.

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.

Overload set accepting execution space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <class ExecutionSpace, class InputIterator, class T>
   InputIterator find(const ExecutionSpace& exespace,                                   (1)
		      InputIterator first, InputIterator last,
		      const T& value);

   template <class ExecutionSpace, class InputIterator, class T>
   InputIterator find(const std::string& label, const ExecutionSpace& exespace,         (2)
		      InputIterator first, InputIterator last,
		      const T& value);

   template <class ExecutionSpace, class DataType, class... Properties, class T>
   auto find(const ExecutionSpace& exespace,                                            (3)
	     const Kokkos::View<DataType, Properties...>& view,
	     const T& value);

   template <class ExecutionSpace, class DataType, class... Properties, class T>
   auto find(const std::string& label, const ExecutionSpace& exespace,                  (4)
	     const Kokkos::View<DataType, Properties...>& view,
	     const T& value);

Overload set accepting a team handle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 4.2

.. code-block:: cpp

   template <class TeamHandleType, class InputIterator, class T>
   KOKKOS_FUNCTION
   InputIterator find(const TeamHandleType& teamHandle,                                 (5)
		      InputIterator first, InputIterator last,
		      const T& value);

   template <class TeamHandleType, class DataType, class... Properties, class T>
   KOKKOS_FUNCTION
   auto find(const TeamHandleType& teamHandle,                                          (6)
	     const Kokkos::View<DataType, Properties...>& view,
	     const T& value);


Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``exespace``: execution space instance

- ``teamHandle``: team handle instance given inside a parallel region when using a TeamPolicy

- ``label``: string forwarded to internal parallel kernels for debugging purposes

  - for 1, the default string is: "Kokkos::find_iterator_api_default"

  - for 3, the default string is: "Kokkos::find_view_api_default"

  - NOTE: overloads accepting a team handle do not use a label internally

- ``first, last``: range of elements to search in

  - must be *random access iterators*, e.g., returned from ``Kokkos::Experimental::(c)begin/(c)end``

  - must represent a valid range, i.e., ``last >= first``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``view``: view to search in

  - must be rank-1, and have ``LayoutLeft``, ``LayoutRight``, or ``LayoutStride``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

Return Value
~~~~~~~~~~~~

- (1,2,5): ``InputIterator`` instance pointing to the first element that equals ``value``,
  or ``last`` if no element is found

- (3,4,6): iterator to the first element that equals ``value``,
  or ``Kokkos::Experimental::end(view)`` if no element is found


Example
-------

.. code-block:: cpp

   namespace KE = Kokkos::Experimental;
   auto exespace = Kokkos::DefaultExecutionSpace;
   using view_type = Kokkos::View<exespace, int*>;
   view_type a("a", 15);
   // fill "a" somehow

   auto exespace = Kokkos::DefaultExecutionSpace;
   auto it1 = KE::find(exespace, KE::cbegin(a), KE::cend(a), 5);

   // assuming OpenMP is enabled and "a" is host-accessible, you can also do
   auto it2 = KE::find(Kokkos::OpenMP(), KE::begin(a), KE::end(a), 5);
