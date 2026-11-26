
``reverse``
===========

Header: ``<Kokkos_StdAlgorithms.hpp>``

Description
-----------

Reverses the order of the elements in a range or in rank-1 ``View``.

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.

Overload set accepting execution space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <class ExecutionSpace, class InputIterator>
   void reverse(const ExecutionSpace& ex, InputIterator first, InputIterator last);  (1)

   template <class ExecutionSpace, class InputIterator>
   void reverse(const std::string& label, const ExecutionSpace& ex,                  (2)
                InputIterator first, InputIterator last);

   template <class ExecutionSpace, class DataType, class... Properties>
   void reverse(const ExecutionSpace& ex,                                            (3)
                const ::Kokkos::View<DataType, Properties...>& view);

   template <class ExecutionSpace, class DataType, class... Properties>
   void reverse(const std::string& label, const ExecutionSpace& ex,                  (4)
                const ::Kokkos::View<DataType, Properties...>& view);

Overload set accepting a team handle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 4.2

.. code-block:: cpp

   template <class TeamHandleType, class InputIterator>
   KOKKOS_FUNCTION
   void reverse(const TeamHandleType& teamHandle, InputIterator first,               (5)
                InputIterator last);

   template <class TeamHandleType, class DataType, class... Properties>
   KOKKOS_FUNCTION
   void reverse(const TeamHandleType& teamHandle,                                    (6)
                const ::Kokkos::View<DataType, Properties...>& view);

Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``exespace``: execution space instance

- ``teamHandle``: team handle instance given inside a parallel region when using a TeamPolicy

- ``label``: string forwarded to internal parallel kernels for debugging purposes

  - for 1, the default string is: "Kokkos::reverse_iterator_api_default"

  - for 3, the default string is: "Kokkos::reverse_view_api_default"

  - NOTE: overloads accepting a team handle do not use a label internally

- ``first, last``: range of elements to reverse

  - must be *random access iterators*, e.g., returned from ``Kokkos::Experimental::(c)begin/(c)end``

  - must represent a valid range, i.e., ``last >= first``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``view``:

  - must be rank-1, and have ``LayoutLeft``, ``LayoutRight``, or ``LayoutStride``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

Return Value
~~~~~~~~~~~~

None
