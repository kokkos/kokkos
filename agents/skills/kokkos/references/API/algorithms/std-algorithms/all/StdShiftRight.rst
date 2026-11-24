``shift_right``
===============

Header: ``Kokkos_StdAlgorithms.hpp``

Description
-----------

Shifts the elements in a range or in ``view`` by ``n`` positions towards the end of the range or the view.

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.

Overload set accepting execution space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <class ExecutionSpace, class IteratorType>
   IteratorType shift_right(const ExecutionSpace& exespace,                  (1)
                            IteratorType first, IteratorType last,
                            typename IteratorType::difference_type n);

   template <class ExecutionSpace, class IteratorType>
   IteratorType shift_right(const std::string& label,                        (2)
                            const ExecutionSpace& exespace,
                            IteratorType first, IteratorType last,
                            typename IteratorType::difference_type n);

   template <class ExecutionSpace, class DataType, class... Properties>
   auto shift_right(const ExecutionSpace& exespace,                          (3)
                    const Kokkos::View<DataType, Properties...>& view,
                    typename decltype(begin(view))::difference_type n);

   template <class ExecutionSpace, class DataType, class... Properties>
   auto shift_right(const std::string& label,                                (4)
                    const ExecutionSpace& exespace,
                    const Kokkos::View<DataType, Properties...>& view,
                    typename decltype(begin(view))::difference_type n);


Overload set accepting a team handle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 4.2

.. code-block:: cpp

   template <class TeamHandleType, class IteratorType>
   KOKKOS_FUNCTION
   IteratorType shift_right(const TeamHandleType& teamHandle,                (5)
                            IteratorType first, IteratorType last,
                            typename IteratorType::difference_type n);

   template <class TeamHandleType, class DataType, class... Properties>
   KOKKOS_FUNCTION
   auto shift_right(const TeamHandleType& teamHandle,                        (6)
                    const Kokkos::View<DataType, Properties...>& view,
                    typename decltype(begin(view))::difference_type n);

Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. |ShiftLeft| replace:: ``shift_left``
.. _ShiftLeft: ./StdShiftLeft.rst

- ``exespace`` ``teamHandle``, ``first``, ``last``, ``view``: same as in |ShiftLeft|_

- ``label``: string forwarded to internal parallel kernels for debugging purposes

  - 1: The default string is "Kokkos::shift_right_iterator_api_default"

  - 3: The default string is "Kokkos::shift_right_view_api_default"

  - NOTE: overloads accepting a team handle do not use a label internally

- ``n``: the number of positions to shift

  - must be non-negative

Return Value
~~~~~~~~~~~~

The beginning of the resulting range. If ``n`` is less than ``last - first``, returns ``first + n``. Otherwise, returns ``last``.
