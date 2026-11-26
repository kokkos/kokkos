``copy_n``
==========

Header: ``<Kokkos_StdAlgorithms.hpp>``

Description
-----------

Copies the first ``n`` elements from a source range or rank-1 ``View`` to another range or rank-1 ``View``.

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.

Overload set accepting execution space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

  template <
    class ExecutionSpace, class InputIteratorType,
    class SizeType, class OutputIteratorType>
  OutputIteratorType copy_n(const ExecutionSpace& exespace,                    (1)
                            InputIteratorType first_from,
                            SizeType n,
                            OutputIteratorType first_to);

  template <
    class ExecutionSpace, class InputIteratorType,
    class SizeType, class OutputIteratorType>
  OutputIteratorType copy_n(const std::string & label,
                            const ExecutionSpace& exespace,                    (2)
                            InputIteratorType first_from,
                            SizeType n,
                            OutputIteratorType first_to);

  template <
    class ExecutionSpace,
    class DataType1, class... Properties1,
    class SizeType,
    class DataType2, class... Properties2>
  auto copy_n(const ExecutionSpace& exespace,                                  (3)
              const Kokkos::View<DataType1, Properties1...>& view_from,
              SizeType n,
              const Kokkos::View<DataType2, Properties2...>& view_to);

  template <
    class ExecutionSpace,
    class DataType1, class... Properties1,
    class SizeType,
    class DataType2, class... Properties2>
  auto copy_n(const std::string& label, const ExecutionSpace& exespace,        (4)
              const Kokkos::View<DataType1, Properties1...>& view_from,
              SizeType n,
              const Kokkos::View<DataType2, Properties2...>& view_to);

Overload set accepting a team handle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 4.2

.. code-block:: cpp

  template <
    class TeamHandleType, class InputIteratorType,
    class SizeType, class OutputIteratorType>
  KOKKOS_FUNCTION
  OutputIteratorType copy_n(const TeamHandleType& teamHandle,                 (5)
                            InputIteratorType first_from,
                            SizeType n,
			    OutputIteratorType first_to);

  template <
    class TeamHandleType,
    class DataType1, class... Properties1, class SizeType,
    class DataType2, class... Properties2>
  KOKKOS_FUNCTION
  auto copy_n(const TeamHandleType& teamHandle,                               (6)
              const ::Kokkos::View<DataType1, Properties1...>& view_from,
	      SizeType n,
              ::Kokkos::View<DataType2, Properties2...>& view_to);

Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. |copy| replace:: ``copy``
.. _copy: ./StdCopy.rst


- ``exespace``, ``teamHandle``, ``first_from``, ``first_to``, ``view_from``, ``view_to``: same as in |copy|_

- ``label``: used to name the implementation kernels for debugging purposes

  - for 1, the default string is: "Kokkos::copy_n_if_iterator_api_default"

  - for 3, the default string is: "Kokkos::copy_n_if_view_api_default"

  - NOTE: overloads accepting a team handle do not use a label internally

- ``n``: number of elements to copy (must be non-negative)


Return Value
~~~~~~~~~~~~

If ``n>0``, returns an iterator to the destination element *after* the last element copied.

Otherwise, returns ``first_to`` (for 1,2,5) or ``Kokkos::begin(view_to)`` (for 3,4,6).
