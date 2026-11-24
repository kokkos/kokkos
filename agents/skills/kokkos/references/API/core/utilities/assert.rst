``KOKKOS_ASSERT``
=================

.. role:: cpp(code)
    :language: cpp

Defined in header ``<Kokkos_Core.hpp>``

.. code-block:: cpp

    #if defined(NDEBUG) and not defined(KOKKOS_ENABLE_DEBUG)
    #  define KOKKOS_ASSERT(condition) ((void)0)
    #else
    #  define KOKKOS_ASSERT(condition) if (!bool(condition)) /*call Kokkos::abort()*/
    #endif

The definition of the macro ``KOKKOS_ASSERT`` depends on other macros,
``NDEBUG`` and ``KOKKOS_ENABLE_DEBUG``.

If ``NDEBUG`` is defined and ``KOKKOS_ENABLE_DEBUG`` is not
defined at the point in the source code where ``<Kokkos_Assert.hpp>`` or ``<Kokkos_Core.hpp>`` is
included, then assert does nothing.

If ``NDEBUG`` is not defined or ``KOKKOS_ENABLE_DEBUG`` is defined,  then
``KOKKOS_ASSERT`` checks if its argument converted to ``bool`` evaluates to
``false``. If it does, ``KOKKOS_ASSERT`` calls ``Kokkos::abort`` with
diagnostic information that includes the text of expression, as well as the
values of the predefined macros ``__FILE__`` and ``__LINE__``.

Example
-------

.. code-block:: cpp

    int main(int argc, char* argv[]) {
        Kokkos::initialize(argc, argv);
        KOKKOS_ASSERT(Kokkos::is_initialized());  // callable from the host

        Kokkos::parallel_for(1, KOKKOS_LAMBDA(int i) {
          KOKKOS_ASSERT(i == 0);  // also callable from the device side
        });

        Kokkos::finalize();
        assert(Kokkos::is_finalized());  // exclusively callable on the host


Notes
-----

.. _KokkosAssert: https://github.com/kokkos/kokkos/blob/4.2.00/core/src/Kokkos_Assert.hpp

.. |KokkosAssert| replace:: ``<Kokkos_Assert.hpp>``

* Since version 4.2, ``KOKKOS_ASSERT`` is also available from |KokkosAssert|_.
* In contrast to `assert` from the C++ standard library, it is legal to call
  ``KOKKOS_ASSERT`` from a ``KOKKOS_FUNCTION``.

See also
--------
* `Kokkos::abort() <abort.rst>`_ causes abnormal program termination
