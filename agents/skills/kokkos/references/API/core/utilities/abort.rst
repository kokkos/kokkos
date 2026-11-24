``Kokkos::abort``
=================

.. role:: cpp(code)
    :language: cpp

Defined in header ``<Kokkos_Core.hpp>``

.. code-block:: cpp

    KOKKOS_FUNCTION void abort(const char *const msg);

Causes abnormal program termination with error explanatory string being printed.

Notes
-----

.. _KokkosAbort: https://github.com/kokkos/kokkos/blob/4.2.00/core/src/Kokkos_Abort.hpp

.. |KokkosAbort| replace:: ``<Kokkos_Abort.hpp>``

* Since version 4.2, one may include |KokkosAbort|_ instead of ``<Kokkos_Core.hpp>``.
