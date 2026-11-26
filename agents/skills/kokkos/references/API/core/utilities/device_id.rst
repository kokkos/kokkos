``Kokkos::device_id``
=====================

.. role:: cpp(code)
    :language: cpp

Defined in header ``<Kokkos_Core.hpp>``

.. code-block:: cpp

    [[nodiscard]] int device_id() noexcept;  // (since 4.1)

Returns the id of the device that is used by ``DefaultExecutionSpace`` or
``-1`` if only host backends are enabled.

----

**See also**

.. _num_devices : num_devices.rst

.. |num_devices| replace:: ``num_devices``

.. _num_threads : num_threads.rst

.. |num_threads| replace:: ``num_threads``

.. _initialize: ../initialize_finalize/initialize.rst

.. |initialize| replace:: ``initialize``

.. _InitializationSettings: ../initialize_finalize/InitializationSettings.rst

.. |InitializationSettings| replace:: ``InitializationSettings``

|num_devices|_: returns the number of devices available to Kokkos

|num_threads|_: returns the number of threads used by Kokkos

|initialize|_: initializes the Kokkos execution environment

|InitializationSettings|_: settings for initializing Kokkos
