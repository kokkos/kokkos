Initialize and Finalize
=======================

The following functions and classes manage the Kokkos execution environment
environment and resource cleanup:

.. list-table::
   :align: left

   * - :doc:`initialize_finalize/initialize`
     - Initializes Kokkos internal objects and all enabled Kokkos backends.
   * - :doc:`initialize_finalize/finalize`
     - Shutdown Kokkos execution environment and release internally managed resources.
   * - :doc:`initialize_finalize/InitializationSettings`
     - A class representing control knobs of the runtime behavior (such as thread counts or device ID).
   * - :doc:`initialize_finalize/ScopeGuard`
     - RAII-based approach to ensure initialization and finalization are handled correctly.
   * - :doc:`initialize_finalize/push_finalize_hook`
     - Register a function to be called on :cpp:func:`finalize` invocation.
   * - :doc:`initialize_finalize/is_initialized_or_finalized`
     - Query initialization status of Kokkos

.. toctree::
   :hidden:
   :maxdepth: 1

   ./initialize_finalize/initialize
   ./initialize_finalize/finalize
   ./initialize_finalize/InitializationSettings
   ./initialize_finalize/ScopeGuard
   ./initialize_finalize/push_finalize_hook
   ./initialize_finalize/is_initialized_or_finalized
