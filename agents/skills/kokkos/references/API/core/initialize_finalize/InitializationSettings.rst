``InitializationSettings``
==========================

.. role:: cpp(code)
   :language: cpp

Defined in header ``<KokkosCore.cpp>``

Usage
-----

.. code-block:: cpp

    auto settings = Kokkos::InitializationSettings()
                    .set_num_threads(8)
                    .set_device_id(0)
                    .set_disable_warnings(false);

``InitializationSettings`` is a class that can be used to define the settings
for initializing Kokkos programmatically without having to call the two
parameter form (``argc`` and ``argv``) of :cpp:func:`initialize`.

Interface
---------

.. cpp:class:: InitializationSettings

   .. cpp:function:: InitializationSettings();

      Constructs a new object that does not contain any value for any of the settings.

   .. cpp:function:: InitializationSettings& set_PARAMETER_NAME(PARAMETER_TYPE value);

      Replaces the content of the ``PARAMETER_NAME`` setting with ``value`` and return a reference to the object. ``value`` must be a valid value for ``PARAMETER_NAME``.

   .. cpp:function:: bool has_PARAMETER_NAME() const;

      Checks whether the object contains a value for the ``PARAMETER_NAME`` setting. Returns ``true`` if it contains a value, ``false`` otherwise.

   .. cpp:function:: PARAMETER_TYPE get_PARAMETER_NAME() const;

      Accesses the contained value for the ``PARAMETER_NAME`` setting. The behavior is undefined if the object does not contain a value for setting ``PARAMETER_NAME``.

The table below summarizes what settings are available.

=======================        ==================    ===========
**PARAMETER_NAME**             **PARAMETER_TYPE**    Description
=======================        ==================    ===========
``num_threads``                ``int``               Number of threads to use with the host parallel backend.  Must be greater than zero.
``device_id``                  ``int``               Device to use with the device parallel backend.  Valid IDs are zero to number of GPU(s) available for execution minus one.
``map_device_id_by``           ``std::string``       Strategy to select a device automatically from the GPUs available for execution. Must be either ``"mpi_rank"`` for round-robin assignment based on the local MPI rank or ``"random"``.
``disable_warnings``           ``bool``              Whether to disable warning messages.
``print_configuration``        ``bool``              Whether to print the configuration after initialization.
``tune_internals``             ``bool``              Whether to allow autotuning internals instead of using heuristics.
``tools_libs``                 ``std::string``       Which tool dynamic library to load. Must either be the full path to library or the name of library if the path is present in the runtime library search path (e.g. ``LD_LIBRARY_PATH``)
``tools_help``                 ``bool``              Query the loaded tool for its command-line options support.
``tools_args``                 ``std::string``       Options to pass to the loaded tool as command-line arguments.
=======================        ==================    ===========

Example
~~~~~~~

.. code-block:: cpp

    #include <Kokkos_Core.hpp>

    int main() {
        Kokkos::initialize(Kokkos::InitializationSettings()
                                .set_print_configuration(true)
                                .set_map_device_id_by("random")
                                .set_num_threads(1));
        // ...
        Kokkos::finalize();
    }

.. seealso::

  :doc:`initialize`
    Start the Kokkos execution environment.
  :doc:`ScopeGuard`
    A RAII-based approach to ensure initialization and finalization are handled
    correctly.
