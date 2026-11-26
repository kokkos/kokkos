Initialization
==============

In order to use Kokkos, an initialization call is required. That call is responsible for initializing internal objects and acquiring hardware resources such as threads. Typically, this call should be placed right at the start of a program. If you use both MPI and Kokkos, your program should initialize Kokkos right after calling `MPI_Init`. That way, if MPI sets up process binding masks, Kokkos will get that information and use it for best performance. Your program must also _finalize_ Kokkos when done using it in order to free hardware resources.

Include Headers
---------------

All primary capabilities of Kokkos are provided by the `Kokkos_Core.hpp` header file.
Some capabilities - specifically data structures in the `containers` subpackage and algorithmic capabilities in the `algorithms` subpackage are included via separate header files.
For specific capabilities check their API reference:

- `API: Core <../API/core-index.rst>`_
- `API: Containers <../API/containers-index.rst>`_
- `API: Algorithms <../API/algorithms-index.rst>`_

Initialization by command-line arguments
----------------------------------------

The simplest way to initialize Kokkos is by calling the following function:

.. code-block:: cpp

    Kokkos::initialize(int& argc, char* argv[]);

Just like `MPI_Init`, this function interprets command-line arguments to determine the requested settings. Also like `MPI_Init`, it reserves the right to remove command-line arguments from the input list. This is why it takes `argc` by reference, rather than by value.

During initialization, one or more execution spaces will be initialized and assigned to one of the following aliases.

.. code-block:: cpp

    Kokkos::DefaultExecutionSpace;

.. code-block:: cpp

    Kokkos::DefaultHostExecutionSpace;

`DefaultExecutionSpace` is the execution space used with policies and views where one is not explicitly specified.  Primarily, Kokkos will initialize one of the heterogeneous backends (CUDA, HIP, OpenACC, SYCL) as the `DefaultExecutionSpace` if enabled in the build configuration.  In addition, Kokkos requires a `DefaultHostExecutionSpace`.  The `DefaultHostExecutionSpace` is the default execution space used when host operations are required.  If one of the parallel host execution spaces is enabled in the build environment then `Kokkos::Serial` is only initialized if it is explicitly enabled in the build configuration.  If a parallel host execution space is not enabled in the build configuration, then `Kokkos::Serial` is initialized as the `DefaultHostExecutionSpace`.
Kokkos chooses the two spaces using the following list:

1. `Kokkos::Cuda`
2. `Kokkos::Experimental::HPX`
3. `Kokkos::Experimental::OpenACC`
4. `Kokkos::SYCL`
5. `Kokkos::HIP`
6. `Kokkos::OpenMP`
7. `Kokkos::Threads`
8. `Kokkos::Serial`

The highest execution space in the list that is enabled is Kokkos' default execution space, and the highest enabled host execution space is Kokkos' default host execution space. For example, if  `Kokkos::Cuda`, `Kokkos::OpenMP`, and `Kokkos::Serial` are enabled, then `Kokkos::Cuda` is the default execution space and `Kokkos::OpenMP` is the default host execution space\ :sup:`1`.  In cases where the highest enabled backend is a host parallel execution space the `DefaultExecutionSpace` and the `DefaultHostExecutionSpace` will be the same.

`Kokkos::initialize <../API/Initialize-and-Finalize.rst#kokos-initialize>`_ parses the command line for flags prefixed with `--kokkos-`, and removes all recognized flags. Argument options are given with an equals (`=`) sign. If the same argument occurs more than once, the last one is used. For example, the arguments

    --kokkos-threads=4 --kokkos-threads=3

set the number of threads to 3. :ref:`Table 4.1 <Table_cli-opts>` gives a full list of command-line options.

.. _Table_cli-opts:

Table 4.1: Command-line options for Kokkos::initialize

.. list-table::

  * - Argument
    - Description
  * - :code:`--kokkos-help`
    - print this message
  * - :code:`--kokkos-disable-warnings`     
    - disable kokkos warning messages
  * - :code:`--kokkos-print-configuration` 
    - print configuration
  * - :code:`--kokkos-tune-internals`      
    - allow Kokkos to autotune policies and declare tuning features through the tuning system. If left off, Kokkos uses heuristics.
  * - :code:`--kokkos-num-threads=INT`     
    - specify total number of threads to use for parallel regions on the host
  * - :code:`--kokkos-device-id=INT`
    - specify device id to be used by Kokkos
  * - :code:`--kokkos-map-device-id-by=(random\|mpi_rank)`, default: :code:`mpi_rank`
    - strategy to select device-id automatically from available devices: random or mpi_rank\ :sup:`2`
  * - :code:`--kokkos-tools-libs=STR`      
    - specify which of the tools to use. Must either be full path to library or name of library if the path is present in the runtime library search path (e.g. LD_LIBRARY_PATH)
  * - :code:`--kokkos-tools-help`          
    - query the (loaded) kokkos-tool for its command-line option support (which should then be passed via --kokkos-tools-args="...")
  * - :code:`--kokkos-tools-args=STR`      
    - a single (quoted) string of options which will be whitespace delimited and passed to the loaded kokkos-tool as command-line arguments. E.g. :code:`<EXE> --kokkos-tools-args="-c input.txt"` will pass :code:`<EXE> -c input.txt` as argc/argv to tool

When passing a boolean as a string, the acceptable values are:
 - true, yes, 1
 - false, no, 0

The values are case insensitive.


:sup:`1` This is the preferred set of defaults when CUDA and OpenMP are enabled. If you use a thread-parallel host execution space, we prefer Kokkos' OpenMP backend, as this ensures compatibility of Kokkos' threads with the application's direct use of OpenMP threads. Kokkos cannot promise that its Threads backend will not conflict with the application's direct use of operating system threads.

:sup:`2` The two device-id mapping strategies are:
- random: select a random device from available.
- mpi_rank: select device based on a round robin assignment of local MPI ranks. Works with OpenMPI, MVAPICH, SLURM, and derived implementations. Support for MPICH was added in Kokkos 4.0

Initialization by environment variable
--------------------------------------

Instead of using command-line arguments, one may use environment variables. The environment variables are identical to the arguments in :ref:`Table 4.1 <Table_cli-opts>` but they are upper case and the dash is replaced by an underscore. For example, if we want to set the number of threads to 3, we may use

.. code-block:: sh

  KOKKOS_NUM_THREADS=3


Initialization by struct
------------------------

Instead of giving `Kokkos::initialize() <../API/core/initialize_finalize/initialize.rst>`_ command-line arguments, one may directly pass in initialization parameters using the `Kokkos::InitializationSettings` struct.  If one wants to set options using the struct, one can use the functions `set_xxx` where `xxx` is identical to the arguments in :ref:`Table 4.1 <Table_cli-opts>` where the dash has been replaced by an underscore. To check if a variable has been set, one can use the `has_xxx` functions. Finally, to get the value that was set, one can use the `get_xxx` functions.


If you do not set `num_threads`, Kokkos will try to determine a default value if possible or otherwise set it to 1. In particular, Kokkos can use the `hwloc` library to determine default settings using the assumption that the process binding mask is unique, i.e., that this process does not share any cores with another process. Note that the default value of each parameter is -1.

Here is an example of how to use the struct.

.. code-block:: cpp

    Kokkos::InitializationSettings settings;
    // 8 (CPU) threads
    settings.set_num_threads(8);
    // If Kokkos was built with CUDA enabled, use the GPU with device ID 1.
    settings.set_device_id(1);

    Kokkos::initialize(settings);

Finalization
------------

At the end of each program, Kokkos needs to be shut down in order to free resources; do this by calling `Kokkos::finalize() <../API/core/initialize_finalize/finalize.rst>`_. You may wish to set this to be called automatically at program exit, either by setting an `atexit` hook or by attaching the function to `MPI_COMM_SELF` so that it is called automatically at `MPI_Finalize`.

Example Code
------------

A minimal Kokkos code thus would look like this:

.. code-block:: cpp

    #include<Kokkos_Core.hpp>
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc,argv);
    
      Kokkos::finalize();
    }
