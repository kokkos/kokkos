Core API
########

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Reducer
     - Description
   * - `Initialization and Finalization <core/Initialize-and-Finalize.rst>`__
     - Initialization and finalization of Kokkos.
   * - `View and related <core/View.rst>`__
     - Kokkos MultiDimensional View class and related free functions.
   * - `Parallel Execution/Dispatch <core/ParallelDispatch.rst>`__
     - Parallel Execution Dispatch.
   * - `Built-in Reducers <core/builtin_reducers.rst>`__
     - Built-in Reducers
   * - `Execution Policies <core/Execution-Policies.rst>`__
     - Execution policies.
   * - `Spaces <core/Spaces.rst>`__
     - Description of Memory and Execution Spaces.
   * - `Task-Parallelism <core/Task-Parallelism.rst>`__
     - Creating and dispatching Task Graphs.
   * - `MultiGPU Support <core/MultiGPUSupport.rst>`__
     - Launching kernels on multiple GPUs from one process.
   * - `Atomics <core/atomics.rst>`__
     - Atomics
   * - `Numerics <core/Numerics.rst>`__
     - Common mathematical functions, mathematical constants, numeric traits,
       complex numbers, half-precision floating-point types.
   * - `C-style memory management <core/c_style_memory_management.rst>`__
     - C-style memory management
   * - `Traits <core/Prod.rst>`__
     - Traits
   * - `Kokkos Concepts <core/KokkosConcepts.rst>`__
     - Kokkos Concepts
   * - `STL Compatibility Issues <core/STL-Compatibility.rst>`__
     - Ports of standard C++ capabilities, which otherwise do not work on various hardware platforms.
   * - `Utilities <core/Utilities.rst>`__
     - Utility functionality part of Kokkos Core.
   * - `Detection Idiom <core/Detection-Idiom.rst>`__
     - Used to recognize, in an SFINAE-friendly way, the validity of any C++ expression.
   * - `Macros <core/Macros.rst>`__
     - Global macros defined by Kokkos, used for architectures, general settings, etc.

.. toctree::
   :hidden:
   :maxdepth: 1

   ./core/Initialize-and-Finalize
   ./core/View
   ./core/ParallelDispatch
   ./core/builtin_reducers
   ./core/Execution-Policies
   ./core/Spaces
   ./core/Task-Parallelism
   ./core/MultiGPUSupport
   ./core/atomics
   ./core/Numerics
   ./core/c_style_memory_management
   ./core/Traits
   ./core/KokkosConcepts
   ./core/STL-Compatibility
   ./core/Utilities
   ./core/Detection-Idiom
   ./core/Macros
   ./core/Profiling
