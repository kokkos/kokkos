# Bazel host backends

This integration builds Kokkos Core, Containers, and Algorithms with C++20.
It supports Serial (the default), OpenMP, and Threads on Linux and macOS with
GCC or Clang toolchains. Accelerator backends, Windows, architecture tuning,
SIMD, and the full CMake option set are not implemented by these rules.
Use the CMake build for those configurations.

## Build and test

Install Bazelisk, CMake 3.22 or newer, and a C++20 compiler. Make `cmake`
available on `PATH`, then run from the repository root:

```sh
bazelisk test --jobs="$(nproc)" //bazel:smoke_test
bazelisk test --jobs="$(nproc)" --//:backend=threads //bazel:smoke_test
bazelisk test --jobs="$(nproc)" --//:backend=openmp //bazel:smoke_test
```

The checked-in `.bazelversion` selects Bazel. Set `CC` and `CXX` to select the
compiler for Bazel's local C++ toolchain, or register your own toolchain.
The smoke test initializes Kokkos, fills a View in parallel, reduces its
contents, and checks a host mirror. It runs with the selected backend.

Serial requires no OpenMP runtime. GCC OpenMP builds use the compiler's
libgomp runtime. Linux Clang OpenMP builds use the Bazel `openmp` module.
On macOS, Clang OpenMP builds require libomp headers and libraries available
to the selected toolchain. Configure its include and library search paths
for your libomp installation. Apple Clang uses `-Xpreprocessor -fopenmp`.

Linux links libdl explicitly. macOS provides the dynamic loader interfaces
through its system libraries. Other operating systems are marked incompatible.

## Use as a dependency

Depend on `@kokkos//:kokkos` from a `cc_library`, `cc_binary`, or `cc_test`.
Select a threaded backend with `--@kokkos//:backend=threads` or
`--@kokkos//:backend=openmp`. Serial remains enabled alongside either backend.

Kokkos's own library and smoke test explicitly compile as C++20. Its `.bazelrc`
also sets C++20 for standalone builds. Bazel does not load dependency `.bazelrc`
files or propagate `copts` to consumers, so downstream projects must configure
C++20 or newer in their toolchain or pass `--cxxopt=-std=c++20`.

OpenMP consumers must also compile their template instantiations with OpenMP:
use `--cxxopt=-fopenmp` for GCC or Linux Clang, and
`--cxxopt=-Xpreprocessor --cxxopt=-fopenmp` for Apple Clang. Runtime link
options and the Linux Clang runtime dependency propagate from `:kokkos`.
All consumers must use a consistent compiler, C++ standard, and backend.

## Configuration maintenance

The header generator uses `cmake/KokkosCore_config.h.in`, reads the Kokkos
version from the shared `cmake/kokkos_version.cmake`, and reads bundled dependency revisions from
`tpls/*-hash.txt`. The module declaration deliberately omits a separate
version snapshot. Registry releases identify the published module version.

Feature settings are maintained explicitly in `bazel/generate_config.cmake`.
The generator enables C++20, Serial, the selected host backend, dynamic loader
support, deprecated APIs, deprecation warnings, complex alignment, and the
reference-count branch hint. Other features remain undefined, including debug
features regardless of Bazel compilation mode. CMake option defaults are not
read or evaluated. Changes to upstream configuration logic and new template
options should be reviewed for Bazel compatibility. Bazel runs the script with
`cmake -P`, and CMake expands the header using `configure_file(... @ONLY)`.
This does not configure the full Kokkos project or perform compiler detection.
