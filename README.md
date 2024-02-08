[![Kokkos](https://avatars2.githubusercontent.com/u/10199860?s=200&v=4)](https://kokkos.org)

# Kokkos: Core Libraries

Kokkos Core implements a programming model in C++ for writing performance portable
applications targeting all major HPC platforms. For that purpose it provides
abstractions for both parallel execution of code and data management.
Kokkos is designed to target complex node architectures with N-level memory
hierarchies and multiple types of execution resources. It currently can use
CUDA, HIP, SYCL, HPX, OpenMP and C++ threads as backend programming models with several other
backends in development.

**Kokkos Core is part of the [Kokkos C++ Performance Portability Programming EcoSystem](https://kokkos.org/about/abstract/).**

## Learning about Kokkos

To start learning about Kokkos:

- [Kokkos Lectures](https://kokkos.org/kokkos-core-wiki/videolectures.html): they contain a mix of lecture videos and hands-on exercises covering all the important Kokkos Ecosystem capabilities.

- [Programming guide](https://kokkos.org/kokkos-core-wiki/programmingguide.html): contains in "narrative" form a technical description of the programming model, machine model, and the main building blocks like the Views and parallel dispatch.

- [API reference](https://kokkos.org/kokkos-core-wiki/): organized by category, i.e., [core](https://kokkos.org/kokkos-core-wiki/API/core-index.html), [algorithms](https://kokkos.org/kokkos-core-wiki/API/algorithms-index.html) and [containers](https://kokkos.org/kokkos-core-wiki/API/containers-index.html) or, if you prefer, in [alphabetical order](https://kokkos.org/kokkos-core-wiki/API/alphabetical.html).

- [Use cases and Examples](https://kokkos.org/kokkos-core-wiki/usecases.html): a series of examples ranging from how to use Kokkos with MPI to Fortran interoperability.

## Quick Start for the impatient

### Obtaining Kokkos

The latest release of Kokkos can be obtained from the [GitHub releases page](https://github.com/kokkos/kokkos/releases/latest)
or programmatically using curl to get the latest release tarball and extract it into a directory called `kokkos`:

```bash
mkdir kokkos && curl -s -L https://github.com/kokkos/kokkos/archive/refs/tags/4.2.00.tar.gz | tar xz - -C kokkos --strip-components 1
```

To clone the latest development version of Kokkos from GitHub:

```bash
git clone -b develop  https://github.com/kokkos/kokkos.git
```

### Building Kokkos

To build Kokkos, you will need to have a C++ compiler that supports C++14 or later.
All requirements including minimum and primary tested compiler versions can be found [here](https://kokkos.org/kokkos-core-wiki/requirements.html).

Building and installation instructions are described [here](https://kokkos.org/kokkos-core-wiki/building.html).

#### Building for CPU on Linux or macOS

To use the OpenMP backend, you will need a compiler that supports OpenMP, such as GCC, Clang, or Intel C++ Compiler.
On MacOS, you will need to install a recent version of `libomp` for example using Homebrew: `brew install libomp`.

To build Kokkos targetting OpenMP for CPU on Linux or macOS, you can use the following commands:

```bash
cd kokkos
cmake -B build -S . -DKokkos_ENABLE_OPENMP=On -DCMAKE_INSTALL_PREFIX=/opt/kokkos
cmake --build build && cmake --install build
```

Note that you can also use `Kokkos_ENABLE_THREADS` to enable the use of threads instead of OpenMP.

#### Building for CUDA on Linux

To build Kokkos targetting CUDA on Linux, you can use the following commands:

```bash
cd kokkos
cmake -B build -S . -DKokkos_ENABLE_CUDA=On -DCMAKE_INSTALL_PREFIX=/opt/kokkos
cmake --build build && cmake --install build
```

#### Building for HIP on Linux

To build Kokkos targetting HIP on Linux, you can use the following commands:

```bash
cd kokkos
cmake -B build -S . -DKokkos_ENABLE_HIP=On -DCMAKE_INSTALL_PREFIX=/opt/kokkos
cmake --build build && cmake --install build
```

#### Installing Kokkos with Spack

You can also install Kokkos using [Spack](https://spack.io/): `spack install kokkos`. [Available configuration options](https://packages.spack.io/package.html?name=kokkos) can be displayed using `spack info kokkos`.


### Using Kokkos

To use Kokkos in your CMake project, you can use the following commands in your `CMakeLists.txt`:

```cmake
find_package(Kokkos REQUIRED)
add_executable(myapp myapp.cpp)
target_link_libraries(myapp Kokkos::kokkos)
```

To compile your application, you can use the following commands:

```bash
cmake -B build -S . -DCMAKE_INSTALL_PREFIX=/opt/kokkos && cmake --build build
```

More examples can be found in the [example](https://github.com/kokkos/kokkos/example) directory of the Kokkos source code.

## For the complete documentation: [kokkos.org/kokkos-core-wiki/](https://kokkos.org/kokkos-core-wiki/)

## Support

For questions find us on Slack: https://kokkosteam.slack.com or open a GitHub issue.

For non-public questions send an email to: *crtrott(at)sandia.gov*

## Contributing to Kokkos

Please see [this page](https://kokkos.org/kokkos-core-wiki/contributing.html) for details on how to contribute.

## Requirements, Building and Installing



## Citing Kokkos

Please see the [following page](https://kokkos.org/kokkos-core-wiki/citation.html).

## License

[![License](https://img.shields.io/badge/License-Apache--2.0_WITH_LLVM--exception-blue)](https://spdx.org/licenses/LLVM-exception.html)

Under the terms of Contract DE-NA0003525 with NTESS,
the U.S. Government retains certain rights in this software.

The full license statement used in all headers is available [here](https://kokkos.org/kokkos-core-wiki/license.html) or
[here](https://github.com/kokkos/kokkos/blob/develop/LICENSE).
