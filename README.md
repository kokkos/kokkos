![Kokkos](https://avatars2.githubusercontent.com/u/10199860?s=200&v=4)

# Kokkos: Core Libraries

Kokkos Core implements a programming model in C++ for writing performance portable
applications targeting all major HPC platforms. For that purpose it provides
abstractions for both parallel execution of code and data management.
Kokkos is designed to target complex node architectures with N-level memory
hierarchies and multiple types of execution resources. It currently can use
CUDA, HIP, SYCL, HPX, OpenMP and C++ threads as backend programming models with several other
backends in development.

**Kokkos Core is part of the Kokkos C++ Performance Portability Programming EcoSystem.**

For the complete documentation, click below:

# [kokkos.github.io/kokkos-core-wiki](https://kokkos.github.io/kokkos-core-wiki)

# Learning about Kokkos

To start learning about Kokkos:

- [Kokkos Lectures](https://kokkos.github.io/kokkos-core-wiki/videolectures.html): they contain a mix of lecture videos and hands-on exercises covering all the important Kokkos Ecosystem capabilities.

- [Programming guide](https://kokkos.github.io/kokkos-core-wiki/programmingguide.html): contains in "narrative" form a technical description of the programming model, machine model, and the main building blocks like the Views and parallel dispatch.

- [API reference](https://kokkos.github.io/kokkos-core-wiki/): organized by category, i.e., [core](https://kokkos.github.io/kokkos-core-wiki/API/core-index.html), [algorithms](https://kokkos.github.io/kokkos-core-wiki/API/algorithms-index.html) and [containers](https://kokkos.github.io/kokkos-core-wiki/API/containers-index.html) or, if you prefer, in [alphabetical order](https://kokkos.github.io/kokkos-core-wiki/API/alphabetical.html).

- [Use cases and Examples](https://kokkos.github.io/kokkos-core-wiki/usecases.html): a series of examples ranging from how to use Kokkos with MPI to Fortran interoperability.

For questions find us on Slack: https://kokkosteam.slack.com or open a github issue.

For non-public questions send an email to: *crtrott(at)sandia.gov*

# Contributing to Kokkos

Please see [this page](https://kokkos.github.io/kokkos-core-wiki/contributing.html) for details on how to contribute.

# Requirements, Building and Installing

All requirements including minimum and primary tested compiler versions can be found [here](https://kokkos.github.io/kokkos-core-wiki/requirements.html).

Building and installation instructions are described [here](https://kokkos.github.io/kokkos-core-wiki/building.html).

# Citing Kokkos

Please see the [following page](https://kokkos.github.io/kokkos-core-wiki/citation.html).

# License

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Under the terms of Contract DE-NA0003525 with NTESS,
the U.S. Government retains certain rights in this software.

The full license statement used in all headers is available [here](https://kokkos.github.io/kokkos-core-wiki/license.html) or
[here](https://github.com/kokkos/kokkos/blob/master/LICENSE).

# Test Status

[![jenkins-nightly](https://cloud.cees.ornl.gov/jenkins-ci/job/Kokkos-nightly/badge/icon?style=plastic&subject=jenkins-nightly)](https://cloud.cees.ornl.gov/jenkins-ci/job/Kokkos-nightly/)
[![workflows/linux-64bit](https://github.com/kokkos/kokkos/actions/workflows/continuous-integration-workflow.yml/badge.svg?branch=develop)](https://github.com/kokkos/kokkos/actions/workflows/continuous-integration-workflow.yml)
[![workflows/linux-32bit](https://github.com/kokkos/kokkos/actions/workflows/continuous-integration-workflow-32bit.yml/badge.svg?branch=develop)](https://github.com/kokkos/kokkos/actions/workflows/continuous-integration-workflow-32bit.yml)
[![workflows/hpx](https://github.com/kokkos/kokkos/actions/workflows/continuous-integration-workflow-hpx.yml/badge.svg?branch=develop)](https://github.com/kokkos/kokkos/actions/workflows/continuous-integration-workflow-hpx.yml)
[![workflows/osx](https://github.com/kokkos/kokkos/actions/workflows/osx.yml/badge.svg?branch=develop)](https://github.com/kokkos/kokkos/actions/workflows/osx.yml)
[![workflows/performance-benchmark](https://github.com/kokkos/kokkos/actions/workflows/performance-benchmark.yml/badge.svg?branch=develop)](https://github.com/kokkos/kokkos/actions/workflows/performance-benchmark.yml)
[![appveyor/windows](https://ci.appveyor.com/api/projects/status/github/kokkos/kokkos?branch=develop?svg=true)](https://ci.appveyor.com/project/dalg24/kokkos)
