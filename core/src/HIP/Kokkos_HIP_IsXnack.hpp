//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

/*--------------------------------------------------------------------------*/

#ifndef KOKKOS_HIP_ISXNACK_HPP
#define KOKKOS_HIP_ISXNACK_HPP
#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#endif  // KOKKOS_IMPL_PUBLIC_INCLUDE

namespace Kokkos::Impl {

/* Returns true iff we think the AMD GPU can access allocations created with the
system allocator.

Based on AMD's ROCm 6.3.1 documentation:
https://github.com/ROCm/HIP/blob/2c240cacff16c2bb18ce9e5b4c1b937ab17a0199/docs/how-to/hip_runtime_api/memory_management/unified_memory.rst?plain=1#L141-L146

    To ensure the proper functioning of system allocated unified memory on
    supported GPUs, it is essential to configure the environment variable
    ``XNACK=1`` (sic) and use a kernel that supports HMM. Without this
    configuration, the behavior will be similar to that of systems without HMM
    support.

This clearly states two things are required:
* HSA_XNACK=1 is set in the environment
* The kernel must support HMM

Across a couple Nvidia and AMD systems, we have observed that
CONFIG_HMM_MIRROR=y was set in /boot/config-$(uname -r). This test may need to
be modified if a better way is determined to check for HMM support in Linux.
Checking for CONFIG_HMM was considered, but it was not present on El Capitan,
so we infer its presence is not necessary.
*/
bool xnack_enabled();
}  // namespace Kokkos::Impl

#endif  // KOKKOS_HIP_ISXNACK_HPP
