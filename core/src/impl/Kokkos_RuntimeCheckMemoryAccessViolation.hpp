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

#ifndef KOKKOS_RUNTIME_CHECK_MEMORY_ACCESS_VIOLATION_HPP
#define KOKKOS_RUNTIME_CHECK_MEMORY_ACCESS_VIOLATION_HPP
#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#endif

#include <Kokkos_Macros.hpp>

namespace Kokkos {

template <class AccessSpace, class MemorySpace>
struct SpaceAccessibility;

namespace Impl {

// primary template: memory space is accessible, do nothing.
template <class MemorySpace, class AccessSpace,
          bool = SpaceAccessibility<AccessSpace, MemorySpace>::accessible>
struct RuntimeCheckMemoryAccessViolation {
  KOKKOS_FUNCTION RuntimeCheckMemoryAccessViolation(char const *const) {}
};

// explicit specialization: memory access violation will occur, call abort with
// the specified error message.
template <class MemorySpace, class AccessSpace>
struct RuntimeCheckMemoryAccessViolation<MemorySpace, AccessSpace, false> {
  KOKKOS_FUNCTION RuntimeCheckMemoryAccessViolation(char const *const msg) {
    Kokkos::abort(msg);
  }
};

// calls abort with default error message at runtime if memory access violation
// will occur
template <class MemorySpace>
KOKKOS_FUNCTION void runtime_check_memory_access_violation() {
  KOKKOS_IF_ON_HOST((
      RuntimeCheckMemoryAccessViolation<MemorySpace, DefaultHostExecutionSpace>(
          "ERROR: attempt to access inaccessible memory space");))
  KOKKOS_IF_ON_DEVICE(
      (RuntimeCheckMemoryAccessViolation<MemorySpace, DefaultExecutionSpace>(
           "ERROR: attempt to access inaccessible memory space");))
}

// calls abort with specified error message at runtime if memory access
// violation will occur
template <class MemorySpace>
KOKKOS_FUNCTION void runtime_check_memory_access_violation(
    char const *const msg) {
  KOKKOS_IF_ON_HOST((
      (void)RuntimeCheckMemoryAccessViolation<MemorySpace,
                                              DefaultHostExecutionSpace>(msg);))
  KOKKOS_IF_ON_DEVICE((
      (void)
          RuntimeCheckMemoryAccessViolation<MemorySpace, DefaultExecutionSpace>(
              msg);))
}

}  // namespace Impl
}  //  namespace Kokkos

#endif  //  KOKKOS_RUNTIME_CHECK_MEMORY_ACCESS_VIOLATION_HPP
