// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <TestCudaRDCLockArrays.hpp>

namespace Test {
namespace RDCLockArrays {

KOKKOS_FUNCTION void atomic_add_elsewhere(value_type* dest, value_type val) {
  Kokkos::atomic_add(dest, val);
}

}  // namespace RDCLockArrays
}  // namespace Test
