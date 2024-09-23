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

#include <TestAtomicOperations.hpp>

namespace Test {
TEST(TEST_CATEGORY, atomic_operations_unsigned) {
  // FIXME_OPENMPTARGET - causes ICE with CrayClang compiler
#if defined(KOKKOS_COMPILER_CRAY_LLVM) && defined(KOKKOS_ENABLE_OPENMPTARGET)
  GTEST_SKIP() << "known to fail with OpenMPTarget+Cray LLVM";
#endif
  const int start = 0;
  const int end   = 11;
  for (int i = start; i < end; ++i) {
    for (int t = 0; t < 16; t++)
      ASSERT_TRUE((TestAtomicOperations::AtomicOperationsTestIntegralType<
                   unsigned int, TEST_EXECSPACE>(i, end - i + start, t)));
    ASSERT_TRUE(
        (TestAtomicOperations::AtomicOperationsTestUnsignedIntegralType<
            unsigned int, TEST_EXECSPACE>(i, end - i, 1)));  // Wrapping Inc
    ASSERT_TRUE(
        (TestAtomicOperations::AtomicOperationsTestUnsignedIntegralType<
            unsigned int, TEST_EXECSPACE>(i, end - i, 2)));  // Wrapping Dec
  }
}
}  // namespace Test
