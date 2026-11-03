// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <TestAtomicOperations.hpp>

namespace Test {
TEST(TEST_CATEGORY, atomic_operations_float) {
  const int start = -5;
  const int end   = 11;
  for (int i = start; i < end; ++i) {
    for (int t = 0; t < 7; t++) {
      if (t == 5 && TestAtomicOperations::division_may_differ_by_one_ulp<
                        TEST_EXECSPACE>) {
        ASSERT_TRUE((TestAtomicOperations::AtomicOperationsTestNonIntegralType<
                     float, TEST_EXECSPACE, true>(i, end - i + start, t)));
      } else {
        ASSERT_TRUE((TestAtomicOperations::AtomicOperationsTestNonIntegralType<
                     float, TEST_EXECSPACE>(i, end - i + start, t)));
      }
    }
  }
}
}  // namespace Test
