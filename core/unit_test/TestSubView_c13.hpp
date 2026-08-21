// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_TEST_SUBVIEW_C13_HPP
#define KOKKOS_TEST_SUBVIEW_C13_HPP
#include <TestViewSubview.hpp>

namespace Test {

TEST(TEST_CATEGORY, view_test_unmanaged_subview_reset) {
#if defined(KOKKOS_ENABLE_OPENACC) && (KOKKOS_COMPILER_NVHPC > 240500) && \
    (KOKKOS_COMPILER_NVHPC <= 260500)
  // FIXME_OPENACC: compiling below is known to fail if 24.5 < NVHPC version
  // <= 26.5.
  // Error behavior: a device kernel accesses a undefined global
  // symbol.
  // Error message: parse use of undefined value '@_T1166_96704'
  GTEST_SKIP() << "skipping since the OpenACC backend fails to comple this "
                  "test if 24.5 < NVHPC version <= 26.5";
#else
  TestViewSubview::test_unmanaged_subview_reset<TEST_EXECSPACE>();
#endif
}

}  // namespace Test
#endif
