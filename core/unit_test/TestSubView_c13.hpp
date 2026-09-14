// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_TEST_SUBVIEW_C13_HPP
#define KOKKOS_TEST_SUBVIEW_C13_HPP
#include <TestViewSubview.hpp>

namespace Test {

TEST(TEST_CATEGORY, view_test_unmanaged_subview_reset) {
  TestViewSubview::test_unmanaged_subview_reset<TEST_EXECSPACE>();
}

#if !defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_CUDA_CONSTEXPR)
TEST(TEST_CATEGORY, view_subview_std_pair_in_kernel) {
  TEST_EXECSPACE::execution_space exec;
  (void)TestViewSubview::TestSubviewStdPairInKernel(exec);
}
#endif

}  // namespace Test
#endif
