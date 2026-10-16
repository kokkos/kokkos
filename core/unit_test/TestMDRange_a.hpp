// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <TestMDRange.hpp>

namespace Test {

TEST(TEST_CATEGORY, mdrange_5d) {
// FIXME_NEXTSILICON hangs or very slow on 1.4.0-136
#ifdef KOKKOS_ENABLE_NEXTSILICON
  if (std::is_same_v<TEST_EXECSPACE, Kokkos::Experimental::NextSilicon>) {
    GTEST_SKIP() << "low performance on 1.4.0-136";
  }
#endif

  TestMDRange_5D<TEST_EXECSPACE>::test_reduce5(100, 10, 10, 10, 5);
  TestMDRange_5D<TEST_EXECSPACE>::test_for5(100, 10, 10, 10, 5);
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || \
    defined(KOKKOS_ENABLE_SYCL)
  const int size_x = 2 << 19;  // 2^20
  TestMDRange_5D<TEST_EXECSPACE>::test_for5_eval_once(size_x, 1, 1, 1, 1);
  TestMDRange_5D<TEST_EXECSPACE>::test_for5_eval_once(1, size_x, 1, 1, 1);
  TestMDRange_5D<TEST_EXECSPACE>::test_for5_eval_once(1, 1, size_x, 1, 1);
  TestMDRange_5D<TEST_EXECSPACE>::test_for5_eval_once(1, 1, 1, size_x, 1);
  TestMDRange_5D<TEST_EXECSPACE>::test_for5_eval_once(1, 1, 1, 1, size_x);
#endif
}

}  // namespace Test
