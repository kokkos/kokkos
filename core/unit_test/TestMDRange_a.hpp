// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <TestMDRange.hpp>

namespace Test {

TEST(TEST_CATEGORY, mdrange_5d) {
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

TEST(TEST_CATEGORY, mdrange_5d_static_batch_size) {
  {
    Test5DStaticBatchSize<TEST_EXECSPACE,
                          Kokkos::Experimental::StaticBatchSize<1>>
        f(16, 16, 16, 16, 16);
    f.test_batch_size();
  }
  {
    Test5DStaticBatchSize<TEST_EXECSPACE,
                          Kokkos::Experimental::StaticBatchSize<4>>
        f(8, 16, 32, 16, 8);
    f.test_batch_size();
  }

  // md range dim not divisible by static batch size
  {
    Test5DStaticBatchSize<TEST_EXECSPACE,
                          Kokkos::Experimental::StaticBatchSize<8>>
        f(17, 15, 31, 7, 19);
    f.test_batch_size();
  }

  // md range dim smaller than static batch size
  {
    Test5DStaticBatchSize<TEST_EXECSPACE,
                          Kokkos::Experimental::StaticBatchSize<16>>
        f(3, 2, 1, 2, 3);
    f.test_batch_size();
  }
}

}  // namespace Test
