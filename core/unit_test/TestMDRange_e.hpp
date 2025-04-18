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

#include <TestMDRange.hpp>

namespace Test {

TEST(TEST_CATEGORY, mdrange_4d) {
// FIXME_OPENMPTARGET requires MDRange parallel_reduce
#ifndef KOKKOS_ENABLE_OPENMPTARGET
  TestMDRange_4D<TEST_EXECSPACE>::test_reduce4(100, 10, 10, 10);
#endif
  TestMDRange_4D<TEST_EXECSPACE>::test_for4(100, 10, 10, 10);
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || \
    defined(KOKKOS_ENABLE_SYCL)
  const int size_x = 2 << 19;  // 2^20
  TestMDRange_4D<TEST_EXECSPACE>::test_for4_eval_once(size_x, 1, 1, 1);
  TestMDRange_4D<TEST_EXECSPACE>::test_for4_eval_once(1, size_x, 1, 1);
  TestMDRange_4D<TEST_EXECSPACE>::test_for4_eval_once(1, 1, size_x, 1);
  TestMDRange_4D<TEST_EXECSPACE>::test_for4_eval_once(1, 1, 1, size_x);
#endif
}

}  // namespace Test
