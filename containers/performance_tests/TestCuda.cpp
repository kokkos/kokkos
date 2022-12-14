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

#include <Kokkos_Macros.hpp>

#include <cstdint>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <TestDynRankView.hpp>

#include <Kokkos_UnorderedMap.hpp>

#include <TestGlobal2LocalIds.hpp>

#include <TestUnorderedMapPerformance.hpp>

namespace Performance {

TEST(TEST_CATEGORY, dynrankview_perf) {
  std::cout << "Cuda" << std::endl;
  std::cout << " DynRankView vs View: Initialization Only " << std::endl;
  test_dynrankview_op_perf<Kokkos::Cuda>(40960);
}

TEST(TEST_CATEGORY, global_2_local) {
#if defined(KOKKOS_ENABLE_CUDA) && \
    defined(KOKKOS_COMPILER_NVHPC)  // FIXME_NVHPC
  GTEST_SKIP() << "errors reported for all sizes";
#endif
  std::cout << "Cuda" << std::endl;
  std::cout << "size, create, generate, fill, find" << std::endl;
  for (unsigned i = Performance::begin_id_size; i <= Performance::end_id_size;
       i *= Performance::id_step)
    test_global_to_local_ids<Kokkos::Cuda>(i);
}

TEST(TEST_CATEGORY, unordered_map_performance_near) {
  Perf::run_performance_tests<Kokkos::Cuda, true>("cuda-near");
}

TEST(TEST_CATEGORY, unordered_map_performance_far) {
  Perf::run_performance_tests<Kokkos::Cuda, false>("cuda-far");
}

}  // namespace Performance
