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

#include "PerfTest_ViewFirstTouch.hpp"

#define BENCHMARK_SET_ARG_PARALLEL_FOR(FUNCTION, DATA_TYPE) \
  BENCHMARK_TEMPLATE(FUNCTION, DATA_TYPE)                   \
      ->ArgName("N")                                        \
      ->RangeMultiplier(4)                                  \
      ->Range(int64_t(1) << 4, int64_t(1) << 30)            \
      ->UseManualTime();

#define BENCHMARK_SET_ARG_DEEP_COPY(FUNCTION, DATA_TYPE)      \
  BENCHMARK_TEMPLATE(FUNCTION, DATA_TYPE)                     \
      ->ArgNames({"N", "init_value"})                         \
      ->RangeMultiplier(4)                                    \
      ->Ranges({{int64_t(1) << 4, int64_t(1) << 30}, {0, 1}}) \
      ->UseManualTime();

namespace Test {

BENCHMARK_SET_ARG_PARALLEL_FOR(ViewFirstTouch_Initialize, double)
BENCHMARK_SET_ARG_PARALLEL_FOR(ViewFirstTouch_Initialize, int)
BENCHMARK_SET_ARG_PARALLEL_FOR(ViewFirstTouch_Initialize, float)

BENCHMARK_SET_ARG_PARALLEL_FOR(ViewFirstTouch_ParallelFor, double)
BENCHMARK_SET_ARG_PARALLEL_FOR(ViewFirstTouch_ParallelFor, int)
BENCHMARK_SET_ARG_PARALLEL_FOR(ViewFirstTouch_ParallelFor, float)

BENCHMARK_SET_ARG_DEEP_COPY(ViewFirstTouch_DeepCopy, double)
BENCHMARK_SET_ARG_DEEP_COPY(ViewFirstTouch_DeepCopy, int)
BENCHMARK_SET_ARG_DEEP_COPY(ViewFirstTouch_DeepCopy, float)

}  // namespace Test

#undef BENCHMARK_SET_ARG_PARALLEL_FOR
#undef BENCHMARK_SET_ARG_DEEP_COPY
