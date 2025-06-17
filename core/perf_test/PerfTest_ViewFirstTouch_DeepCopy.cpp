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

namespace Test {

BENCHMARK_TEMPLATE(ViewFirstTouch_DeepCopy, double)
    ->ArgNames({"N", "init_value"})
    ->RangeMultiplier(8)
    ->Ranges({{int64_t(1) << 3, int64_t(1) << 27}, {0, 1}})
    ->UseManualTime();

BENCHMARK_TEMPLATE(ViewFirstTouch_DeepCopy, float)
    ->ArgNames({"N", "init_value"})
    ->RangeMultiplier(8)
    ->Ranges({{int64_t(1) << 3, int64_t(1) << 27}, {0, 1}})
    ->UseManualTime();

BENCHMARK_TEMPLATE(ViewFirstTouch_DeepCopy, int)
    ->ArgNames({"N", "init_value"})
    ->RangeMultiplier(8)
    ->Ranges({{int64_t(1) << 3, int64_t(1) << 27}, {0, 1}})
    ->UseManualTime();

}  // namespace Test
