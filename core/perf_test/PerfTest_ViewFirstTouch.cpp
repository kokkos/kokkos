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

BENCHMARK(ViewFirstTouch)
    ->ArgName("N")
    ->RangeMultiplier(4)
    ->Range(int64_t(1) << 1, int64_t(1) << 30)
    ->UseManualTime();

BENCHMARK(ViewFirstTouch_deepcopy)
    ->ArgName("N")
    ->RangeMultiplier(4)
    ->Range(int64_t(1) << 1, int64_t(1) << 30)
    ->UseManualTime();

}  // namespace Test
