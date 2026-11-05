// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

// Copyright Contributors to the Kokkos project
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PerfTest_ViewResize.hpp"

namespace Test {

BENCHMARK(ViewResize_Rank4<Kokkos::LayoutLeft>)
    ->ArgName("N")
    ->Arg(N)
    ->UseManualTime()
    ->Iterations(R);

BENCHMARK(ViewResize_Rank4<Kokkos::LayoutRight>)
    ->ArgName("N")
    ->Arg(N)
    ->UseManualTime()
    ->Iterations(R);

BENCHMARK(ViewResize_Rank5<Kokkos::LayoutLeft>)
    ->ArgName("N")
    ->Arg(N)
    ->UseManualTime()
    ->Iterations(R);

BENCHMARK(ViewResize_Rank5<Kokkos::LayoutRight>)
    ->ArgName("N")
    ->Arg(N)
    ->UseManualTime()
    ->Iterations(R);

BENCHMARK(ViewResize_NoInit_Rank4<Kokkos::LayoutLeft>)
    ->ArgName("N")
    ->Arg(N)
    ->UseManualTime()
    ->Iterations(R);

BENCHMARK(ViewResize_NoInit_Rank4<Kokkos::LayoutRight>)
    ->ArgName("N")
    ->Arg(N)
    ->UseManualTime()
    ->Iterations(R);

BENCHMARK(ViewResize_NoInit_Rank5<Kokkos::LayoutLeft>)
    ->ArgName("N")
    ->Arg(N)
    ->UseManualTime()
    ->Iterations(R);

BENCHMARK(ViewResize_NoInit_Rank5<Kokkos::LayoutRight>)
    ->ArgName("N")
    ->Arg(N)
    ->UseManualTime()
    ->Iterations(R);

}  // namespace Test
