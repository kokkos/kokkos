// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

// Copyright Contributors to the Kokkos project
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PerfTest_ViewCopy.hpp"

namespace Test {

BENCHMARK(ViewDeepCopy_Rank4<Kokkos::LayoutLeft, Kokkos::LayoutLeft>)
    ->ArgName("N")
    ->Arg(10)
    ->UseManualTime();

BENCHMARK(ViewDeepCopy_Rank5<Kokkos::LayoutLeft, Kokkos::LayoutLeft>)
    ->ArgName("N")
    ->Arg(10)
    ->UseManualTime();

}  // namespace Test
