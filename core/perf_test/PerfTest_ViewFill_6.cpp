// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

// Copyright Contributors to the Kokkos project
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PerfTest_ViewFill.hpp"

namespace Test {

BENCHMARK(ViewFill_Rank6<Kokkos::LayoutLeft>)
    ->ArgName("N")
    ->Arg(N)
    ->UseManualTime();

BENCHMARK(ViewFill_Rank6<Kokkos::LayoutRight>)
    ->ArgName("N")
    ->Arg(N)
    ->UseManualTime();

}  // namespace Test
