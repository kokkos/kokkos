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

#include "Benchmark_Context.hpp"

namespace Benchmark {

template <typename DataType>
void ViewFirstTouch_Initialize(benchmark::State& state) {
  const int N    = state.range(0);
  using ViewType = Kokkos::View<DataType*>;

  for (auto _ : state) {
    Kokkos::Timer timer;
    ViewType v_a("A", N);
    Kokkos::fence();
    KokkosBenchmark::report_results(state, v_a, 1, timer.seconds());
  }
}

BENCHMARK_TEMPLATE(ViewFirstTouch_Initialize, double)
    ->ArgName("N")
    ->RangeMultiplier(8)
    ->Range(int64_t(1) << 6, int64_t(1) << 24)
    ->UseManualTime();

}  // namespace Benchmark
