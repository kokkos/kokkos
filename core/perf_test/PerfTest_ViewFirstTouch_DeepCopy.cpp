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

template <typename DataType>
void ViewFirstTouch_DeepCopy(benchmark::State& state) {
  const int N               = state.range(0);
  const DataType init_value = static_cast<DataType>(state.range(1));
  using ViewType            = Kokkos::View<DataType*>;

  for (auto _ : state) {
    Kokkos::fence();

    ViewType v_a("A", N);
    Kokkos::fence();
    Kokkos::Timer timer;
    Kokkos::deep_copy(v_a, init_value);
    KokkosBenchmark::report_results(state, v_a, 2, timer.seconds());
  }
}

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
