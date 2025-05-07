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

void ViewFirstTouch(benchmark::State& state) {
  const int N    = state.range(0);
  using ViewType = Kokkos::View<double*>;

  for (auto _ : state) {
    Kokkos::fence();

    Kokkos::Timer timer;
    ViewType v_a("A", N);
    KokkosBenchmark::report_results(state, v_a, 1, timer.seconds());
  }
}

void ViewFirstTouch_deepcopy(benchmark::State& state) {
  const int N    = state.range(0);
  using ViewType = Kokkos::View<double*>;

  for (auto _ : state) {
    Kokkos::fence();

    Kokkos::Timer timer;
    ViewType v_a(Kokkos::view_alloc(Kokkos::WithoutInitializing, "A"), N);
    Kokkos::deep_copy(v_a, 1.1);
    KokkosBenchmark::report_results(state, v_a, 1, timer.seconds());
  }
}
