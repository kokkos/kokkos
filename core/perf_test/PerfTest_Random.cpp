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

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <benchmark/benchmark.h>
#include "Benchmark_Context.hpp"

namespace Benchmark {

// Fills each entry of a view of size N with the sum of K random numbers
template <typename Pool>
static void Random(benchmark::State &state) {
  const size_t N = state.range(0);
  const size_t K = state.range(1);

  Kokkos::View<double *> out("out", N);
  Pool random_pool(/*seed=*/12345);

  for (auto _ : state) {
    Kokkos::parallel_for(
        N, KOKKOS_LAMBDA(const int i) {
          auto generator = random_pool.get_state();
          double acc     = 0;
          for (size_t k = 0; k < K; ++k) {
            acc += generator.drand(0., 1.);
          }
          random_pool.free_state(generator);
          out(i) = acc;
        });
    Kokkos::fence("");
  }

  state.counters[KokkosBenchmark::benchmark_fom("rate")] = benchmark::Counter(
      state.iterations() * N * K, benchmark::Counter::kIsRate);
}

static void Random64(benchmark::State &state) {
  return Random<Kokkos::Random_XorShift64_Pool<>>(state);
}

static void Random1024(benchmark::State &state) {
  return Random<Kokkos::Random_XorShift1024_Pool<>>(state);
}

BENCHMARK(Random64)
    ->ArgNames({"N", "K"})
    ->ArgsProduct({{1 << 15, 1 << 18, 1 << 21}, {1, 8, 64, 512}})
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(Random1024)
    ->ArgNames({"N", "K"})
    ->ArgsProduct({{1 << 15, 1 << 18, 1 << 21}, {1, 8, 64, 512}})
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

}  // namespace Benchmark
