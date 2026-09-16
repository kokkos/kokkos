// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <cstddef>

#include <benchmark/benchmark.h>

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
import kokkos.random;
import kokkos.std_algorithms;
#else
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_Sort.hpp>
#endif
#include <Kokkos_Timer.hpp>
// FIXME: Benchmark_Context.hpp should be moved to a common location
#include "../../core/perf_test/Benchmark_Context.hpp"

namespace {

template <class T>
void BM_sort_by_key(benchmark::State& state) {
  const std::size_t n = state.range(0);

  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace    = typename ExecutionSpace::memory_space;

  ExecutionSpace space;

  Kokkos::Random_XorShift1024_Pool<MemorySpace> rand_pool(5374857);

  Kokkos::View<T*, MemorySpace> keys_orig(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing, "keys_orig"), n);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n), KOKKOS_LAMBDA(int i) {
        auto rand_gen = rand_pool.get_state();
        keys_orig(i)  = rand_gen.urand();
        rand_pool.free_state(rand_gen);
      });

  space.fence();

  Kokkos::View<unsigned*, MemorySpace> values_orig(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing, "values_orig"), n);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n),
      KOKKOS_LAMBDA(int i) { values_orig(i) = i; });

  Kokkos::View<T*, MemorySpace> keys(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing, "keys"), n);

  Kokkos::View<unsigned*, MemorySpace> values(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing, "values"), n);

  auto data_ratio =
      1 + sizeof(typename decltype(values)::value_type) / (double)sizeof(T);

  for (auto _ : state) {
    Kokkos::deep_copy(space, keys, keys_orig);
    Kokkos::deep_copy(space, values, values_orig);

    space.fence();
    Kokkos::Timer timer;
    Kokkos::Experimental::sort_by_key(space, keys, values);
    space.fence();
    double time = timer.seconds();

    KokkosBenchmark::report_results(state, keys_orig, data_ratio, time);
    state.counters["Passed"] = true;
  }
}

constexpr std::size_t PROB_SIZE = 100'000;

}  // anonymous namespace

// FIXME: Add logic to pass min. warm-up time. Also, the value should be set
// by the user. Say, via the environment variable BENCHMARK_MIN_WARMUP_TIME.

BENCHMARK(BM_sort_by_key<unsigned>)->Arg(PROB_SIZE)->UseManualTime();
BENCHMARK(BM_sort_by_key<int>)->Arg(PROB_SIZE)->UseManualTime();
BENCHMARK(BM_sort_by_key<long long>)->Arg(PROB_SIZE)->UseManualTime();
BENCHMARK(BM_sort_by_key<float>)->Arg(PROB_SIZE)->UseManualTime();
BENCHMARK(BM_sort_by_key<double>)->Arg(PROB_SIZE)->UseManualTime();
