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

#include "PerfTestMDRange_Stream.hpp"

namespace Benchmark {

namespace {
// Number of times the test kernel is run
const int kKernelIterations = 32;

void report_mdrange_result(benchmark::State& state, double N, int data_ratio) {
  double num_elements            = std::pow(N, 6);
  state.counters["Problem Size"] = benchmark::Counter(num_elements);
  state.SetBytesProcessed(static_cast<long long>(
      state.iterations() * num_elements * sizeof(double) * data_ratio));
}
}  // namespace

template <int Rank>
static void MDRangePolicy_Set(benchmark::State& state) {
  double N = static_cast<double>(state.range(0));
  using MDRangePolicy_StreamTest =
      MDRangePolicy_StreamTest<TEST_EXECSPACE, Rank, double>;

  for (auto _ : state) {
    double seconds = MDRangePolicy_StreamTest::test_set(N, kKernelIterations);
    state.SetIterationTime(seconds);
  }
  report_mdrange_result(state, N, 1);
}

template <int Rank>
static void MDRangePolicy_Scale(benchmark::State& state) {
  double N = static_cast<double>(state.range(0));
  using MDRangePolicy_StreamTest =
      MDRangePolicy_StreamTest<TEST_EXECSPACE, Rank, double>;

  for (auto _ : state) {
    double seconds = MDRangePolicy_StreamTest::test_scale(N, kKernelIterations);
    state.SetIterationTime(seconds);
  }
  report_mdrange_result(state, N, 2);
}

template <int Rank>
static void MDRangePolicy_Add(benchmark::State& state) {
  double N = static_cast<double>(state.range(0));
  using MDRangePolicy_StreamTest =
      MDRangePolicy_StreamTest<TEST_EXECSPACE, Rank, double>;

  for (auto _ : state) {
    double seconds = MDRangePolicy_StreamTest::test_add(N, kKernelIterations);
    state.SetIterationTime(seconds);
  }
  report_mdrange_result(state, N, 3);
}

template <int Rank>
static void MDRangePolicy_Triad(benchmark::State& state) {
  double N = static_cast<double>(state.range(0));
  using MDRangePolicy_StreamTest =
      MDRangePolicy_StreamTest<TEST_EXECSPACE, Rank, double>;

  for (auto _ : state) {
    double seconds = MDRangePolicy_StreamTest::test_triad(N, kKernelIterations);
    state.SetIterationTime(seconds);
  }
  report_mdrange_result(state, N, 3);
}

// Macros to generate benchmarks
#define MDRANGE_BENCHMARK_ARGS(BENCH_FUNCTION, RANKS) \
  BENCHMARK_TEMPLATE(BENCH_FUNCTION, RANKS)           \
      ->Arg(24)                                       \
      ->Iterations(10)                                \
      ->UseManualTime()                               \
      ->Unit(benchmark::kMillisecond);

#define MDRANGE_MAKE_BENCHMARK(BENCH_FUNCTION) \
  MDRANGE_BENCHMARK_ARGS(BENCH_FUNCTION, 2)    \
  MDRANGE_BENCHMARK_ARGS(BENCH_FUNCTION, 3)    \
  MDRANGE_BENCHMARK_ARGS(BENCH_FUNCTION, 4)    \
  MDRANGE_BENCHMARK_ARGS(BENCH_FUNCTION, 5)    \
  MDRANGE_BENCHMARK_ARGS(BENCH_FUNCTION, 6)

// Only run the benchmarks if not running on the host
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || \
    defined(KOKKOS_ENABLE_SYCL)
MDRANGE_MAKE_BENCHMARK(MDRangePolicy_Set)
MDRANGE_MAKE_BENCHMARK(MDRangePolicy_Add)
MDRANGE_MAKE_BENCHMARK(MDRangePolicy_Scale)
MDRANGE_MAKE_BENCHMARK(MDRangePolicy_Triad)
#endif

#undef MDRANGE_BENCHMARK_ARGS
#undef MDRANGE_MAKE_BENCHMARK

}  // namespace Benchmark
