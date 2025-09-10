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

#include "PerfTestMDRange.hpp"
#include <cstdint>
#include <limits>

namespace Benchmark {

template <int Rank = 2, typename index_type = Kokkos::IndexType<int32_t>>
static void MDRangePolicy_Triad(benchmark::State& state) {
  double N                = static_cast<double>(state.range(0));
  uint32_t total_elements = std::pow(N, 6.0);

  for (auto _ : state) {
    double seconds = MDRangePolicyTriad<TEST_EXECSPACE, Rank, double,
                                        index_type>::test_triad(N, 32);
    state.SetIterationTime(seconds);
  }
  state.counters["Problem Size"] = benchmark::Counter(total_elements);
  state.SetItemsProcessed(state.iterations() * total_elements);
}

#define MDRangePolicy_MAKE_BENCHMARK(RANK, INDEX_TYPE)      \
  BENCHMARK_TEMPLATE(MDRangePolicy_Triad, RANK, INDEX_TYPE) \
      ->RangeMultiplier(2)                                  \
      ->Range(1 << 2, 1 << 4)                               \
      ->Iterations(16)                                      \
      ->UseManualTime()                                     \
      ->Unit(benchmark::kMillisecond);

MDRangePolicy_MAKE_BENCHMARK(2, Kokkos::IndexType<int32_t>)
    MDRangePolicy_MAKE_BENCHMARK(2, Kokkos::IndexType<int64_t>)
        MDRangePolicy_MAKE_BENCHMARK(3, Kokkos::IndexType<int32_t>)
            MDRangePolicy_MAKE_BENCHMARK(3, Kokkos::IndexType<int64_t>)
                MDRangePolicy_MAKE_BENCHMARK(4, Kokkos::IndexType<int32_t>)
                    MDRangePolicy_MAKE_BENCHMARK(4, Kokkos::IndexType<int64_t>)
                        MDRangePolicy_MAKE_BENCHMARK(5,
                                                     Kokkos::IndexType<int32_t>)
                            MDRangePolicy_MAKE_BENCHMARK(
                                5, Kokkos::IndexType<int64_t>)
                                MDRangePolicy_MAKE_BENCHMARK(
                                    6, Kokkos::IndexType<int32_t>)
                                    MDRangePolicy_MAKE_BENCHMARK(
                                        6, Kokkos::IndexType<int64_t>)

#undef MDRangePolicy_MAKE_BENCHMARK

}  // namespace Benchmark
