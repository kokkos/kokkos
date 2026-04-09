// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include "Benchmark_Context.hpp"

#include <benchmark/benchmark.h>

#include <cstdint>

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif

#include "PerfTest_Category.hpp"

namespace Benchmark {
namespace {

using execution_space = TEST_EXECSPACE;
using index_type      = std::int64_t;
using view_type       = Kokkos::View<float*, execution_space>;

constexpr float triad_scalar = 1.61803398875f;

struct TriadFixture {
  view_type a;
  view_type b;
  view_type c;

  explicit TriadFixture(index_type n) : a("a", n), b("b", n), c("c", n) {
    Kokkos::deep_copy(a, 1.0f);
    Kokkos::deep_copy(b, 2.0f);
    Kokkos::deep_copy(c, 0.0f);
    execution_space().fence();
  }

  template <class Policy>
  void run(benchmark::State& state, const Policy& policy, int tile_size) const {
    const auto local_a = a;
    const auto local_b = b;
    const auto local_c = c;
    for (auto _ : state) {
      Kokkos::Timer timer;
      Kokkos::parallel_for(
          "triad1d", policy,
          KOKKOS_LAMBDA(const index_type i) {
            local_c(i) = local_a(i) + triad_scalar * local_b(i);
          });
      execution_space().fence();
      KokkosBenchmark::report_results(state, c, 3, timer.seconds());
    }
    state.counters["Tile"] = benchmark::Counter(tile_size);
  }
};

using range_policy =
    Kokkos::RangePolicy<execution_space, Kokkos::IndexType<index_type>>;
using md_range_policy = Kokkos::MDRangePolicy<execution_space, Kokkos::Rank<1>,
                                              Kokkos::IndexType<index_type>>;

void RangePolicyTriad(benchmark::State& state) {
  const index_type n = state.range(0);
  TriadFixture fixture(n);
  fixture.run(state, range_policy(0, n), 0);
}

void MDRangePolicyTriad(benchmark::State& state, int tile_size) {
  const index_type n = state.range(0);
  TriadFixture fixture(n);

  const md_range_policy::point_type lower{0};
  const md_range_policy::point_type upper{n};
  if (tile_size > 0) {
    const md_range_policy::point_type tile{
        static_cast<md_range_policy::index_type>(tile_size)};
    fixture.run(state, md_range_policy(lower, upper, tile), tile_size);
  } else {
    fixture.run(state, md_range_policy(lower, upper), tile_size);
  }
}

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || \
    defined(KOKKOS_ENABLE_SYCL)
#define RANGE_VS_MDRANGE_TRIAD1D_SMALL_ARG_SIZE (1 << 23)
#define RANGE_VS_MDRANGE_TRIAD1D_SMALL_ARG_SIZE_UNALIGNED ((1 << 23) + 17)
#define RANGE_VS_MDRANGE_TRIAD1D_LARGE_ARG_SIZE (1 << 26)
#else
#define RANGE_VS_MDRANGE_TRIAD1D_SMALL_ARG_SIZE (1 << 18)
#define RANGE_VS_MDRANGE_TRIAD1D_SMALL_ARG_SIZE_UNALIGNED ((1 << 18) + 17)
#define RANGE_VS_MDRANGE_TRIAD1D_LARGE_ARG_SIZE (1 << 24)
#endif

#define RANGE_VS_MDRANGE_TRIAD1D_BENCHMARK_ARGS(BENCHMARK_HANDLE) \
  BENCHMARK_HANDLE->Arg(RANGE_VS_MDRANGE_TRIAD1D_SMALL_ARG_SIZE)  \
      ->Arg(RANGE_VS_MDRANGE_TRIAD1D_SMALL_ARG_SIZE_UNALIGNED)    \
      ->Arg(RANGE_VS_MDRANGE_TRIAD1D_LARGE_ARG_SIZE)              \
      ->UseManualTime()                                            \
      ->Unit(benchmark::kMicrosecond);

RANGE_VS_MDRANGE_TRIAD1D_BENCHMARK_ARGS(BENCHMARK(RangePolicyTriad))
RANGE_VS_MDRANGE_TRIAD1D_BENCHMARK_ARGS(
    BENCHMARK_CAPTURE(MDRangePolicyTriad, Default, 0))
RANGE_VS_MDRANGE_TRIAD1D_BENCHMARK_ARGS(
    BENCHMARK_CAPTURE(MDRangePolicyTriad, Tile64, 64))
RANGE_VS_MDRANGE_TRIAD1D_BENCHMARK_ARGS(
    BENCHMARK_CAPTURE(MDRangePolicyTriad, Tile128, 128))
RANGE_VS_MDRANGE_TRIAD1D_BENCHMARK_ARGS(
    BENCHMARK_CAPTURE(MDRangePolicyTriad, Tile256, 256))
RANGE_VS_MDRANGE_TRIAD1D_BENCHMARK_ARGS(
    BENCHMARK_CAPTURE(MDRangePolicyTriad, Tile512, 512))

#undef RANGE_VS_MDRANGE_TRIAD1D_BENCHMARK_ARGS
#undef RANGE_VS_MDRANGE_TRIAD1D_SMALL_ARG_SIZE
#undef RANGE_VS_MDRANGE_TRIAD1D_SMALL_ARG_SIZE_UNALIGNED
#undef RANGE_VS_MDRANGE_TRIAD1D_LARGE_ARG_SIZE

}  // namespace
}  // namespace Benchmark
