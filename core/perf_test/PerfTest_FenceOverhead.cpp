#include <Kokkos_Core.hpp>
#include <benchmark/benchmark.h>

static void FenceOverhead(benchmark::State& state) {
  for (auto _ : state) {
    Kokkos::parallel_for("", 0, KOKKOS_LAMBDA(int i){});
    Kokkos::fence();
  }
}
BENCHMARK(FenceOverhead)
    ->Unit(benchmark::kMicrosecond);

static void FenceOverhead_NoFence(benchmark::State& state) {
  for (auto _ : state) {
    Kokkos::parallel_for("", 0, KOKKOS_LAMBDA(int i){});
  }
}
BENCHMARK(FenceOverhead_NoFence)
    ->Unit(benchmark::kMicrosecond);

