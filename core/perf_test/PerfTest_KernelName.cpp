#include <Kokkos_Core.hpp>
#include <benchmark/benchmark.h>

static void BM_EmptyString(benchmark::State& state) {
  for (auto _ : state) {
    int r;
    Kokkos::parallel_reduce("", 0, KOKKOS_LAMBDA(int, int&){}, r);
  }
}
BENCHMARK(BM_EmptyString);

static void BM_Defaulted(benchmark::State& state) {
  for (auto _ : state) {
    int r;
    Kokkos::parallel_reduce(0, KOKKOS_LAMBDA(int, int&){}, r);
  }
}
BENCHMARK(BM_Defaulted);

static void BM_UserDefined(benchmark::State& state) {
  std::string lbl = "Hello World";
  for (auto _ : state) {
    int r;
    Kokkos::parallel_reduce(lbl, 0, KOKKOS_LAMBDA(int, int&){}, r);
  }
}
BENCHMARK(BM_UserDefined);

static void BM_LongString(benchmark::State& state) {
  std::string lbl = "Hello World a long string that does not fit god dammit";
  for (auto _ : state) {
    Kokkos::parallel_for(lbl, 0, KOKKOS_LAMBDA(int){});
  }
}
BENCHMARK(BM_LongString);

struct StupidTagWithALongNameAAAAAAAAAAAAAAAAAAAA {};

static void BM_UserDefinedWithTag(benchmark::State& state) {
  std::string lbl = "Hello World";
  for (auto _ : state) {
    Kokkos::parallel_for(
        lbl,
        Kokkos::RangePolicy<StupidTagWithALongNameAAAAAAAAAAAAAAAAAAAA>(0, 0),
        KOKKOS_LAMBDA(StupidTagWithALongNameAAAAAAAAAAAAAAAAAAAA, int){});
  }
}
BENCHMARK(BM_UserDefinedWithTag);
