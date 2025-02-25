#include <Kokkos_Core.hpp>
#include <benchmark/benchmark.h>

static void KernalName_EmptyString(benchmark::State& state) {
  for (auto _ : state) {
    Kokkos::parallel_for("", 0, KOKKOS_LAMBDA(int){});
    Kokkos::fence();
  }
}
BENCHMARK(KernalName_EmptyString)
  ->Unit(benchmark::kMicrosecond);

static void KernalName_Defaulted(benchmark::State& state) {
  for (auto _ : state) {
    Kokkos::parallel_for(0, KOKKOS_LAMBDA(int){});
    Kokkos::fence();
  }
}
BENCHMARK(KernalName_Defaulted)
  ->Unit(benchmark::kMicrosecond);

static void KernalName_UserDefined(benchmark::State& state) {
  std::string lbl = "Hello World";
  for (auto _ : state) {
    Kokkos::parallel_for(lbl, 0, KOKKOS_LAMBDA(int){});
    Kokkos::fence();
  }
}
BENCHMARK(KernalName_UserDefined)
  ->Unit(benchmark::kMicrosecond);

static void KernalName_LongString(benchmark::State& state) {
  std::string lbl = "Hello World a long string that does not fit god dammit";
  for (int i = 0; i < 15; ++i) {
    lbl+=lbl;
  }
  for (auto _ : state) {
    Kokkos::parallel_for(lbl, 0, KOKKOS_LAMBDA(int){});
    Kokkos::fence();
  }
}
BENCHMARK(KernalName_LongString)
  ->Unit(benchmark::kMicrosecond);

struct StupidTagWithALongNameAAAAAAAAAAAAAAAAAAAA {};

static void KernalName_UserDefinedWithTag(benchmark::State& state) {
  std::string lbl = "Hello World";
  for (auto _ : state) {
    Kokkos::parallel_for(
        lbl,
        Kokkos::RangePolicy<StupidTagWithALongNameAAAAAAAAAAAAAAAAAAAA>(0, 0),
        KOKKOS_LAMBDA(StupidTagWithALongNameAAAAAAAAAAAAAAAAAAAA, int){});
    Kokkos::fence();
  }
}
BENCHMARK(KernalName_UserDefinedWithTag)
  ->Unit(benchmark::kMicrosecond);
