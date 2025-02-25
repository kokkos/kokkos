#include <Kokkos_Core.hpp>
#include <benchmark/benchmark.h>

static void OperatorParentesisImpact_View(benchmark::State& state) {
  for (auto _ : state) {
    int N = state.range(0);

    Kokkos::View<double*> a("a", N);
    Kokkos::View<double*> b("b", N);

    Kokkos::parallel_for("", N, KOKKOS_LAMBDA(int i){
          a(i)+=b(i);
        });
    Kokkos::fence();
  }
}
BENCHMARK(OperatorParentesisImpact_View)
    ->ArgName("N")
    ->RangeMultiplier(16)
    ->Range(1, int64_t(1) << 20)
    ->Unit(benchmark::kMicrosecond);

static void OperatorParentesisImpact_Ptr1(benchmark::State& state) {
  for (auto _ : state) {
    int N = state.range(0);

    Kokkos::View<double*> a("a", N);
    Kokkos::View<double*> b("b", N);

    Kokkos::parallel_for("", N, KOKKOS_LAMBDA(int i){
          a.data()[i]+=b.data()[i];
        });
    Kokkos::fence();
  }
}
BENCHMARK(OperatorParentesisImpact_Ptr1)
    ->ArgName("N")
    ->RangeMultiplier(16)
    ->Range(1, int64_t(1) << 20)
    ->Unit(benchmark::kMicrosecond);

static void OperatorParentesisImpact_Ptr2(benchmark::State& state) {
  for (auto _ : state) {
    int N = state.range(0);

    Kokkos::View<double*> a("a", N);
    Kokkos::View<double*> b("b", N);

    double* aptr = a.data();
    double* bptr = b.data();

    Kokkos::parallel_for("", N, KOKKOS_LAMBDA(int i){
          aptr[i]+=bptr[i];
        });
    Kokkos::fence();
  }
}
BENCHMARK(OperatorParentesisImpact_Ptr2)
    ->ArgName("N")
    ->RangeMultiplier(16)
    ->Range(1, int64_t(1) << 20)
    ->Unit(benchmark::kMicrosecond);
