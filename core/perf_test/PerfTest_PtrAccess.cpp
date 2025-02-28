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
#include <benchmark/benchmark.h>

static void OperatorParentesisImpact_View(benchmark::State& state) {
  int N = state.range(0);

  Kokkos::View<double*> a("a", N);
  Kokkos::View<double*> b("b", N);

  Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> policy(0, N);
  std::string name = "name";

  for (auto _ : state) {
    Kokkos::parallel_for(
        name, policy, KOKKOS_LAMBDA(int i) { a(i) += b(i); });
    Kokkos::fence();
  }
}
BENCHMARK(OperatorParentesisImpact_View)
    ->ArgName("N")
    ->RangeMultiplier(16)
    ->Range(1, int64_t(1) << 20)
    ->Unit(benchmark::kNanosecond);

static void OperatorParentesisImpact_Bracket(benchmark::State& state) {
  int N = state.range(0);
  Kokkos::View<double*> a("a", N);
  Kokkos::View<double*> b("b", N);
  Kokkos::fence();

  Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> policy(0, N);
  std::string name = "name";

  for (auto _ : state) {
    Kokkos::parallel_for(
        name, policy, KOKKOS_LAMBDA(int i) { a[i] += b[i]; });
    Kokkos::fence();
  }
}
BENCHMARK(OperatorParentesisImpact_Bracket)
    ->ArgName("N")
    ->RangeMultiplier(16)
    ->Range(1, int64_t(1) << 20)
    ->Unit(benchmark::kNanosecond);

static void OperatorParentesisImpact_Hybrid(benchmark::State& state) {
  int N = state.range(0);
  Kokkos::View<double*> a("a", N);
  Kokkos::View<double*> b("b", N);
  Kokkos::fence();

  Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> policy(0, N);
  std::string name = "name";

  for (auto _ : state) {
    Kokkos::parallel_for(
        name, policy, KOKKOS_LAMBDA(int i) { a.data()[i] += b.data()[i]; });
    Kokkos::fence();
  }
}
BENCHMARK(OperatorParentesisImpact_Hybrid)
    ->ArgName("N")
    ->RangeMultiplier(16)
    ->Range(1, int64_t(1) << 20)
    ->Unit(benchmark::kNanosecond);

static void OperatorParentesisImpact_Ptr(benchmark::State& state) {
  int N = state.range(0);

  Kokkos::View<double*> a("a", N);
  Kokkos::View<double*> b("b", N);
  Kokkos::fence();

  Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> policy(0, N);
  std::string name = "name";

  double* aptr = static_cast<double*>(__builtin_assume_aligned(a.data(), 256));
  double* bptr = static_cast<double*>(__builtin_assume_aligned(b.data(), 256));

  for (auto _ : state) {
    Kokkos::parallel_for(
        name, policy, KOKKOS_LAMBDA(int i) { aptr[i] += bptr[i]; });
    Kokkos::fence();
  }
}
BENCHMARK(OperatorParentesisImpact_Ptr)
    ->ArgName("N")
    ->RangeMultiplier(16)
    ->Range(1, int64_t(1) << 20)
    ->Unit(benchmark::kNanosecond);

static void OperatorParentesisImpact_Manual(benchmark::State& state) {
  int N = state.range(0);

  double* aptr =
      static_cast<double*>(aligned_alloc(size_t(256), sizeof(double) * N));
  double* bptr =
      static_cast<double*>(aligned_alloc(size_t(256), sizeof(double) * N));

  auto f = [=](int i) { aptr[i] += bptr[i]; };
  for (auto _ : state) {
    for (int i = 0; i < N; ++i) {
      f(i);
    }
  }
}
BENCHMARK(OperatorParentesisImpact_Manual)
    ->ArgName("N")
    ->RangeMultiplier(16)
    ->Range(1, int64_t(1) << 20)
    ->Unit(benchmark::kNanosecond);

#include <immintrin.h>
static void OperatorParentesisImpact_Vectorized(benchmark::State& state) {
  int N = state.range(0);

  double* aptr =
      static_cast<double*>(aligned_alloc(size_t(256), sizeof(double) * N));
  double* bptr =
      static_cast<double*>(aligned_alloc(size_t(256), sizeof(double) * N));

  for (auto _ : state) {
    int i           = 0;
    const int bound = N - (N % 4);
    for (; i < bound; i += 4) {
      __m256d av   = _mm256_load_pd(aptr + i);
      __m256d bv   = _mm256_load_pd(bptr + i);
      __m256d resv = _mm256_add_pd(av, bv);
      _mm256_store_pd(aptr + i, resv);
    }

    for (; i < N; ++i) {
      aptr[i] += bptr[i];
    }
  }
}
BENCHMARK(OperatorParentesisImpact_Vectorized)
    ->ArgName("N")
    ->RangeMultiplier(16)
    ->Range(1, int64_t(1) << 20)
    ->Unit(benchmark::kNanosecond);
