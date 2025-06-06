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
#include <iostream>
#include <vector>
#if !defined(KOKKOS_ENABLE_CXX17)
#include <memory>
#endif

/*
 * Set of micro benchmarks comparing different ways of accessing a
 * Kokkos::View, and putting it into perspective by comparing with access
 * time through native container types
 */

void init(Kokkos::View<double*> a, Kokkos::View<double*> b, int N) {
  Kokkos::parallel_for(
      "Init", N, KOKKOS_LAMBDA(int i) {
        a(i) = i;
        b(i) = N - i;
      });
  Kokkos::fence();
}

void init(double* a, double* b, int N) {
  for (int i = 0; i < N; ++i) {
    a[i] = i;
    b[i] = N - i;
  }
}

void check_errors(Kokkos::View<double*> c, int N) {
  int errors;
  Kokkos::fence();
  Kokkos::parallel_reduce(
      "Find errors", N,
      KOKKOS_LAMBDA(int i, int& error) { error += (c(i) != N); }, errors);

  if (errors != 0) {
    std::cout << "Error " << errors << "\n";
    return;
  }
}

void check_errors(double* c, int N) {
  int errors = 0;

  for (int i = 0; i < N; ++i) {
    errors += (c[i] != N);
  }

  if (errors != 0) {
    std::cout << "Error " << errors << "\n";
    return;
  }
}

static void OperatorParentesisImpact_View(benchmark::State& state) {
  int N = state.range(0);

  Kokkos::View<double*> a("a", N);
  Kokkos::View<double*> b("b", N);
  Kokkos::View<double*> c("c", N);
  Kokkos::fence();

  init(a, b, N);

  std::string kernel_name = "kernel";
  Kokkos::DefaultExecutionSpace space;
  Kokkos::RangePolicy policy(space, 0, N);
  const auto lambda      = KOKKOS_LAMBDA(int i) { c(i) = a(i) + b(i); };
  std::string fence_name = "fence_name";

  for (auto _ : state) {
    Kokkos::parallel_for(kernel_name, policy, lambda);
    space.fence(fence_name);
  }

  check_errors(c, N);
}
BENCHMARK(OperatorParentesisImpact_View)
    ->ArgName("N")
    ->RangeMultiplier(16)
    ->Range(1, int64_t(1) << 20)
    ->Unit(benchmark::kMicrosecond);

static void OperatorParentesisImpact_Bracket(benchmark::State& state) {
  int N = state.range(0);

  Kokkos::View<double*> a("a", N);
  Kokkos::View<double*> b("b", N);
  Kokkos::View<double*> c("c", N);
  Kokkos::fence();

  init(a, b, N);

  std::string kernel_name = "kernel";
  Kokkos::DefaultExecutionSpace space;
  Kokkos::RangePolicy policy(space, 0, N);
  const auto lambda      = KOKKOS_LAMBDA(int i) { c[i] = a[i] + b[i]; };
  std::string fence_name = "fence_name";

  for (auto _ : state) {
    Kokkos::parallel_for(kernel_name, policy, lambda);
    space.fence(fence_name);
  }

  check_errors(c, N);
}
BENCHMARK(OperatorParentesisImpact_Bracket)
    ->ArgName("N")
    ->RangeMultiplier(16)
    ->Range(1, int64_t(1) << 20)
    ->Unit(benchmark::kMicrosecond);

static void OperatorParentesisImpact_Hybrid(benchmark::State& state) {
  int N = state.range(0);

  Kokkos::View<double*> a("a", N);
  Kokkos::View<double*> b("b", N);
  Kokkos::View<double*> c("c", N);
  Kokkos::fence();

  init(a, b, N);

  std::string kernel_name = "kernel";
  Kokkos::DefaultExecutionSpace space;
  Kokkos::RangePolicy policy(space, 0, N);
  const auto lambda = KOKKOS_LAMBDA(int i) {
    c.data()[i] = a.data()[i] + b.data()[i];
  };
  std::string fence_name = "fence_name";

  for (auto _ : state) {
    Kokkos::parallel_for(kernel_name, policy, lambda);
    space.fence(fence_name);
  }
  check_errors(c, N);
}
BENCHMARK(OperatorParentesisImpact_Hybrid)
    ->ArgName("N")
    ->RangeMultiplier(16)
    ->Range(1, int64_t(1) << 20)
    ->Unit(benchmark::kMicrosecond);

static void OperatorParentesisImpact_Ptr(benchmark::State& state) {
  int N = state.range(0);

  Kokkos::View<double*> a("a", N);
  Kokkos::View<double*> b("b", N);
  Kokkos::View<double*> c("c", N);
  Kokkos::fence();

  init(a, b, N);

  std::string kernel_name = "kernel";
  Kokkos::DefaultExecutionSpace space;
  Kokkos::RangePolicy policy(space, 0, N);
#if !defined(KOKKOS_ENABLE_CXX17)
  double* aptr = std::assume_aligned<256>(a.data());
  double* bptr = std::assume_aligned<256>(b.data());
  double* cptr = std::assume_aligned<256>(c.data());
#else
  double* aptr = a.data();
  double* bptr = b.data();
  double* cptr = c.data();
#endif
  const auto lambda = KOKKOS_LAMBDA(int i) { cptr[i] = aptr[i] + bptr[i]; };
  std::string fence_name = "fence_name";

  for (auto _ : state) {
    Kokkos::parallel_for(kernel_name, policy, lambda);
    space.fence(fence_name);
  }

  check_errors(c, N);
}
BENCHMARK(OperatorParentesisImpact_Ptr)
    ->ArgName("N")
    ->RangeMultiplier(16)
    ->Range(1, int64_t(1) << 20)
    ->Unit(benchmark::kMicrosecond);

static void OperatorParentesisImpact_CPUVector(benchmark::State& state) {
  int N = state.range(0);

  std::vector<double> a(N);
  std::vector<double> b(N);
  std::vector<double> c(N);

  init(a.data(), b.data(), N);

  for (auto _ : state) {
    for (int i = 0; i < N; ++i) {
      c[i] = a[i] + b[i];
    }
  }

  check_errors(c.data(), N);
}
BENCHMARK(OperatorParentesisImpact_CPUVector)
    ->ArgName("N")
    ->RangeMultiplier(16)
    ->Range(1, int64_t(1) << 20)
    ->Unit(benchmark::kMicrosecond);

static void OperatorParentesisImpact_CPUPtr(benchmark::State& state) {
  int N = state.range(0);

  double* aptr =
      static_cast<double*>(std::aligned_alloc(size_t(256), sizeof(double) * N));
  double* bptr =
      static_cast<double*>(std::aligned_alloc(size_t(256), sizeof(double) * N));
  double* cptr =
      static_cast<double*>(std::aligned_alloc(size_t(256), sizeof(double) * N));

  init(aptr, bptr, N);

  for (auto _ : state) {
    for (int i = 0; i < N; ++i) {
      cptr[i] = aptr[i] + bptr[i];
    }
  }

  check_errors(cptr, N);

  std::free(aptr);
  std::free(bptr);
  std::free(cptr);
}
BENCHMARK(OperatorParentesisImpact_CPUPtr)
    ->ArgName("N")
    ->RangeMultiplier(16)
    ->Range(1, int64_t(1) << 20)
    ->Unit(benchmark::kMicrosecond);

#if defined(__AVX2__)
#include <immintrin.h>
static void OperatorParentesisImpact_CPUVectorized(benchmark::State& state) {
  int N = state.range(0);

  double* aptr =
      static_cast<double*>(std::aligned_alloc(size_t(256), sizeof(double) * N));
  double* bptr =
      static_cast<double*>(std::aligned_alloc(size_t(256), sizeof(double) * N));
  double* cptr =
      static_cast<double*>(std::aligned_alloc(size_t(256), sizeof(double) * N));

  init(aptr, bptr, N);

  for (auto _ : state) {
    int i           = 0;
    const int bound = (N / 4) * 4;
    for (; i < bound; i += 4) {
      __m256d av   = _mm256_load_pd(aptr + i);
      __m256d bv   = _mm256_load_pd(bptr + i);
      __m256d resv = _mm256_add_pd(av, bv);
      _mm256_store_pd(cptr + i, resv);
    }

    for (; i < N; ++i) {
      cptr[i] = bptr[i] + aptr[i];
    }
  }

  check_errors(cptr, N);

  std::free(aptr);
  std::free(bptr);
  std::free(cptr);
}
BENCHMARK(OperatorParentesisImpact_CPUVectorized)
    ->ArgName("N")
    ->RangeMultiplier(16)
    ->Range(1, int64_t(1) << 20)
    ->Unit(benchmark::kMicrosecond);
#endif
