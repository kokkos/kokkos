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

/**
 * @file PerfTest_Stream.cpp
 * @brief Implementation of STREAM benchmark operations for Kokkos.
 *
 * @details This file provides a set of memory bandwidth benchmarks based on the
 * STREAM benchmark suite. It implements the five core STREAM operations (Set,
 * Copy, Scale, Add, and Triad) using Kokkos parallel primitives. It includes
 * validation.
 *
 * The implementation strives to use as few Kokkos features as possible, and so
 * validation is performed on the host rather than via parallel_reduce
 */

#include <Kokkos_Core.hpp>
#include <benchmark/benchmark.h>
#include "Benchmark_Context.hpp"

constexpr static double aInit  = 1.0;
constexpr static double bInit  = 2.0;
constexpr static double cInit  = 3.0;
constexpr static double scalar = 4.0;

using StreamDeviceArray =
    Kokkos::View<double*, Kokkos::MemoryTraits<Kokkos::Restrict>>;
using StreamHostArray = typename StreamDeviceArray::HostMirror;

using StreamIndex =
    int64_t;  // different than benchmarks/stream, which uses int
using Policy = Kokkos::RangePolicy<Kokkos::IndexType<StreamIndex>>;

template <typename V>
void perform_set(V& a, const double scalar_) {
  Kokkos::parallel_for(
      "set", Policy(0, a.extent(0)),
      KOKKOS_LAMBDA(const StreamIndex i) { a[i] = scalar_; });

  Kokkos::fence();
}

template <typename V>
void perform_copy(V& a, V& b) {
  Kokkos::parallel_for(
      "copy", Policy(0, a.extent(0)),
      KOKKOS_LAMBDA(const StreamIndex i) { b[i] = a[i]; });

  Kokkos::fence();
}

template <typename V>
void perform_scale(V& b, V& c, const double scalar_) {
  Kokkos::parallel_for(
      "scale", Policy(0, b.extent(0)),
      KOKKOS_LAMBDA(const StreamIndex i) { b[i] = scalar_ * c[i]; });

  Kokkos::fence();
}

template <typename V>
void perform_add(V& a, V& b, V& c) {
  Kokkos::parallel_for(
      "add", Policy(0, a.extent(0)),
      KOKKOS_LAMBDA(const StreamIndex i) { c[i] = a[i] + b[i]; });

  Kokkos::fence();
}

template <typename V>
void perform_triad(V& a, V& b, V& c, const double scalar_) {
  Kokkos::parallel_for(
      "triad", Policy(0, a.extent(0)),
      KOKKOS_LAMBDA(const StreamIndex i) { a[i] = b[i] + scalar_ * c[i]; });

  Kokkos::fence();
}

template <typename V>
int validate_array(V& a_dev, const double expected) {
  auto a = Kokkos::create_mirror_view(a_dev);
  Kokkos::deep_copy(a, a_dev);

  double error = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    error += std::abs(a[i] - expected);
  }
  double avgError = error / (double)a.size();

  const double epsilon = 1.0e-13;
  return std::abs(avgError / expected) > epsilon;
}

template <unsigned MemTraits>
static void StreamSet(benchmark::State& state) {
  const size_t N8                 = std::pow(state.range(0), 8);
  static constexpr int DATA_RATIO = 1;

  Kokkos::View<double*, Kokkos::MemoryTraits<MemTraits>> a("a", N8);

  for (auto _ : state) {
    Kokkos::Timer timer;
    perform_set(a, scalar);
    KokkosBenchmark::report_results(state, a, DATA_RATIO, timer.seconds());
  }

  if (validate_array(a, scalar)) {
    state.SkipWithError("validation failure");
  }
}

template <unsigned MemTraits>
static void StreamCopy(benchmark::State& state) {
  const size_t N8                 = std::pow(state.range(0), 8);
  static constexpr int DATA_RATIO = 2;

  Kokkos::View<double*, Kokkos::MemoryTraits<MemTraits>> a("a", N8);
  Kokkos::View<double*, Kokkos::MemoryTraits<MemTraits>> b("b", N8);

  perform_set(a, aInit);

  for (auto _ : state) {
    Kokkos::Timer timer;
    perform_copy(a, b);
    KokkosBenchmark::report_results(state, a, DATA_RATIO, timer.seconds());
  }

  if (validate_array(b, aInit)) {
    state.SkipWithError("validation failure");
  }
}

template <unsigned MemTraits>
static void StreamScale(benchmark::State& state) {
  const size_t N8                 = std::pow(state.range(0), 8);
  static constexpr int DATA_RATIO = 2;

  Kokkos::View<double*, Kokkos::MemoryTraits<MemTraits>> a("a", N8);
  Kokkos::View<double*, Kokkos::MemoryTraits<MemTraits>> b("b", N8);

  perform_set(b, bInit);

  for (auto _ : state) {
    Kokkos::Timer timer;
    perform_scale(a, b, scalar);
    KokkosBenchmark::report_results(state, b, DATA_RATIO, timer.seconds());
  }

  if (validate_array(a, bInit * scalar)) {
    state.SkipWithError("validation failure");
  }
}

template <unsigned MemTraits>
static void StreamAdd(benchmark::State& state) {
  const size_t N8                 = std::pow(state.range(0), 8);
  static constexpr int DATA_RATIO = 3;

  Kokkos::View<double*, Kokkos::MemoryTraits<MemTraits>> a("a", N8);
  Kokkos::View<double*, Kokkos::MemoryTraits<MemTraits>> b("b", N8);
  Kokkos::View<double*, Kokkos::MemoryTraits<MemTraits>> c("c", N8);

  perform_set(a, aInit);
  perform_set(b, bInit);
  perform_set(c, cInit);

  for (auto _ : state) {
    Kokkos::Timer timer;
    perform_add(a, b, c);
    KokkosBenchmark::report_results(state, c, DATA_RATIO, timer.seconds());
  }

  if (validate_array(c, aInit + bInit)) {
    state.SkipWithError("validation failure");
  }
}

template <unsigned MemTraits>
static void StreamTriad(benchmark::State& state) {
  const size_t N8                 = std::pow(state.range(0), 8);
  static constexpr int DATA_RATIO = 3;

  Kokkos::View<double*, Kokkos::MemoryTraits<MemTraits>> a("a", N8);
  Kokkos::View<double*, Kokkos::MemoryTraits<MemTraits>> b("b", N8);
  Kokkos::View<double*, Kokkos::MemoryTraits<MemTraits>> c("c", N8);

  perform_set(a, aInit);
  perform_set(b, bInit);
  perform_set(c, cInit);

  for (auto _ : state) {
    Kokkos::Timer timer;
    perform_triad(a, b, c, scalar);
    KokkosBenchmark::report_results(state, a, DATA_RATIO, timer.seconds());
  }

  if (validate_array(a, bInit + scalar * cInit)) {
    state.SkipWithError("validation failure");
  }
}

// skips a benchmark with an error from thrown exceptions
template <void (*bm)(benchmark::State&)>
static void or_skip(benchmark::State& state) {
  try {
    bm(state);
  } catch (const std::runtime_error& e) {
    state.SkipWithError(e.what());
  }
};

namespace Test {

BENCHMARK(or_skip<StreamSet<0>>)
    ->Name("StreamSet")
    ->ArgName("N")
    ->Arg(10)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

BENCHMARK(or_skip<StreamSet<Kokkos::Restrict>>)
    ->Name("StreamSet<Restrict>")
    ->ArgName("N")
    ->Arg(10)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

BENCHMARK(or_skip<StreamCopy<0>>)
    ->Name("StreamCopy")
    ->ArgName("N")
    ->Arg(10)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

BENCHMARK(or_skip<StreamCopy<Kokkos::Restrict>>)
    ->Name("StreamCopy<Restrict>")
    ->ArgName("N")
    ->Arg(10)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

BENCHMARK(or_skip<StreamScale<0>>)
    ->Name("StreamScale")
    ->ArgName("N")
    ->Arg(10)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

BENCHMARK(or_skip<StreamScale<Kokkos::Restrict>>)
    ->Name("StreamScale<Restrict>")
    ->ArgName("N")
    ->Arg(10)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

BENCHMARK(or_skip<StreamAdd<0>>)
    ->Name("StreamAdd")
    ->ArgName("N")
    ->Arg(10)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

BENCHMARK(or_skip<StreamAdd<Kokkos::Restrict>>)
    ->Name("StreamAdd<Restrict>")
    ->ArgName("N")
    ->Arg(10)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

BENCHMARK(or_skip<StreamTriad<0>>)
    ->Name("StreamTriad")
    ->ArgName("N")
    ->Arg(10)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

BENCHMARK(or_skip<StreamTriad<Kokkos::Restrict>>)
    ->Name("StreamTriad<Restrict>")
    ->ArgName("N")
    ->Arg(10)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

}  // namespace Test
