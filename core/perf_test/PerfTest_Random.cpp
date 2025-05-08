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
#include <Kokkos_Random.hpp>
#include <benchmark/benchmark.h>
#include "Benchmark_Context.hpp"

namespace Benchmark {

// Fills each entry of
// * a view of size N
// * with the sum of K random numbers
// * between 0 and I
template <typename Pool, typename Scalar>
static void Random(benchmark::State &state) {
  const size_t N = state.range(0);
  const size_t K = state.range(1);
  const size_t I = state.range(2);

  Kokkos::View<Scalar *> out("out", N);
  Pool random_pool(/*seed=*/12345);

  for (auto _ : state) {
    Kokkos::parallel_for(
        N, KOKKOS_LAMBDA(const int i) {
          auto generator = random_pool.get_state();
          Scalar acc     = 0;

          // work around "cannot first-capture I in a constexpr-if context"
          auto SI = Scalar(I);
          for (size_t k = 0; k < K; ++k) {
            if constexpr (std::is_same_v<Scalar, double>) {
              acc += generator.drand(SI);
            } else if constexpr (std::is_same_v<Scalar, float>) {
              acc += generator.frand(SI);
            } else if constexpr (std::is_same_v<Scalar, uint64_t>) {
              acc += generator.urand64(SI);
            } else if constexpr (std::is_same_v<Scalar, uint32_t>) {
              acc += generator.urand(SI);
            } else if constexpr (std::is_same_v<Scalar, int64_t>) {
              acc += generator.rand64(SI);
            } else if constexpr (std::is_same_v<Scalar, int32_t>) {
              acc += generator.rand(SI);
            } else {
              static_assert(std::is_void_v<Scalar>, "unhandled Scalar type");
            }
          }
          random_pool.free_state(generator);
          out(i) = acc;
        });
    Kokkos::fence("");
  }

  state.counters[KokkosBenchmark::benchmark_fom("rate")] = benchmark::Counter(
      state.iterations() * N * K, benchmark::Counter::kIsRate);
}

template <typename Scalar>
static void Random64(benchmark::State &state) {
  return Random<Kokkos::Random_XorShift64_Pool<>, Scalar>(state);
}

template <typename Scalar>
static void Random1024(benchmark::State &state) {
  return Random<Kokkos::Random_XorShift1024_Pool<>, Scalar>(state);
}

#define RANDOM_ARGS()                                                   \
  ArgNames({"N", "K", "I"})                                             \
      ->ArgsProduct({{1 << 15, 1 << 21}, {1, 256}, {1, 1'000'000'000}}) \
      ->UseRealTime()                                                   \
      ->Unit(benchmark::kMicrosecond)

BENCHMARK(Random64<double>)->RANDOM_ARGS();
BENCHMARK(Random64<float>)->RANDOM_ARGS();
BENCHMARK(Random64<uint64_t>)->RANDOM_ARGS();
BENCHMARK(Random64<uint32_t>)->RANDOM_ARGS();
BENCHMARK(Random64<int64_t>)->RANDOM_ARGS();
BENCHMARK(Random64<int32_t>)->RANDOM_ARGS();

BENCHMARK(Random1024<double>)->RANDOM_ARGS();
BENCHMARK(Random1024<float>)->RANDOM_ARGS();
BENCHMARK(Random1024<uint64_t>)->RANDOM_ARGS();
BENCHMARK(Random1024<uint32_t>)->RANDOM_ARGS();
BENCHMARK(Random1024<int64_t>)->RANDOM_ARGS();
BENCHMARK(Random1024<int32_t>)->RANDOM_ARGS();

#undef RANDOM_ARGS

}  // namespace Benchmark
