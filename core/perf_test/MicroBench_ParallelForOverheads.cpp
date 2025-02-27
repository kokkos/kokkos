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

/*
 * Set of micro benchmarks measuring the cost of initializing the different
 * parts surrounding a parallel_for
 */

static void ParallelFor_NoOverhead(benchmark::State& state) {
  Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> policy(0,0);
  std::string name = "name";
  for (auto _ : state) {
    Kokkos::parallel_for(name, policy, KOKKOS_LAMBDA(int){});
  }
}
BENCHMARK(ParallelFor_NoOverhead)
    ->Unit(benchmark::kNanosecond);

static void ParallelFor_OverheadDefaultName(benchmark::State& state) {
  Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> policy(0,0);
  for (auto _ : state) {
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(int){});
  }
}
BENCHMARK(ParallelFor_OverheadDefaultName)
    ->Unit(benchmark::kNanosecond);

static void ParallelFor_OverheadEmptyName(benchmark::State& state) {
  Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> policy(0,0);
  for (auto _ : state) {
    Kokkos::parallel_for("", policy, KOKKOS_LAMBDA(int){});
  }
}
BENCHMARK(ParallelFor_OverheadEmptyName)
    ->Unit(benchmark::kNanosecond);

static void ParallelFor_OverheadLongName(benchmark::State& state) {
  std::string name = "Hello World a long string that does not fit god dammit";
  for (int i = 0; i < 15; ++i) {
    name+=name;
  }
  Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> policy(0,0);
  for (auto _ : state) {
    Kokkos::parallel_for(name, policy, KOKKOS_LAMBDA(int){});
  }
}
BENCHMARK(ParallelFor_OverheadLongName)
    ->Unit(benchmark::kNanosecond);

struct StupidTagWithALongNameAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA {};

static void ParallelFor_OverheadLongTag(benchmark::State& state) {
  std::string name = "Hello World a long string that does not fit god dammit";
  Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, StupidTagWithALongNameAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA> policy(0,0);
  for (auto _ : state) {
    Kokkos::parallel_for(name, policy, KOKKOS_LAMBDA(StupidTagWithALongNameAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA, int){});
  }
}
BENCHMARK(ParallelFor_OverheadLongTag)
    ->Unit(benchmark::kNanosecond);

struct Func {
  void operator() (int) const {}
};
static void ParallelFor_OverheadFunctor(benchmark::State& state) {
  Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> policy(0,0);
  Func func;
  std::string name = "name";
  for (auto _ : state) {
    Kokkos::parallel_for("name", policy, func);
  }
}
BENCHMARK(ParallelFor_OverheadFunctor)
    ->Unit(benchmark::kNanosecond);

static void ParallelFor_OverheadPolicyCreation(benchmark::State& state) {
  std::string name = "name";
  for (auto _ : state) {
    Kokkos::parallel_for(name, 0, KOKKOS_LAMBDA(int){});
  }
}
BENCHMARK(ParallelFor_OverheadPolicyCreation)
    ->Unit(benchmark::kNanosecond);

static void ParallelFor_OverheadSpaceSpecificFence(benchmark::State& state) {
  Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> policy(0,0);
  std::string name = "name";
  Kokkos::DefaultExecutionSpace space;
  for (auto _ : state) {
    Kokkos::parallel_for(name, policy, KOKKOS_LAMBDA(int){});
    space.fence();
  }
}
BENCHMARK(ParallelFor_OverheadSpaceSpecificFence)
    ->Unit(benchmark::kNanosecond);

static void ParallelFor_OverheadSpaceCreation(benchmark::State& state) {
  Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> policy(0,0);
  std::string name = "name";
  for (auto _ : state) {
    Kokkos::parallel_for(name, policy, KOKKOS_LAMBDA(int){});
    Kokkos::DefaultExecutionSpace{}.fence();
  }
}
BENCHMARK(ParallelFor_OverheadSpaceCreation)
    ->Unit(benchmark::kNanosecond);

static void ParallelFor_OverheadGlobalFence(benchmark::State& state) {
  Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> policy(0,0);
  std::string name = "name";
  for (auto _ : state) {
    Kokkos::parallel_for(name, policy, KOKKOS_LAMBDA(int){});
    Kokkos::fence();
  }
}
BENCHMARK(ParallelFor_OverheadGlobalFence)
    ->Unit(benchmark::kNanosecond);

static void ParallelFor_OverheadFull(benchmark::State& state) {
  for (auto _ : state) {
    Kokkos::parallel_for("", 0, KOKKOS_LAMBDA(int){});
    Kokkos::fence();
  }
}
BENCHMARK(ParallelFor_OverheadFull)
    ->Unit(benchmark::kNanosecond);
