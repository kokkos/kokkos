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
  Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> policy(0, 0);
  std::string name = "name";
  for (auto _ : state) {
    Kokkos::parallel_for(name, policy, KOKKOS_LAMBDA(int){});
  }
}
BENCHMARK(ParallelFor_NoOverhead)->Unit(benchmark::kNanosecond);

static void ParallelFor_Overhead_DefaultName(benchmark::State& state) {
  Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> policy(0, 0);
  for (auto _ : state) {
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(int){});
  }
}
BENCHMARK(ParallelFor_Overhead_DefaultName)->Unit(benchmark::kNanosecond);

static void ParallelFor_Overhead_EmptyName(benchmark::State& state) {
  Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> policy(0, 0);
  for (auto _ : state) {
    Kokkos::parallel_for("", policy, KOKKOS_LAMBDA(int){});
  }
}
BENCHMARK(ParallelFor_Overhead_EmptyName)->Unit(benchmark::kNanosecond);

static void ParallelFor_Overhead_LongName(benchmark::State& state) {
  std::string name = "Hello World a long string that does not fit god dammit";
  for (int i = 0; i < 15; ++i) {
    name += name;
  }
  Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> policy(0, 0);
  for (auto _ : state) {
    Kokkos::parallel_for(name, policy, KOKKOS_LAMBDA(int){});
  }
}
BENCHMARK(ParallelFor_Overhead_LongName)->Unit(benchmark::kNanosecond);

struct
    StupidTagWithALongNameAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA {
};

static void ParallelFor_Overhead_LongTag(benchmark::State& state) {
  std::string name = "Hello World a long string that does not fit god dammit";
  Kokkos::RangePolicy<
      Kokkos::DefaultExecutionSpace,
      StupidTagWithALongNameAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA>
      policy(0, 0);
  for (auto _ : state) {
    Kokkos::parallel_for(
        name, policy,
        KOKKOS_LAMBDA(
            StupidTagWithALongNameAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA,
            int){});
  }
}
BENCHMARK(ParallelFor_Overhead_LongTag)->Unit(benchmark::kNanosecond);

struct Func {
  KOKKOS_FUNCTION void operator()(int) const {}
};
static void ParallelFor_Overhead_Functor(benchmark::State& state) {
  Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> policy(0, 0);
  Func func;
  std::string name = "name";
  for (auto _ : state) {
    Kokkos::parallel_for("name", policy, func);
  }
}
BENCHMARK(ParallelFor_Overhead_Functor)->Unit(benchmark::kNanosecond);

static void ParallelFor_Overhead_PolicyCreation(benchmark::State& state) {
  std::string name = "name";
  for (auto _ : state) {
    Kokkos::parallel_for(name, 0, KOKKOS_LAMBDA(int){});
  }
}
BENCHMARK(ParallelFor_Overhead_PolicyCreation)->Unit(benchmark::kNanosecond);

static void ParallelFor_Overhead_SpaceSpecificFence(benchmark::State& state) {
  Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> policy(0, 0);
  std::string name = "name";
  Kokkos::DefaultExecutionSpace space;
  for (auto _ : state) {
    Kokkos::parallel_for(name, policy, KOKKOS_LAMBDA(int){});
    space.fence();
  }
}
BENCHMARK(ParallelFor_Overhead_SpaceSpecificFence)
    ->Unit(benchmark::kNanosecond);

static void ParallelFor_Overhead_NamedSpaceSpecificFence(
    benchmark::State& state) {
  Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> policy(0, 0);
  std::string kernel_name = "kernel";
  std::string fence_name  = "fence";
  Kokkos::DefaultExecutionSpace space;
  for (auto _ : state) {
    Kokkos::parallel_for(kernel_name, policy, KOKKOS_LAMBDA(int){});
    space.fence(fence_name);
  }
}
BENCHMARK(ParallelFor_Overhead_NamedSpaceSpecificFence)
    ->Unit(benchmark::kNanosecond);

static void ParallelFor_Overhead_FenceWithSpaceCreation(
    benchmark::State& state) {
  Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> policy(0, 0);
  std::string name = "name";
  for (auto _ : state) {
    Kokkos::parallel_for(name, policy, KOKKOS_LAMBDA(int){});
    Kokkos::DefaultExecutionSpace{}.fence();
  }
}
BENCHMARK(ParallelFor_Overhead_FenceWithSpaceCreation)
    ->Unit(benchmark::kNanosecond);

static void ParallelFor_Overhead_GlobalFence(benchmark::State& state) {
  Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> policy(0, 0);
  std::string name = "name";
  for (auto _ : state) {
    Kokkos::parallel_for(name, policy, KOKKOS_LAMBDA(int){});
    Kokkos::fence();
  }
}
BENCHMARK(ParallelFor_Overhead_GlobalFence)->Unit(benchmark::kNanosecond);

static void ParallelFor_Overhead_NamedGlobalFence(benchmark::State& state) {
  Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> policy(0, 0);
  std::string kernel_name = "kernel";
  std::string fence_name  = "fence";
  for (auto _ : state) {
    Kokkos::parallel_for(kernel_name, policy, KOKKOS_LAMBDA(int){});
    Kokkos::fence(fence_name);
  }
}
BENCHMARK(ParallelFor_Overhead_NamedGlobalFence)->Unit(benchmark::kNanosecond);

static void ParallelFor_FenceOnly(benchmark::State& state) {
  for (auto _ : state) {
    Kokkos::fence();
  }
}
BENCHMARK(ParallelFor_FenceOnly)->Unit(benchmark::kNanosecond);

static void ParallelFor_Overhead_Full(benchmark::State& state) {
  for (auto _ : state) {
    Kokkos::parallel_for("", 0, KOKKOS_LAMBDA(int){});
    Kokkos::fence();
  }
}
BENCHMARK(ParallelFor_Overhead_Full)->Unit(benchmark::kNanosecond);
