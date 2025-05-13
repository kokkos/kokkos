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

// Minimal overhead, reuse the name, the policy and the lambda and only measure
// the time it takes to call Kokkos::parallel_for
static void ParallelFor_NoOverhead(benchmark::State& state) {
  std::string name = "kernel";
  Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> policy(0, 0);
  const auto lambda = KOKKOS_LAMBDA(int){};

  for (auto _ : state) {
    Kokkos::parallel_for(name, policy, lambda);
  }
}
BENCHMARK(ParallelFor_NoOverhead)->Unit(benchmark::kNanosecond);

// Cost of creating the default name for the kernel
static void ParallelFor_Overhead_DefaultedName(benchmark::State& state) {
  Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> policy(0, 0);
  const auto lambda = KOKKOS_LAMBDA(int){};

  for (auto _ : state) {
    Kokkos::parallel_for(policy, lambda);
  }
}
BENCHMARK(ParallelFor_Overhead_DefaultedName)->Unit(benchmark::kNanosecond);

// Cost of constructing a small string (it should fit in the small string
// optimisation) to use as the kernel name.
static void ParallelFor_Overhead_EmptyName(benchmark::State& state) {
  Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> policy(0, 0);
  const auto lambda = KOKKOS_LAMBDA(int){};

  for (auto _ : state) {
    Kokkos::parallel_for("", policy, lambda);
  }
}
BENCHMARK(ParallelFor_Overhead_EmptyName)->Unit(benchmark::kNanosecond);

// Is there a cost for using a very long name as kernel name?
// (checks that we don't uselessly copy the string around)
static void ParallelFor_Overhead_UsingLongKernelName(benchmark::State& state) {
  std::string name =
      "A very long string that should not be able to fit in the short string "
      "optimization of std::string";
  Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> policy(0, 0);
  const auto lambda = KOKKOS_LAMBDA(int){};

  for (auto _ : state) {
    Kokkos::parallel_for(name, policy, lambda);
  }
}
BENCHMARK(ParallelFor_Overhead_UsingLongKernelName)
    ->Unit(benchmark::kNanosecond);

// Cost for constructing a name that will not fit inside short string
// optimization
static void ParallelFor_Overhead_LongNameConstruction(benchmark::State& state) {
  Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> policy(0, 0);
  const auto lambda = KOKKOS_LAMBDA(int){};

  for (auto _ : state) {
    Kokkos::parallel_for("Name not fitting in the short string optimisation",
                         policy, lambda);
  }
}
BENCHMARK(ParallelFor_Overhead_LongNameConstruction)
    ->Unit(benchmark::kNanosecond);

// Is there a cost for creating the lambda?
static void ParallelFor_Overhead_LambdaCreation(benchmark::State& state) {
  std::string name = "kernel";
  Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> policy(0, 0);

  for (auto _ : state) {
    Kokkos::parallel_for(name, policy, KOKKOS_LAMBDA(int){});
  }
}
BENCHMARK(ParallelFor_Overhead_LambdaCreation)->Unit(benchmark::kNanosecond);

// Is there a cost for using a functor instead of a lambda?
struct Func {
  KOKKOS_FUNCTION void operator()(int) const {}
};
static void ParallelFor_Overhead_Functor(benchmark::State& state) {
  std::string name = "kernel";
  Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> policy(0, 0);
  Func func;

  for (auto _ : state) {
    Kokkos::parallel_for(name, policy, func);
  }
}
BENCHMARK(ParallelFor_Overhead_Functor)->Unit(benchmark::kNanosecond);

// Is there a cost for using a functor with a very long name?
struct
    FunctorWithAVeryLongNameAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA {
  KOKKOS_FUNCTION void operator()(int) const {}
};
static void ParallelFor_Overhead_FunctorLongName(benchmark::State& state) {
  std::string name = "kernel";
  Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> policy(0, 0);
  FunctorWithAVeryLongNameAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
      func;

  for (auto _ : state) {
    Kokkos::parallel_for(name, policy, func);
  }
}
BENCHMARK(ParallelFor_Overhead_FunctorLongName)->Unit(benchmark::kNanosecond);

// Is there a cost for using a very long name for the range policy tag?
struct
    TagWithALongNameAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA {
};

static void ParallelFor_Overhead_LongTag(benchmark::State& state) {
  std::string name = "kernel";
  Kokkos::RangePolicy<
      Kokkos::DefaultExecutionSpace,
      TagWithALongNameAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA>
      policy(0, 0);
  const auto lambda = KOKKOS_LAMBDA(
      TagWithALongNameAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA,
      int){};

  for (auto _ : state) {
    Kokkos::parallel_for(name, policy, lambda);
  }
}
BENCHMARK(ParallelFor_Overhead_LongTag)->Unit(benchmark::kNanosecond);

// Cost of using the default range policy (and thus incur the cost of creating
// it)
static void ParallelFor_Overhead_PolicyCreation(benchmark::State& state) {
  std::string name  = "kernel";
  const auto lambda = KOKKOS_LAMBDA(int){};

  for (auto _ : state) {
    Kokkos::parallel_for(name, 0, lambda);
  }
}
BENCHMARK(ParallelFor_Overhead_PolicyCreation)->Unit(benchmark::kNanosecond);

// Fences cost

// Cost of calling a space specific fence in-between kernel
static void ParallelFor_Overhead_SpaceSpecificFence(benchmark::State& state) {
  std::string name = "kernel";
  Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> policy(0, 0);
  const auto lambda = KOKKOS_LAMBDA(int){};
  Kokkos::DefaultExecutionSpace space;

  for (auto _ : state) {
    Kokkos::parallel_for(name, policy, lambda);
    space.fence();
  }
}
BENCHMARK(ParallelFor_Overhead_SpaceSpecificFence)
    ->Unit(benchmark::kNanosecond);

// Cost of calling a space specific fence with a non defaulted name in-between
// kernel
static void ParallelFor_Overhead_NamedSpaceSpecificFence(
    benchmark::State& state) {
  std::string kernel_name = "kernel";
  const auto lambda       = KOKKOS_LAMBDA(int){};
  Kokkos::DefaultExecutionSpace space;
  Kokkos::RangePolicy policy(space, 0, 0);
  std::string fence_name = "fence";

  for (auto _ : state) {
    Kokkos::parallel_for(kernel_name, policy, lambda);
    space.fence(fence_name);
  }
}
BENCHMARK(ParallelFor_Overhead_NamedSpaceSpecificFence)
    ->Unit(benchmark::kNanosecond);

// Cost of calling a fence over a specific space in-between kernel,
// when this space needs to be created.
static void ParallelFor_Overhead_FenceWithSpaceCreation(
    benchmark::State& state) {
  std::string name = "kernel";
  Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> policy(0, 0);
  const auto lambda = KOKKOS_LAMBDA(int){};

  for (auto _ : state) {
    Kokkos::parallel_for(name, policy, lambda);
    Kokkos::DefaultExecutionSpace{}.fence();
  }
}
BENCHMARK(ParallelFor_Overhead_FenceWithSpaceCreation)
    ->Unit(benchmark::kNanosecond);

// Cost of calling a global fence with a non defaulted name in-between kernels
static void ParallelFor_Overhead_NamedGlobalFence(benchmark::State& state) {
  std::string kernel_name = "kernel";
  Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> policy(0, 0);
  const auto lambda      = KOKKOS_LAMBDA(int){};
  std::string fence_name = "fence";

  for (auto _ : state) {
    Kokkos::parallel_for(kernel_name, policy, lambda);
    Kokkos::fence(fence_name);
  }
}
BENCHMARK(ParallelFor_Overhead_NamedGlobalFence)->Unit(benchmark::kNanosecond);

// Cost of calling a global fence in-between kernel (including fence name
// creation cost)
static void ParallelFor_Overhead_GlobalFence(benchmark::State& state) {
  std::string name = "kernel";
  Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> policy(0, 0);
  const auto lambda = KOKKOS_LAMBDA(int){};

  for (auto _ : state) {
    Kokkos::parallel_for(name, policy, lambda);
    Kokkos::fence();
  }
}
BENCHMARK(ParallelFor_Overhead_GlobalFence)->Unit(benchmark::kNanosecond);

// Full overhead for a classic call to parallel_for
static void ParallelFor_Overhead_Full(benchmark::State& state) {
  for (auto _ : state) {
    Kokkos::parallel_for("Name not fitting in the short string optimisation", 0,
                         KOKKOS_LAMBDA(int){});
    Kokkos::fence();
  }
}
BENCHMARK(ParallelFor_Overhead_Full)->Unit(benchmark::kNanosecond);
