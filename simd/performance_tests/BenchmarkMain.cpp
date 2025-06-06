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

#include <benchmark/benchmark.h>
#include <Kokkos_Core.hpp>

#include <Benchmark_Context.hpp>
#if defined(KOKKOS_IMPL_SIMD_HOST_PERFTEST)
#include <PerfTest_Host.hpp>
#else
#include <PerfTest_Device.hpp>
#endif

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);

#if defined(KOKKOS_IMPL_SIMD_HOST_PERFTEST)
  register_host_benchmarks();
#else
  register_device_benchmarks();
#endif

  benchmark::Initialize(&argc, argv);
  benchmark::SetDefaultTimeUnit(benchmark::kMillisecond);
  KokkosBenchmark::add_benchmark_context(true);

  benchmark::RunSpecifiedBenchmarks();

  benchmark::Shutdown();
  Kokkos::finalize();
  return 0;
}
