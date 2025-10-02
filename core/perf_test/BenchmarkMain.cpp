// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <benchmark/benchmark.h>

#include "Benchmark_Context.hpp"
#include <Kokkos_Core.hpp>

#include "PerfTest_Category.hpp"

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  benchmark::Initialize(&argc, argv);
  // FIXME: seconds as default time unit leads to precision loss
  benchmark::SetDefaultTimeUnit(benchmark::kSecond);
  KokkosBenchmark::add_benchmark_context(true);

  (void)Test::command_line_num_args(argc);
  (void)Test::command_line_arg(0, argv);

  benchmark::RunSpecifiedBenchmarks();

  benchmark::Shutdown();
  Kokkos::finalize();
  return 0;
}
