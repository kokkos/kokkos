// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <cstdint>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
import kokkos.unordered_impl;
#else
#include <Kokkos_Core.hpp>
#include <Kokkos_UnorderedMap.hpp>
#endif

#include <TestDynRankView.hpp>
#include <TestGlobal2LocalIds.hpp>
#include <TestUnorderedMapPerformance.hpp>

namespace Performance {

void dynrankview_perf() {
  std::cout << "Cuda: dynrankview_perf" << std::endl;
  std::cout << " DynRankView vs View: Initialization Only " << std::endl;
  test_dynrankview_op_perf<Kokkos::Cuda>(40960);
}

void global_2_local() {
  std::cout << "Cuda: global_2_local" << std::endl;
  std::cout << "size, create, generate, fill, find" << std::endl;
  for (unsigned i = Performance::begin_id_size; i <= Performance::end_id_size;
       i *= Performance::id_step)
    test_global_to_local_ids<Kokkos::Cuda>(i);
}

void unordered_map_performance_near() {
  std::cout << "Cuda: unordered_map_performance_near" << std::endl;
  Perf::run_performance_tests<Kokkos::Cuda, true>("cuda-near");
}

void unordered_map_performance_far() {
  std::cout << "Cuda: unordered_map_performance_far" << std::endl;
  Perf::run_performance_tests<Kokkos::Cuda, false>("cuda-far");
}

}  // namespace Performance

int main() {
  Kokkos::ScopeGuard scope_guard;
  Performance::dynrankview_perf();
  Performance::global_2_local();
  Performance::unordered_map_performance_near();
  Performance::unordered_map_performance_far();
}
