// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Macros.hpp>

#include <cstdint>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
import kokkos.unordered_map;
#else
#include <Kokkos_Core.hpp>
#include <Kokkos_UnorderedMap.hpp>
#endif

#include <TestDynRankView.hpp>
#include <TestGlobal2LocalIds.hpp>
#include <TestUnorderedMapPerformance.hpp>

namespace Performance {

void dynrankview_perf() {
  std::cout << "HIP: dynrankview_perf" << std::endl;
  std::cout << " DynRankView vs View: Initialization Only " << std::endl;
  test_dynrankview_op_perf<Kokkos::HIP>(40960);
}

void global_2_local() {
  std::cout << "HIP: global_2_local" << std::endl;
  std::cout << "size, create, generate, fill, find" << std::endl;
  for (unsigned i = Performance::begin_id_size; i <= Performance::end_id_size;
       i *= Performance::id_step)
    test_global_to_local_ids<Kokkos::HIP>(i);
}

void unordered_map_performance_near() {
  std::cout << "HIP: unordered_map_performance_near" << std::endl;
  Perf::run_performance_tests<Kokkos::HIP, true>("hip-near");
}

void unordered_map_performance_far() {
  std::cout << "HIP: unordered_map_performance_far" << std::endl;
  Perf::run_performance_tests<Kokkos::HIP, false>("hip-far");
}

}  // namespace Performance

int main() {
  Kokkos::ScopeGuard scope_guard;
  Performance::dynrankview_perf();
  Performance::global_2_local();
  Performance::unordered_map_performance_near();
  Performance::unordered_map_performance_far();
}
