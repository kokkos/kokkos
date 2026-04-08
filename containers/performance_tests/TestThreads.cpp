// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
import kokkos.unordered_map;
#else
#include <Kokkos_Core.hpp>
#include <Kokkos_UnorderedMap.hpp>
#endif

#include <iomanip>

#include <TestGlobal2LocalIds.hpp>
#include <TestUnorderedMapPerformance.hpp>

#include <TestDynRankView.hpp>
#include <TestScatterView.hpp>

#include <iomanip>
#include <sstream>
#include <string>
#include <fstream>

namespace Performance {

void dynrankview_perf() {
  std::cout << "Threads: dynrankvie_perf" << std::endl;
  std::cout << " DynRankView vs View: Initialization Only " << std::endl;
  test_dynrankview_op_perf<Kokkos::Threads>(8192);
}

void global_2_local() {
  std::cout << "Threads: global_2_local" << std::endl;
  std::cout << "size, create, generate, fill, find" << std::endl;
  for (unsigned i = Performance::begin_id_size; i <= Performance::end_id_size;
       i *= Performance::id_step)
    test_global_to_local_ids<Kokkos::Threads>(i);
}

void unordered_map_performance_near() {
  std::cout << "Threads: unordered_map_performance_near" << std::endl;
  unsigned num_threads = 4;
  if (Kokkos::hwloc::available()) {
    num_threads = Kokkos::hwloc::get_available_numa_count() *
                  Kokkos::hwloc::get_available_cores_per_numa() *
                  Kokkos::hwloc::get_available_threads_per_core();
  }
  std::ostringstream base_file_name;
  base_file_name << "threads-" << num_threads << "-near";
  Perf::run_performance_tests<Kokkos::Threads, true>(base_file_name.str());
}

void unordered_map_performance_far() {
  std::cout << "Threads: unordered_map_performance_far" << std::endl;
  unsigned num_threads = 4;
  if (Kokkos::hwloc::available()) {
    num_threads = Kokkos::hwloc::get_available_numa_count() *
                  Kokkos::hwloc::get_available_cores_per_numa() *
                  Kokkos::hwloc::get_available_threads_per_core();
  }
  std::ostringstream base_file_name;
  base_file_name << "threads-" << num_threads << "-far";
  Perf::run_performance_tests<Kokkos::Threads, false>(base_file_name.str());
}

void scatter_view() {
  std::cout << "Threads: ScatterView data-duplicated test:\n";
  Perf::test_scatter_view<Kokkos::Threads, Kokkos::LayoutRight,
                          Kokkos::Experimental::ScatterDuplicated,
                          Kokkos::Experimental::ScatterNonAtomic>(10,
                                                                  1000 * 1000);
  // std::cout << "ScatterView atomics test:\n";
  // Perf::test_scatter_view<Kokkos::Threads, Kokkos::LayoutRight,
  //  Kokkos::Experimental::ScatterNonDuplicated,
  //  Kokkos::Experimental::ScatterAtomic>(10, 1000 * 1000);
}

}  // namespace Performance

int main() {
  Kokkos::ScopeGuard scope_guard;
  Performance::dynrankview_perf();
  Performance::global_2_local();
  Performance::unordered_map_performance_near();
  Performance::unordered_map_performance_far();
  Performance::scatter_view();
}
