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
  std::cout << "OpenMP: dynrankview_perf" << std::endl;
  std::cout << " DynRankView vs View: Initialization Only " << std::endl;
  test_dynrankview_op_perf<Kokkos::OpenMP>(8192);
}

void global_2_local() {
  std::cout << "OpenMP: global_2_local" << std::endl;
  std::cout << "size, create, generate, fill, find" << std::endl;
  for (unsigned i = Performance::begin_id_size; i <= Performance::end_id_size;
       i *= Performance::id_step)
    test_global_to_local_ids<Kokkos::OpenMP>(i);
}

void unordered_map_performance_near() {
  std::cout << "OpenMP: unordered_map_performance_near" << std::endl;
  unsigned num_openmp = 4;
  if (Kokkos::hwloc::available()) {
    num_openmp = Kokkos::hwloc::get_available_numa_count() *
                 Kokkos::hwloc::get_available_cores_per_numa() *
                 Kokkos::hwloc::get_available_threads_per_core();
  }
  std::ostringstream base_file_name;
  base_file_name << "openmp-" << num_openmp << "-near";
  Perf::run_performance_tests<Kokkos::OpenMP, true>(base_file_name.str());
}

void unordered_map_performance_far() {
  std::cout << "OpenMP: unordered_map_performance_far" << std::endl;
  unsigned num_openmp = 4;
  if (Kokkos::hwloc::available()) {
    num_openmp = Kokkos::hwloc::get_available_numa_count() *
                 Kokkos::hwloc::get_available_cores_per_numa() *
                 Kokkos::hwloc::get_available_threads_per_core();
  }
  std::ostringstream base_file_name;
  base_file_name << "openmp-" << num_openmp << "-far";
  Perf::run_performance_tests<Kokkos::OpenMP, false>(base_file_name.str());
}

void scatter_view() {
  std::cout << "OpenMP: ScatterView data-duplicated test:\n";
  Perf::test_scatter_view<Kokkos::OpenMP, Kokkos::LayoutRight,
                          Kokkos::Experimental::ScatterDuplicated,
                          Kokkos::Experimental::ScatterNonAtomic>(10,
                                                                  1000 * 1000);
  // std::cout << "ScatterView atomics test:\n";
  // Perf::test_scatter_view<Kokkos::OpenMP, Kokkos::LayoutRight,
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
