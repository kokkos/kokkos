# Copyright Contributors to the Kokkos project
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

kokkos_add_benchmark(PerformanceTest_InclusiveScan SOURCES test_inclusive_scan.cpp)
kokkos_add_benchmark(PerformanceTest_Random SOURCES test_random.cpp)
kokkos_add_benchmark(PerformanceTest_SortByKey SOURCES test_sort_by_key.cpp)
