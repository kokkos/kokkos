# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

if(NOT Kokkos_INSTALL_TESTING)
  add_subdirectory(src)
endif()

kokkos_add_test_directories(unit_test)
kokkos_add_benchmark_directories(perf_test)
