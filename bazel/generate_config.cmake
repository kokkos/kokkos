# SPDX-FileCopyrightText: Copyright 2026 Antmicro
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.22)

set(supported_backends serial threads openmp)
if(NOT BACKEND IN_LIST supported_backends)
  message(FATAL_ERROR "BACKEND must be serial, threads, or openmp")
endif()
if(NOT DEFINED LIBDL OR NOT DEFINED OUTPUT)
  message(FATAL_ERROR "LIBDL and OUTPUT must be specified")
endif()

set(source_dir "${CMAKE_CURRENT_LIST_DIR}/..")

include("${source_dir}/cmake/kokkos_version.cmake")

foreach(dependency desul mdspan)
  file(READ "${source_dir}/tpls/${dependency}-hash.txt" revision)
  string(STRIP "${revision}" revision)
  string(TOUPPER "${dependency}" name)
  set(KOKKOS_${name}_VERSION "${revision}")
endforeach()

# Bazel owns these settings explicitly. Other features remain undefined.
set(KOKKOS_ENABLE_CXX20 ON)
set(KOKKOS_ENABLE_SERIAL ON)
string(TOUPPER "${BACKEND}" backend_name)
set(KOKKOS_ENABLE_${backend_name} ON)
set(KOKKOS_ENABLE_LIBDL "${LIBDL}")
set(KOKKOS_ENABLE_DEPRECATED_CODE_5 ON)
set(KOKKOS_ENABLE_DEPRECATION_WARNINGS ON)
set(KOKKOS_ENABLE_COMPLEX_ALIGN ON)
set(KOKKOS_ENABLE_IMPL_REF_COUNT_BRANCH_UNLIKELY ON)

configure_file("${source_dir}/cmake/KokkosCore_config.h.in" "${OUTPUT}" @ONLY)
