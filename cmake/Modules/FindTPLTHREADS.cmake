# Copyright Contributors to the Kokkos project
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(FindPackageHandleStandardArgs)
find_package(Threads)

if(TARGET Threads::Threads)
  set(FOUND_THREADS TRUE)
else()
  set(FOUND_THREADS FALSE)
endif()

find_package_handle_standard_args(TPLTHREADS DEFAULT_MSG FOUND_THREADS)
#Only create the TPL if we succeed
if(FOUND_THREADS)
  kokkos_create_imported_tpl(THREADS INTERFACE LINK_OPTIONS ${CMAKE_THREAD_LIBS_INIT})
endif()
