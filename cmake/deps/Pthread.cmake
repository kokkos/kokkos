#@HEADER
# ************************************************************************
#
#                        Kokkos v. 4.0
#       Copyright (2022) National Technology & Engineering
#               Solutions of Sandia, LLC (NTESS).
#
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
#
# Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ************************************************************************
# @HEADER

set(USE_THREADS FALSE)

if(NOT TPL_Pthread_INCLUDE_DIRS AND NOT TPL_Pthread_LIBRARY_DIRS AND NOT TPL_Pthread_LIBRARIES)
  # Use CMake's Thread finder since it is a bit smarter in determining
  # whether pthreads is already built into the compiler and doesn't need
  # a library to link.
  find_package(Threads)
  #If Threads found a copy of pthreads make sure it is one of the cases the tribits
  #tpl system cannot handle.
  if(Threads_FOUND AND CMAKE_USE_PTHREADS_INIT)
    if(CMAKE_THREAD_LIBS_INIT STREQUAL "" OR CMAKE_THREAD_LIBS_INIT STREQUAL "-pthread")
      set(USE_THREADS TRUE)
    endif()
  endif()
endif()

if(USE_THREADS)
  set(TPL_Pthread_INCLUDE_DIRS "")
  set(TPL_Pthread_LIBRARIES "${CMAKE_THREAD_LIBS_INIT}")
  set(TPL_Pthread_LIBRARY_DIRS "")
  kokkos_create_imported_tpl_library(Pthread)
else()
  kokkos_tpl_find_include_dirs_and_libraries(Pthread REQUIRED_HEADERS pthread.h REQUIRED_LIBS_NAMES pthread)
endif()
