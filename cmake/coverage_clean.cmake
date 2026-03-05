# Clean previous LLVM coverage data from the build directory.
# Invoked by kokkos_coverage target with -DCMAKE_BINARY_DIR=<build_dir>.
if(NOT CMAKE_BINARY_DIR)
  message(FATAL_ERROR "coverage_clean.cmake requires -DCMAKE_BINARY_DIR=<path>")
endif()
foreach(_f default.profraw default.profdata)
  set(_path "${CMAKE_BINARY_DIR}/${_f}")
  if(EXISTS "${_path}")
    file(REMOVE "${_path}")
    message(STATUS "Removed ${_path}")
  endif()
endforeach()
