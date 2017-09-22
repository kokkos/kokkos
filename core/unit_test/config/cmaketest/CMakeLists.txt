cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(Kokkos CXX)

foreach(KOKKOS_HOST_ARCH ${KOKKOS_HOST_ARCH_LIST})
  foreach(KOKKOS_DEVICE ${KOKKOS_DEVICES_LIST})

    message(STATUS "${KOKKOS_HOST_ARCH} ${KOKKOS_DEVICE}")

    # Handle the options
    include(${CMAKE_SOURCE_DIR}/cmake/kokkos_options.cmake)

    #---------------------------------------------------------------------
    # File names must be consistent with core/src/Makefile
    # This generates the ${CMAKE_BINARY_DIR}/gen_kokkos.cmake file
    #    execute_process(
    #  COMMAND make -f ${CMAKE_SOURCE_DIR}/core/src/Makefile ${KOKKOS_SETTINGS} build-makefile-cmake-kokkos
    #  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    #  OUTPUT_FILE ${CMAKE_BINARY_DIR}/core_src_make.out
    #  RESULT_VARIABLE res
    #)

  endforeach()
endforeach()
