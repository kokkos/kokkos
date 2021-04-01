

KOKKOS_SUBPACKAGE(Core)

IF (NOT Kokkos_INSTALL_TESTING)
  ADD_SUBDIRECTORY(src)
ENDIF()

KOKKOS_ADD_TEST_DIRECTORIES(unit_test)
KOKKOS_ADD_TEST_DIRECTORIES(perf_test)

KOKKOS_SUBPACKAGE_POSTPROCESS()

IF(Kokkos_ENABLE_TESTS)
  ADD_TEST(
    NAME                compile-time-perf
    COMMAND             ${CMAKE_COMMAND}
                        --build ${PROJECT_BINARY_DIR}
                        --target analyze-kokkos-compile-time
    WORKING_DIRECTORY   ${PROJECT_BINARY_DIR})
ENDIF()
