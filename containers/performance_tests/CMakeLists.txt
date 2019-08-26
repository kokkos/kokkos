
KOKKOS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
KOKKOS_INCLUDE_DIRECTORIES(REQUIRED_DURING_INSTALLATION_TESTING ${CMAKE_CURRENT_SOURCE_DIR})
KOKKOS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../src )

SET(SOURCES
  TestMain.cpp 
  TestCuda.cpp
  )

IF(Kokkos_ENABLE_PTHREAD)
  LIST( APPEND SOURCES TestThreads.cpp)
ENDIF()

IF(Kokkos_ENABLE_OPENMP)
  LIST( APPEND SOURCES TestOpenMP.cpp)
ENDIF()

IF(Kokkos_ENABLE_HPX)
  LIST( APPEND SOURCES TestHPX.cpp)
ENDIF()

# Per #374, we always want to build this test, but we only want to run
# it as a PERFORMANCE test.  That's why we separate building the test
# from running the test.

KOKKOS_ADD_TEST_EXECUTABLE(
  PerfTestExec
  SOURCES ${SOURCES}
)

KOKKOS_ADD_TEST(
  NAME PerformanceTest
  EXE  PerfTestExec
)

