
#INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
#INCLUDE_DIRECTORIES(REQUIRED_DURING_INSTALLATION_TESTING ${CMAKE_CURRENT_SOURCE_DIR})


# warning: PerfTest_CustomReduction.cpp uses
# ../../algorithms/src/Kokkos_Random.hpp
# we'll just allow it to be included, but note
# that in TriBITS KokkosAlgorithms can be disabled...
#INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/../../algorithms/src")

SET(SOURCES
  PerfTestMain.cpp
  PerfTestGramSchmidt.cpp
  PerfTestHexGrad.cpp
  PerfTest_CustomReduction.cpp
  )

# Per #374, we always want to build this test, but we only want to run
# it as a PERFORMANCE test.  That's why we separate building the test
# from running the test.

#leave these as basic includes for now
#I don't need anything transitive
KOKKOS_INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/../../algorithms/src")
KOKKOS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
KOKKOS_INCLUDE_DIRECTORIES(REQUIRED_DURING_INSTALLATION_TESTING ${CMAKE_CURRENT_SOURCE_DIR})

KOKKOS_ADD_EXECUTABLE_AND_TEST(
  PerfTestExec
  SOURCES ${SOURCES}
  CATEGORIES PERFORMANCE
)

KOKKOS_ADD_EXECUTABLE_AND_TEST(
  PerformanceTest_TaskDag
  SOURCES test_taskdag.cpp
  CATEGORIES PERFORMANCE
)
