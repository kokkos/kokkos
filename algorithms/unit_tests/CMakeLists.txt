
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
INCLUDE_DIRECTORIES(REQUIRED_DURING_INSTALLATION_TESTING ${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../src )

SET(GTEST_SOURCE_DIR ${${PARENT_PACKAGE_NAME}_SOURCE_DIR}/tpls/gtest)
INCLUDE_DIRECTORIES(${GTEST_SOURCE_DIR})

# mfh 03 Nov 2017: The gtest library used here must have a different
# name than that of the gtest library built in KokkosCore.  We can't
# just refer to the library in KokkosCore's tests, because it's
# possible to build only (e.g.,) KokkosAlgorithms tests, without
# building KokkosCore tests.

TRIBITS_ADD_LIBRARY(
  kokkosalgorithms_gtest
  HEADERS ${GTEST_SOURCE_DIR}/gtest/gtest.h
  SOURCES ${GTEST_SOURCE_DIR}/gtest/gtest-all.cc
  TESTONLY
  )

SET(SOURCES
  UnitTestMain.cpp 
  TestCuda.cpp
  )

SET(LIBRARIES kokkoscore)

IF(Kokkos_ENABLE_OpenMP)
  LIST( APPEND SOURCES
    TestOpenMP.cpp
  )
ENDIF()

IF(Kokkos_ENABLE_Serial)
  LIST( APPEND SOURCES
    TestSerial.cpp
  )
ENDIF()

IF(Kokkos_ENABLE_Pthread)
  LIST( APPEND SOURCES
    TestThreads.cpp
  )
ENDIF()

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  UnitTest
  SOURCES ${SOURCES}
  COMM serial mpi
  NUM_MPI_PROCS 1
  FAIL_REGULAR_EXPRESSION "  FAILED  "
  TESTONLYLIBS kokkosalgorithms_gtest
  )
