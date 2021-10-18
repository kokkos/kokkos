
#Leave these here for now - I don't need transitive deps anyway
KOKKOS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
KOKKOS_INCLUDE_DIRECTORIES(REQUIRED_DURING_INSTALLATION_TESTING ${CMAKE_CURRENT_SOURCE_DIR})
KOKKOS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../src )
KOKKOS_INCLUDE_DIRECTORIES(${KOKKOS_SOURCE_DIR}/core/unit_test/category_files)


SET(GTEST_SOURCE_DIR ${${PARENT_PACKAGE_NAME}_SOURCE_DIR}/tpls/gtest)
KOKKOS_INCLUDE_DIRECTORIES(${GTEST_SOURCE_DIR})

# mfh 03 Nov 2017: The gtest library used here must have a different
# name than that of the gtest library built in KokkosCore.  We can't
# just refer to the library in KokkosCore's tests, because it's
# possible to build only (e.g.,) KokkosAlgorithms tests, without
# building KokkosCore tests.


KOKKOS_ADD_TEST_LIBRARY(
  kokkosalgorithms_gtest
  HEADERS ${GTEST_SOURCE_DIR}/gtest/gtest.h
  SOURCES ${GTEST_SOURCE_DIR}/gtest/gtest-all.cc
)

# avoid deprecation warnings from MSVC
TARGET_COMPILE_DEFINITIONS(kokkosalgorithms_gtest PUBLIC GTEST_HAS_TR1_TUPLE=0 GTEST_HAS_PTHREAD=0)

IF((NOT (Kokkos_ENABLE_CUDA AND WIN32)) AND (NOT ("${KOKKOS_CXX_COMPILER_ID}" STREQUAL "Fujitsu")))
  TARGET_COMPILE_FEATURES(kokkosalgorithms_gtest PUBLIC cxx_std_14)
ENDIF()

# Suppress clang-tidy diagnostics on code that we do not have control over
IF(CMAKE_CXX_CLANG_TIDY)
  SET_TARGET_PROPERTIES(kokkosalgorithms_gtest PROPERTIES CXX_CLANG_TIDY "")
ENDIF()

SET(ALGORITHM UnitTestMain.cpp)

LIST(APPEND ALGORITHM_SOURCES
  TestStdAlgorithmsCommon.cpp

  #--- general ones ---
  TestStdReducers.cpp
  TestStdAlgorithmsConstraints.cpp
  TestRandomAccessIterator.cpp

  #--- min,max ---
  TestStdAlgorithmsMinMaxOps.cpp
  TestStdAlgorithmsMinMaxElementOps.cpp

  #--- non mod sequence ----
  TestStdAlgorithmsLexicographicalCompare.cpp
  TestStdAlgorithmsForEach.cpp
  TestStdAlgorithmsFind.cpp
  TestStdAlgorithmsFindFirstOf.cpp
  TestStdAlgorithmsFindEnd.cpp
  TestStdAlgorithmsCount.cpp
  TestStdAlgorithmsEqual.cpp
  TestStdAlgorithmsAllAnyNoneOf.cpp
  TestStdAlgorithmsAdjacentFind.cpp
  TestStdAlgorithmsSearch.cpp
  TestStdAlgorithmsSearch_n.cpp
  TestStdAlgorithmsMismatch.cpp

  #--- mod ----
  TestStdAlgorithmsModOps.cpp
  TestStdAlgorithmsModSeqOps.cpp
  TestStdAlgorithmsReplace.cpp
  TestStdAlgorithmsReplaceIf.cpp
  TestStdAlgorithmsReplaceCopy.cpp
  TestStdAlgorithmsReplaceCopyIf.cpp
  TestStdAlgorithmsCopyIf.cpp
  TestStdAlgorithmsUnique.cpp
  TestStdAlgorithmsUniqueCopy.cpp
  TestStdAlgorithmsRemove.cpp
  TestStdAlgorithmsRemoveIf.cpp
  TestStdAlgorithmsRemoveCopy.cpp
  TestStdAlgorithmsRemoveCopyIf.cpp
  TestStdAlgorithmsRotate.cpp
  TestStdAlgorithmsRotateCopy.cpp
  TestStdAlgorithmsReverse.cpp
  TestStdAlgorithmsShiftLeft.cpp
  TestStdAlgorithmsShiftRight.cpp

  #--- sorting ----
  TestStdAlgorithmsIsSorted.cpp
  TestStdAlgorithmsIsSortedUntil.cpp

  #--- partitioning ----
  TestStdAlgorithmsPartitioningOps.cpp
  TestStdAlgorithmsPartitionCopy.cpp

  #--- numerics ----
  TestStdAlgorithmsNumerics.cpp
  TestStdAlgorithmsAdjacentDifference.cpp
  TestStdAlgorithmsExclusiveScan.cpp
  TestStdAlgorithmsTransformExclusiveScan.cpp
  TestStdAlgorithmsInclusiveScan.cpp
  TestStdAlgorithmsTransformInclusiveScan.cpp
)

IF(Kokkos_ENABLE_OPENMP)
  LIST(APPEND ALGORITHM_SOURCES
    TestOpenMP_Sort1D.cpp
    TestOpenMP_Sort3D.cpp
    TestOpenMP_SortDynamicView.cpp
  )
ENDIF()

foreach(Tag Threads;Serial;OpenMP;Cuda;HPX;HIP;SYCL;OpenMPTarget)
  # Because there is always an exception to the rule
  if(Tag STREQUAL "Threads")
    set(DEVICE "PTHREAD")
  else()
    string(TOUPPER ${Tag} DEVICE)
  endif()

  if(Kokkos_ENABLE_${DEVICE})
    set(dir ${CMAKE_CURRENT_BINARY_DIR})
    set(file ${dir}/Test${Tag}.cpp)
      # Write to a temporary intermediate file and call configure_file to avoid
      # updating timestamps triggering unnecessary rebuilds on subsequent cmake runs.
      file(WRITE ${dir}/dummy.cpp
          "#include <Test${Tag}_Category.hpp>\n"
	  "#include <TestRandomCommon.hpp>\n"
	  "#include <TestSortCommon.hpp>\n"
      )
      configure_file(${dir}/dummy.cpp ${file})
      list(APPEND ALGORITHM_SOURCES ${file})
  endif()
endforeach()

KOKKOS_ADD_EXECUTABLE_AND_TEST(
  UnitTest
  SOURCES
    UnitTestMain.cpp
    ${ALGORITHM_SOURCES}
)
