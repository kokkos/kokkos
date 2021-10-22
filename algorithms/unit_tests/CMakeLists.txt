
#Leave these here for now - I don't need transitive deps anyway
KOKKOS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
KOKKOS_INCLUDE_DIRECTORIES(REQUIRED_DURING_INSTALLATION_TESTING ${CMAKE_CURRENT_SOURCE_DIR})
KOKKOS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../src )
KOKKOS_INCLUDE_DIRECTORIES(${KOKKOS_SOURCE_DIR}/core/unit_test/category_files)

enable_testing()
find_package(GTest REQUIRED)

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

foreach(Tag Threads;Serial;OpenMP;Cuda;HPX;HIP;SYCL;OpenMPTarget)
  # Because there is always an exception to the rule
  if(Tag STREQUAL "Threads")
    set(DEVICE "PTHREAD")
  else()
    string(TOUPPER ${Tag} DEVICE)
  endif()
  string(TOLOWER ${Tag} dir)

  if(Kokkos_ENABLE_${DEVICE})
    set(dir ${CMAKE_CURRENT_BINARY_DIR}/${dir})
    file(MAKE_DIRECTORY ${dir})

    # -------------------------
    # Sort1d,3d, Random
    # -------------------------
    set(SOURCES_A)
    if(Tag STREQUAL "OpenMP")
      LIST(APPEND SOURCES_A
	TestOpenMP_Sort1D.cpp
	TestOpenMP_Sort3D.cpp
	TestOpenMP_SortDynamicView.cpp
	)
    endif()

    set(file ${dir}/TestRandomAndSort.cpp)
    # Write to a temporary intermediate file and call configure_file to avoid
    # updating timestamps triggering unnecessary rebuilds on subsequent cmake runs.
    file(WRITE ${dir}/dummy.cpp
      "#include <Test${Tag}_Category.hpp>\n"
      "#include <TestRandomCommon.hpp>\n"
      "#include <TestSortCommon.hpp>\n"
      )
    configure_file(${dir}/dummy.cpp ${file})
    list(APPEND SOURCES_A ${file})

    # ------------------------------------------
    # std set A
    # ------------------------------------------
    set(STDALGO_SOURCES_A)
    foreach(Name
	StdReducers
	StdAlgorithmsConstraints
	RandomAccessIterator
	)
      list(APPEND STDALGO_SOURCES_A Test${Name}.cpp)
    endforeach()

    # ------------------------------------------
    # std set B
    # ------------------------------------------
    set(STDALGO_SOURCES_B)
    foreach(Name
	StdAlgorithmsCommon
	StdAlgorithmsMinMaxElementOps
	)
      list(APPEND STDALGO_SOURCES_B Test${Name}.cpp)
    endforeach()

    # ------------------------------------------
    # std set C
    # ------------------------------------------
    set(STDALGO_SOURCES_C)
    foreach(Name
	StdAlgorithmsCommon
	StdAlgorithmsLexicographicalCompare
	StdAlgorithmsForEach
	StdAlgorithmsFind
	StdAlgorithmsFindFirstOf
	StdAlgorithmsFindEnd
	StdAlgorithmsCount
	StdAlgorithmsEqual
	StdAlgorithmsAllAnyNoneOf
	StdAlgorithmsAdjacentFind
	StdAlgorithmsSearch
	StdAlgorithmsSearch_n
	StdAlgorithmsMismatch
	)
      list(APPEND STDALGO_SOURCES_C Test${Name}.cpp)
    endforeach()

    # ------------------------------------------
    # std set D
    # ------------------------------------------
    set(STDALGO_SOURCES_D)
    foreach(Name
	StdAlgorithmsCommon
	StdAlgorithmsModOps
	StdAlgorithmsModSeqOps
	StdAlgorithmsReplace
	StdAlgorithmsReplaceIf
	StdAlgorithmsReplaceCopy
	StdAlgorithmsReplaceCopyIf
	StdAlgorithmsCopyIf
	StdAlgorithmsUnique
	StdAlgorithmsUniqueCopy
	StdAlgorithmsRemove
	StdAlgorithmsRemoveIf
	StdAlgorithmsRemoveCopy
	StdAlgorithmsRemoveCopyIf
	StdAlgorithmsRotate
	StdAlgorithmsRotateCopy
	StdAlgorithmsReverse
	StdAlgorithmsShiftLeft
	StdAlgorithmsShiftRight
	)
      list(APPEND STDALGO_SOURCES_D Test${Name}.cpp)
    endforeach()

    # ------------------------------------------
    # std set E
    # ------------------------------------------
    set(STDALGO_SOURCES_E)
    foreach(Name
	StdAlgorithmsCommon
	StdAlgorithmsIsSorted
	StdAlgorithmsIsSortedUntil
	StdAlgorithmsPartitioningOps
	StdAlgorithmsPartitionCopy
	StdAlgorithmsNumerics
	StdAlgorithmsAdjacentDifference
	StdAlgorithmsExclusiveScan
	StdAlgorithmsTransformExclusiveScan
	StdAlgorithmsInclusiveScan
	StdAlgorithmsTransformInclusiveScan
	)
      list(APPEND STDALGO_SOURCES_E Test${Name}.cpp)
    endforeach()

  endif()
endforeach()

KOKKOS_ADD_EXECUTABLE_AND_TEST(
  UnitTest_RandomAndSort
  SOURCES
    UnitTestMain.cpp
    ${SOURCES_A}
)

foreach(ID A;B;C;D;E)
  KOKKOS_ADD_EXECUTABLE_AND_TEST(
    UnitTest_StdSet_${ID}
    SOURCES
    UnitTestMain.cpp
    ${STDALGO_SOURCES_${ID}}
    )
endforeach()
