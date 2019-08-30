
KOKKOS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
KOKKOS_INCLUDE_DIRECTORIES(REQUIRED_DURING_INSTALLATION_TESTING ${CMAKE_CURRENT_SOURCE_DIR})
KOKKOS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../src )

foreach(Tag Threads;Serial;OpenMP;HPX;Cuda)
  # Because there is always an exception to the rule
  if(Tag STREQUAL "Threads")
    set(ARCH "PTHREAD")
  else()
    string(TOUPPER ${Tag} ARCH)
  endif()
  # Add test for that backend if it is enabled
  if(Kokkos_ENABLE_${ARCH})
    KOKKOS_ADD_EXECUTABLE_AND_TEST(
      UnitTest_${Tag}
      SOURCES
        UnitTestMain.cpp
        ${Tag}/Test${Tag}_BitSet.cpp
        ${Tag}/Test${Tag}_DualView.cpp
        ${Tag}/Test${Tag}_DynamicView.cpp
        ${Tag}/Test${Tag}_DynRankViewAPI_generic.cpp
        ${Tag}/Test${Tag}_DynRankViewAPI_rank12345.cpp
        ${Tag}/Test${Tag}_DynRankViewAPI_rank67.cpp
        ${Tag}/Test${Tag}_ErrorReporter.cpp
        ${Tag}/Test${Tag}_OffsetView.cpp
        ${Tag}/Test${Tag}_ScatterView.cpp
        ${Tag}/Test${Tag}_StaticCrsGraph.cpp
        ${Tag}/Test${Tag}_UnorderedMap.cpp
        ${Tag}/Test${Tag}_Vector.cpp
        ${Tag}/Test${Tag}_ViewCtorPropEmbeddedDim.cpp
      )
  endif()
endforeach()
