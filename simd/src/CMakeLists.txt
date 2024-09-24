#I have to leave these here for tribits
KOKKOS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
KOKKOS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

#-----------------------------------------------------------------------------

FILE(GLOB SIMD_HEADERS *.hpp)
FILE(GLOB SIMD_SOURCES *.cpp)

INSTALL (
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
  DESTINATION ${KOKKOS_HEADER_DIR}
  FILES_MATCHING PATTERN "*.hpp"
)

#-----------------------------------------------------------------------------

# We have to pass the sources in here for Tribits
# These will get ignored for standalone CMake and a true interface library made
KOKKOS_ADD_LIBRARY(
  kokkossimd
  SOURCES ${SIMD_SOURCES}
  HEADERS ${SIMD_HEADERS}
)

IF(KOKKOS_CXX_COMPILER_ID STREQUAL Intel OR KOKKOS_CXX_COMPILER_ID STREQUAL IntelLLVM)
  FIND_LIBRARY(INTEL_SVML NAMES svml)
  IF(INTEL_SVML)
    MESSAGE(STATUS "Found INTEL_SVML")
    TARGET_LINK_LIBRARIES(kokkossimd PUBLIC ${INTEL_SVML})
    TARGET_COMPILE_DEFINITIONS(kokkossimd PUBLIC KOKKOS_HAVE_INTEL_SVML)
  ENDIF()
ENDIF()

KOKKOS_LIB_INCLUDE_DIRECTORIES(kokkossimd
  ${KOKKOS_TOP_BUILD_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
)
