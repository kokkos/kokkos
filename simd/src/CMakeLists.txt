
KOKKOS_CONFIGURE_FILE(${PACKAGE_NAME}_config.h)

#I have to leave these here for tribits
KOKKOS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
KOKKOS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

#-----------------------------------------------------------------------------

FILE(GLOB SIMD_HEADERS *.hpp)
FILE(GLOB SIMD_SOURCES *.cpp)
LIST(APPEND SIMD_HEADERS ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}_config.h)

INSTALL (
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
  DESTINATION ${KOKKOS_HEADER_DIR}
  FILES_MATCHING PATTERN "*.hpp"
)

#-----------------------------------------------------------------------------

# We have to pass the sources in here for Tribits
# These will get ignored for standalone CMake and a true interface library made
KOKKOS_ADD_INTERFACE_LIBRARY(
  kokkossimd
  HEADERS ${SIMD_HEADERS}
  SOURCES ${SIMD_SOURCES}
)
KOKKOS_LIB_INCLUDE_DIRECTORIES(kokkossimd
  ${KOKKOS_TOP_BUILD_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
)
