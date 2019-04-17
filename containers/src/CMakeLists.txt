
KOKKOS_CONFIGURE_FILE(${PACKAGE_NAME}_config.h)

#need these here for now
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

#-----------------------------------------------------------------------------


INSTALL (
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
  DESTINATION ${KOKKOS_HEADER_DIR}
  FILES_MATCHING PATTERN "*.hpp"
)

KOKKOS_ADD_LIBRARY(
  kokkoscontainers
  SOURCES ${KOKKOS_CONTAINERS_SRCS}
)
KOKKOS_LIB_DEFAULT_INCLUDE_DIRECTORIES(kokkoscontainers)
KOKKOS_LINK_INTERNAL_LIBRARY(kokkoscontainers kokkoscore)

#-----------------------------------------------------------------------------

