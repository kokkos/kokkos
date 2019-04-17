

#I have to leave these here for tribits
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

INSTALL (DIRECTORY
  "${CMAKE_CURRENT_SOURCE_DIR}/"
  DESTINATION ${KOKKOS_HEADER_DIR}
  FILES_MATCHING PATTERN "*.hpp"
)

KOKKOS_ADD_LIBRARY(
  kokkoscore
  SOURCES ${KOKKOS_CORE_SRCS}
)
#include source and binary dirs
KOKKOS_LIB_DEFAULT_INCLUDE_DIRECTORIES(kokkoscore)

# build and install pkgconfig file
#CONFIGURE_FILE(kokkos.pc.in kokkos.pc @ONLY)
#INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/kokkos.pc DESTINATION lib/pkgconfig)

