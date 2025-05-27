#need these here for now
kokkos_include_directories(${CMAKE_CURRENT_BINARY_DIR})
kokkos_include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#-----------------------------------------------------------------------------

set(KOKKOS_CONTAINERS_SRCS)
append_glob(KOKKOS_CONTAINERS_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/impl/*.cpp)
set(KOKKOS_CONTAINER_HEADERS)
append_glob(KOKKOS_CONTAINERS_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/impl/*.hpp)
append_glob(KOKKOS_CONTAINERS_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
  DESTINATION ${KOKKOS_HEADER_DIR}
  FILES_MATCHING
  PATTERN "*.hpp"
)

if(Kokkos_ENABLE_EXPERIMENTAL_CXX20_MODULE)
  set(KOKKOS_CONTAINERS_MODULE_FILES Kokkos_DualView.ccm)
endif()

kokkos_add_library(
  kokkoscontainers
  SOURCES
  ${KOKKOS_CONTAINERS_SRCS}
  MODULE_FILES
  ${KOKKOS_CONTAINERS_MODULE_FILES}
  HEADERS
  ${KOKKOS_CONTAINERS_HEADERS}
)

if(Kokkos_ENABLE_EXPERIMENTAL_CXX20_MODULE)
  set_target_properties(kokkoscontainers PROPERTIES CXX_SCAN_FOR_MODULES ON)
endif()

kokkos_lib_include_directories(
  kokkoscontainers ${KOKKOS_TOP_BUILD_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}
)
kokkos_link_internal_library(kokkoscontainers kokkoscore)

#-----------------------------------------------------------------------------
