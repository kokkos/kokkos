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

kokkos_add_library(kokkoscontainers SOURCES ${KOKKOS_CONTAINERS_SRCS} HEADERS ${KOKKOS_CONTAINERS_HEADERS})

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN TRUE)

include(GenerateExportHeader)
generate_export_header(kokkoscontainers EXPORT_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/Kokkos_Containers_Export.h")

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/Kokkos_Containers_Export.h" DESTINATION ${KOKKOS_HEADER_DIR})

kokkos_lib_include_directories(
  kokkoscontainers ${KOKKOS_TOP_BUILD_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}
)
kokkos_link_internal_library(kokkoscontainers kokkoscore)

#-----------------------------------------------------------------------------
