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
set(KOKKOS_HEADERS_TO_CONVERT
	 Kokkos_Bitset    Kokkos_DynRankView  Kokkos_ErrorReporter  Kokkos_OffsetView
Kokkos_DualView  Kokkos_DynamicView  Kokkos_Functional     Kokkos_ScatterView  Kokkos_UnorderedMap
	)

set(KOKKOS_CONTAINERS_MODULE_FILES)
foreach(file ${KOKKOS_HEADERS_TO_CONVERT})
  file(READ "${file}.hpp" FILE_CONTENTS)
  string(PREPEND FILE_CONTENTS "module;\n")

  string(REPLACE "_" "" MODULE_NAME ${file})
  string(TOLOWER ${MODULE_NAME} MODULE_NAME_LC) 

  string(FIND "${FILE_CONTENTS}" "namespace Kokkos {" find_result)
  string(SUBSTRING "${FILE_CONTENTS}" 0 "${find_result}" before_substring)
  string(LENGTH "${substring_to_replace}" substring_len)
  math(EXPR after_start "${find_result} + ${substring_len}")
  string(SUBSTRING "${FILE_CONTENTS}" "${after_start}" -1 after_substring)
  set(FILE_CONTENTS_EXPORT_MODULE "${before_substring}export module ${MODULE_NAME_LC};\n${after_substring}")

  string(REPLACE "namespace Kokkos {" "export {\nnamespace Kokkos {" FILE_CONTENTS_EXPORT "${FILE_CONTENTS_EXPORT_MODULE}")
  string(REGEX REPLACE "} [^\n]* namespace Kokkos[^\n]*\n" "} // namespace Kokkos\n}" FILE_CONTENTS_MODULE "${FILE_CONTENTS_EXPORT}")

  #file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${file}.ccm" "${FILE_CONTENTS_MODULE}")
  file(WRITE "${file}.ccm" "${FILE_CONTENTS_MODULE}")
  #list(APPEND KOKKOS_CONTAINERS_MODULE_FILES "${CMAKE_CURRENT_BINARY_DIR}/${file}.ccm") 
  list(APPEND KOKKOS_CONTAINERS_MODULE_FILES "${file}.ccm")
endforeach()
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
