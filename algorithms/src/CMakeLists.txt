#I have to leave these here for tribits
kokkos_include_directories(${CMAKE_CURRENT_BINARY_DIR})
kokkos_include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#-----------------------------------------------------------------------------

file(GLOB ALGO_HEADERS *.hpp)
file(GLOB ALGO_SOURCES *.cpp)
append_glob(ALGO_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/std_algorithms/*.hpp)
append_glob(ALGO_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/std_algorithms/impl/*.hpp)

set(KOKKOS_ALGORITHMS_MODULE_FILES)
file(GLOB MODULE_HEADER_SOURCES sorting/*.hpp)
foreach(file ${MODULE_HEADER_SOURCES})
    get_filename_component(FILENAME ${file} NAME_WLE)
    string(REPLACE ".hpp" ".ccm" MODULE_FILE_NAME "${file}")

    file(READ "${file}" FILE_CONTENTS)
    string(PREPEND FILE_CONTENTS "module;\n")

    string(REPLACE "_" "" MODULE_NAME ${FILENAME})
    string(TOLOWER ${MODULE_NAME} MODULE_NAME_LC)

    string(FIND "${FILE_CONTENTS}" "namespace Kokkos" find_result)
    string(SUBSTRING "${FILE_CONTENTS}" 0 "${find_result}" before_substring)
    string(LENGTH "${substring_to_replace}" substring_len)
    math(EXPR after_start "${find_result} + ${substring_len}")
    message(STATUS "${MODULE_NAME_LC}")
    string(SUBSTRING "${FILE_CONTENTS}" "${after_start}" -1 after_substring)
    set(FILE_CONTENTS_EXPORT_MODULE "${before_substring}export module ${MODULE_NAME_LC};\n${after_substring}")

    string(REGEX REPLACE "namespace Kokkos([^{]*){" "export {\nnamespace Kokkos\\1{" FILE_CONTENTS_EXPORT
                   "${FILE_CONTENTS_EXPORT_MODULE}"
    )
    string(REGEX REPLACE "} [^\n]* namespace Kokkos([^\n]*)\n" "}  // namespace Kokkos\\1\n}  // export\n" FILE_CONTENTS_MODULE
                         "${FILE_CONTENTS_EXPORT}"
    )

    file(WRITE "${MODULE_FILE_NAME}" "${FILE_CONTENTS_MODULE}")
    list(APPEND KOKKOS_CONTAINERS_MODULE_FILES "${MODULE_FILE_NAME}")
  endforeach()

install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
  DESTINATION ${KOKKOS_HEADER_DIR}
  FILES_MATCHING
  PATTERN "*.hpp"
)

#-----------------------------------------------------------------------------

# We have to pass the sources in here for Tribits
# These will get ignored for standalone CMake and a true interface library made
kokkos_add_interface_library(kokkosalgorithms NOINSTALLHEADERS ${ALGO_HEADERS}  MODULE_FILES
	${KOKKOS_ALGORITHMS_MODULE_FILES} SOURCES ${ALGO_SOURCES})
kokkos_lib_include_directories(
  kokkosalgorithms ${KOKKOS_TOP_BUILD_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}
)

kokkos_link_tpl(kokkoscontainers PUBLIC ROCTHRUST)
kokkos_link_tpl(kokkoscore PUBLIC ONEDPL)
