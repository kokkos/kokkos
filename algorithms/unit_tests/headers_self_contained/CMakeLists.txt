# Create tests that contain each header separately. We do not  run these tests
# but we just try to compile them.

# Globbing all the header filenames to test for self-containment and presence of header guards
SET(BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../")
file(GLOB KOKKOS_ALGORITHMS_HEADERS RELATIVE  ${BASE_DIR}/algorithms/src
     ${BASE_DIR}/algorithms/src/*.hpp ${BASE_DIR}/algorithms/src/std_algorithms/*.hpp)

foreach (_header ${KOKKOS_ALGORITHMS_HEADERS})
  string(REGEX REPLACE "[\./]" "_" header_test_name ${_header})
  set(header_test_name Kokkos_HeaderSelfContained_${header_test_name})
  set_source_files_properties(tstHeader.cpp PROPERTIES LANGUAGE ${KOKKOS_COMPILE_LANGUAGE})
  add_executable(${header_test_name} tstHeader.cpp)
  target_link_libraries(${header_test_name} PRIVATE Kokkos::kokkos)
  target_compile_definitions(${header_test_name} PRIVATE KOKKOS_HEADER_TEST_NAME=${_header})
endforeach()
