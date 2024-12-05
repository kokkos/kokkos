cmake_minimum_required(VERSION 3.20)

set(CTEST_BINARY_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set(CTEST_SOURCE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

set(CTEST_SUBMIT_URL https://my.cdash.org/submit.php?project=Kokkos)
set(CTEST_NOTES_FILES "${CTEST_BINARY_DIRECTORY}/generated/Kokkos_Version_Info.cpp")
message(STATUS "CTEST_NOTES_FILES: ${CTEST_NOTES_FILES}")

ctest_start(Experimental)
ctest_test(OUTPUT_JUNIT "${OUTPUT_JUNIT_FILE}")
ctest_submit()
