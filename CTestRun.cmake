cmake_minimum_required(VERSION 3.20)

set(CTEST_BINARY_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set(CTEST_SOURCE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

set(CTEST_SUBMIT_URL https://my.cdash.org/submit.php?project=Kokkos)

ctest_start(Experimental)
ctest_test(OUTPUT_JUNIT "${OUTPUT_JUNIT_FILE}")
ctest_submit()
