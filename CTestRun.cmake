cmake_minimum_required(VERSION 3.20)

set(CTEST_BINARY_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)
set(CTEST_SOURCE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

set(CTEST_UPDATE_COMMAND git)
set(CTEST_UPDATE_VERSION_ONLY 1)
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")

set(CTEST_SUBMIT_URL https://my.cdash.org/submit.php?project=Kokkos)

ctest_start(Experimental)
ctest_update()
ctest_configure(OPTIONS "${CMAKE_OPTIONS}")
ctest_build()
ctest_test(OUTPUT_JUNIT "${OUTPUT_JUNIT_FILE}")
ctest_submit()
