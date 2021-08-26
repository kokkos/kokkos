cmake_minimum_required(VERSION 3.16)
project(SpackTestGen)
set(TEST_LIST_DEF ${CMAKE_CURRENT_SOURCE_DIR}/test_list.def)
file(STRINGS ${TEST_LIST_DEF} TEST_FILES)

#Copy test source to Spack test directory
foreach (TEST_FILE ${TEST_FILES})
  set(TEST_FILE_LOCATION ${SPACK_PACKAGE_SOURCE_DIR}/${TEST_FILE})
  file(COPY ${TEST_FILE_LOCATION} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/out)
endforeach()

#Clean up names
foreach(TEST_FILE ${TEST_FILES} )
  string( REGEX REPLACE ".+\/" "" TEST_FILE ${TEST_FILE} )
  list(APPEND SRC_NAME_LIST ${TEST_FILE})
  string( REPLACE ".cpp" "" TEST_FILE ${TEST_FILE} )
  list(APPEND BIN_NAME_LIST ${TEST_FILE})
endforeach()

#Configure test cmake script and run script
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt.in ${CMAKE_CURRENT_SOURCE_DIR}/out/CMakeLists.txt @ONLY)
