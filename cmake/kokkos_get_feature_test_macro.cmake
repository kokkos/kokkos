function(kokkos_get_feature_test_macro FEATURE_TEST_MACRO RESULT)
  set(TEST_MACRO_NAME ${FEATURE_TEST_MACRO})
  message(STATUS "${CMAKE_CURRENT_FUNCTION_LIST_DIR}")
  configure_file("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/compile_tests/get_cpp_version_macro.cpp.in"
                 get_cpp_version_macro.cpp
                 @ONLY
  )
  try_run(TEST_RUN_RES TEST_COMPILE_RES
          ${CMAKE_CURRENT_BINARY_DIR}
          SOURCES ${CMAKE_CURRENT_BINARY_DIR}/get_cpp_version_macro.cpp
          COMPILE_OUTPUT_VARIABLE COMPILE_LOG
          RUN_OUTPUT_VARIABLE RET
          )
  if (NOT TEST_COMPILE_RES)
    message(FATAL_ERROR "Could not test feature test macro \"${FEATURE_TEST_MACRO}\" due to a compile error: ${COMPILE_LOG}")
  endif()
  if (NOT TEST_RUN_RES EQUAL 0)
    message(FATAL_ERROR "Could not test feature test macro \"${FEATURE_TEST_MACRO}\" due to a runtime error: ${RET}.")
  endif()
  message(STATUS "got ${RET}")
  set(${RESULT} ${RET} PARENT_SCOPE)
endfunction()