#These are tribits wrappers only ever called by Kokkos itself

INCLUDE(CMakeParseArguments)
INCLUDE(CTest)

MESSAGE(STATUS "The project name is: ${PROJECT_NAME}")

#Leave this here for now - but only do for tribits
#This breaks the standalone CMake
IF (KOKKOS_HAS_TRILINOS)
  IF(NOT DEFINED ${PROJECT_NAME}_ENABLE_OpenMP)
    SET(${PROJECT_NAME}_ENABLE_OpenMP OFF)
  ENDIF()

  IF(NOT DEFINED ${PROJECT_NAME}_ENABLE_HPX)
    SET(${PROJECT_NAME}_ENABLE_HPX OFF)
  ENDIF()

  IF(NOT DEFINED ${PROJECT_NAME}_ENABLE_DEBUG)
    SET(${PROJECT_NAME}_ENABLE_DEBUG OFF)
  ENDIF()

  IF(NOT DEFINED ${PROJECT_NAME}_ENABLE_CXX11)
    SET(${PROJECT_NAME}_ENABLE_CXX11 ON)
  ENDIF()

  IF(NOT DEFINED ${PROJECT_NAME}_ENABLE_TESTS)
    SET(${PROJECT_NAME}_ENABLE_TESTS OFF)
  ENDIF()

  IF(NOT DEFINED TPL_ENABLE_Pthread)
    SET(TPL_ENABLE_Pthread OFF)
  ENDIF()
ENDIF()

MACRO(KOKKOS_SUBPACKAGE NAME)
  if (KOKKOS_HAS_TRILINOS)
    TRIBITS_SUBPACKAGE(${NAME})
  else()
    SET(PACKAGE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
    SET(PARENT_PACKAGE_NAME ${PACKAGE_NAME})
    SET(PACKAGE_NAME ${PACKAGE_NAME}${NAME})
    STRING(TOUPPER ${PACKAGE_NAME} PACKAGE_NAME_UC)
    SET(${PACKAGE_NAME}_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
    #ADD_INTERFACE_LIBRARY(PACKAGE_${PACKAGE_NAME})
    #GLOBAL_SET(${PACKAGE_NAME}_LIBS "")
  endif()
ENDMACRO()

MACRO(KOKKOS_SUBPACKAGE_POSTPROCESS)
  if (KOKKOS_HAS_TRILINOS)
    TRIBITS_SUBPACKAGE_POSTPROCESS()
  endif()
ENDMACRO()

MACRO(KOKKOS_PACKAGE_DECL)

  if (KOKKOS_HAS_TRILINOS)
    TRIBITS_PACKAGE_DECL(Kokkos)
  else()
    SET(PACKAGE_NAME Kokkos)
    SET(${PACKAGE_NAME}_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
    STRING(TOUPPER ${PACKAGE_NAME} PACKAGE_NAME_UC)
  endif()

  #SET(TRIBITS_DEPS_DIR "${CMAKE_SOURCE_DIR}/cmake/deps")
  #FILE(GLOB TPLS_FILES "${TRIBITS_DEPS_DIR}/*.cmake")
  #FOREACH(TPL_FILE ${TPLS_FILES})
  #  TRIBITS_PROCESS_TPL_DEP_FILE(${TPL_FILE})
  #ENDFOREACH()

ENDMACRO()


MACRO(KOKKOS_PROCESS_SUBPACKAGES)
  if (KOKKOS_HAS_TRILINOS)
    TRIBITS_PROCESS_SUBPACKAGES()
  else()
    ADD_SUBDIRECTORY(core)
    ADD_SUBDIRECTORY(containers)
    ADD_SUBDIRECTORY(algorithms)
    ADD_SUBDIRECTORY(example)
  endif()
ENDMACRO()

MACRO(KOKKOS_PACKAGE_DEF)
  if (KOKKOS_HAS_TRILINOS)
    TRIBITS_PACKAGE_DEF()
  else()
    #do nothing
  endif()
ENDMACRO()

MACRO(KOKKOS_INTERNAL_ADD_LIBRARY_INSTALL LIBRARY_NAME)
  KOKKOS_LIB_TYPE(${LIBRARY_NAME} INCTYPE)
  TARGET_INCLUDE_DIRECTORIES(${LIBRARY_NAME} ${INCTYPE} $<INSTALL_INTERFACE:include>)

  INSTALL(
    TARGETS ${LIBRARY_NAME}
    EXPORT ${PROJECT_NAME}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    COMPONENT ${PACKAGE_NAME}
  )

  INSTALL(
    TARGETS ${LIBRARY_NAME}
    EXPORT KokkosTargets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
  )

  INSTALL(
    TARGETS ${LIBRARY_NAME}
    EXPORT KokkosDeprecatedTargets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
  )

  VERIFY_EMPTY(KOKKOS_ADD_LIBRARY ${PARSE_UNPARSED_ARGUMENTS})
ENDMACRO()

FUNCTION(KOKKOS_ADD_EXECUTABLE EXE_NAME)
  if (KOKKOS_HAS_TRILINOS)
    TRIBITS_ADD_EXECUTABLE(${EXE_NAME} ${ARGN})
  else()
    CMAKE_PARSE_ARGUMENTS(PARSE 
      "TESTONLY"
      ""
      "SOURCES;TESTONLYLIBS"
      ${ARGN})

    ADD_EXECUTABLE(${EXE_NAME} ${PARSE_SOURCES})
    IF (PARSE_TESTONLYLIBS)
      TARGET_LINK_LIBRARIES(${EXE_NAME} ${PARSE_TESTONLYLIBS})
    ENDIF()
    #just link to a single lib kokkos now
    TARGET_LINK_LIBRARIES(${EXE_NAME} kokkos)
    VERIFY_EMPTY(KOKKOS_ADD_EXECUTABLE ${PARSE_UNPARSED_ARGUMENTS})
  endif()
ENDFUNCTION()

IF(NOT TARGET check)
  ADD_CUSTOM_TARGET(check COMMAND ${CMAKE_CTEST_COMMAND} -VV -C ${CMAKE_CFG_INTDIR})
ENDIF()


FUNCTION(KOKKOS_ADD_EXECUTABLE_AND_TEST ROOT_NAME)
IF (KOKKOS_HAS_TRILINOS)
  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    ${ROOT_NAME} 
    TESTONLYLIBS kokkos_gtest 
    ${ARGN}
    NUM_MPI_PROCS 1
    COMM serial mpi
    FAIL_REGULAR_EXPRESSION "  FAILED  "
  )
ELSE()
  CMAKE_PARSE_ARGUMENTS(PARSE 
    ""
    ""
    "SOURCES;CATEGORIES"
    ${ARGN})
  VERIFY_EMPTY(KOKKOS_ADD_EXECUTABLE_AND_TEST ${PARSE_UNPARSED_ARGUMENTS})
  SET(EXE_NAME ${PACKAGE_NAME}_${ROOT_NAME})
  KOKKOS_ADD_TEST_EXECUTABLE(${EXE_NAME}
    SOURCES ${PARSE_SOURCES}
  )
  KOKKOS_ADD_TEST(NAME ${ROOT_NAME} 
    EXE ${EXE_NAME}
    FAIL_REGULAR_EXPRESSION "  FAILED  "
  )
ENDIF()
ENDFUNCTION()

MACRO(KOKKOS_SETUP_BUILD_ENVIRONMENT)
 IF (NOT KOKKOS_HAS_TRILINOS)
  #------------ COMPILER AND FEATURE CHECKS ------------------------------------
  INCLUDE(${KOKKOS_SRC_PATH}/cmake/kokkos_functions.cmake)

  #------------ GET OPTIONS AND KOKKOS_SETTINGS --------------------------------
  # ADD Kokkos' modules to CMake's module path.
  SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${Kokkos_SOURCE_DIR}/cmake/Modules/")

  INCLUDE(${KOKKOS_SRC_PATH}/cmake/kokkos_enable_devices.cmake)
  INCLUDE(${KOKKOS_SRC_PATH}/cmake/kokkos_enable_options.cmake)
  INCLUDE(${KOKKOS_SRC_PATH}/cmake/kokkos_compiler_id.cmake)
  INCLUDE(${KOKKOS_SRC_PATH}/cmake/kokkos_cxx_std.cmake)
  INCLUDE(${KOKKOS_SRC_PATH}/cmake/kokkos_tpls.cmake)
  INCLUDE(${KOKKOS_SRC_PATH}/cmake/kokkos_arch.cmake)
 ENDIF()
ENDMACRO()

MACRO(KOKKOS_ADD_TEST_EXECUTABLE EXE_NAME)
  CMAKE_PARSE_ARGUMENTS(PARSE 
    ""
    ""
    "SOURCES"
    ${ARGN})
  KOKKOS_ADD_EXECUTABLE(${EXE_NAME}
    SOURCES ${PARSE_SOURCES}
    ${PARSE_UNPARSED_ARGUMENTS}
    TESTONLYLIBS kokkos_gtest
  )
  IF (NOT KOKKOS_HAS_TRILINOS)
    TARGET_LINK_LIBRARIES(${EXE_NAME} kokkos_gtest)
    ADD_DEPENDENCIES(check ${EXE_NAME})
  ENDIF()
ENDMACRO()

MACRO(KOKKOS_PACKAGE_POSTPROCESS)
  if (KOKKOS_HAS_TRILINOS)
    TRIBITS_PACKAGE_POSTPROCESS()
  endif()
ENDMACRO()

MACRO(KOKKOS_MAKE_LIBKOKKOS)
  IF (KOKKOS_SEPARATE_LIBS)
    MESSAGE(FATAL_ERROR "Internal error: should not make single libkokkos with -DKOKKOS_SEPARATE_LIBS=On")
  ENDIF()
  IF(${CMAKE_VERSION} VERSION_LESS "3.12" OR MSVC)
    #we are not able to set properties directly on object libraries yet
    #so we have had to delay kokkos_link_options until here
    #and do a bunch of other annoying work 
    ADD_LIBRARY(kokkos ${KOKKOS_SOURCE_DIR}/core/src/dummy.cpp 
      $<TARGET_OBJECTS:kokkoscore>
      $<TARGET_OBJECTS:kokkoscontainers>
    )
    TARGET_LINK_LIBRARIES(kokkos PUBLIC ${KOKKOS_LINK_OPTIONS})
    #still need the header-only library
    TARGET_LINK_LIBRARIES(kokkos PUBLIC kokkosalgorithms)
    KOKKOS_LINK_TPLS(kokkos)
    #these properties do not work transitively correctly so we
    #need some verbose hackery to make it work
    GET_TARGET_PROPERTY(CORE_DIRS kokkoscore       INTERFACE_INCLUDE_DIRECTORIES)
    GET_TARGET_PROPERTY(CTRS_DIRS kokkoscontainers INTERFACE_INCLUDE_DIRECTORIES)
    TARGET_INCLUDE_DIRECTORIES(kokkos PUBLIC ${CORE_DIRS})
    TARGET_INCLUDE_DIRECTORIES(kokkos PUBLIC ${CTRS_DIRS})

    GET_TARGET_PROPERTY(CORE_FLAGS kokkoscore       INTERFACE_COMPILE_OPTIONS)
    GET_TARGET_PROPERTY(CTRS_FLAGS kokkoscontainers INTERFACE_COMPILE_OPTIONS)
    TARGET_COMPILE_OPTIONS(kokkos PUBLIC ${CORE_FLAGS})
    TARGET_COMPILE_OPTIONS(kokkos PUBLIC ${CTRS_FLAGS})
  ELSE()
    ADD_LIBRARY(kokkos ${KOKKOS_SOURCE_DIR}/core/src/dummy.cpp)
    TARGET_LINK_LIBRARIES(kokkos PUBLIC kokkoscore kokkoscontainers)
    TARGET_LINK_LIBRARIES(kokkos PUBLIC kokkosalgorithms)
  ENDIF()
  KOKKOS_INTERNAL_ADD_LIBRARY_INSTALL(kokkos)
ENDMACRO()

FUNCTION(KOKKOS_LINK_TPLS LIBRARY_NAME)
  IF (KOKKOS_ENABLE_CUDA)
    IF (KOKKOS_CXX_COMPILER_ID STREQUAL Clang)
       SET(LIB_cuda "-lcuda -lcudart")
       find_library( cuda_lib_ NAMES libcuda cuda HINTS ${KOKKOS_CUDA_DIR}/lib64 ENV LD_LIBRARY_PATH ENV PATH )
       find_library( cudart_lib_ NAMES libcudart cudart HINTS ${KOKKOS_CUDA_DIR}/lib64 ENV LD_LIBRARY_PATH ENV PATH )
       if (cuda_lib_)
          TARGET_LINK_LIBRARIES(${LIBRARY_NAME} PUBLIC ${cuda_lib_})
       else()
          MESSAGE(SEND_ERROR "libcuda is required but could not be found. Make sure to include it in your LD_LIBRARY_PATH.")
       endif()
       if (cudart_lib_)
          TARGET_LINK_LIBRARIES(${LIBRARY_NAME} PUBLIC ${cudart_lib_})
       else()
         MESSAGE(SEND_ERROR "libcudart is required but could not be found. Make sure to include it in your LD_LIBRARY_PATH.")
       endif()
    else()
       SET(LIB_cuda "-lcuda")
       TARGET_LINK_LIBRARIES(${LIBRARY_NAME} PUBLIC cuda)
    endif()
  ENDIF()

  IF (KOKKOS_ENABLE_HPX)
    TARGET_LINK_LIBRARIES(${LIBRARY_NAME} PUBLIC ${HPX_LIBRARIES})
    TARGET_INCLUDE_DIRECTORIES(${LIBRARY_NAME} PUBLIC ${HPX_INCLUDE_DIRS})
  ENDIF()

  IF (KOKKOS_ENABLE_HWLOC)
    #this is really annoying that I have to do this
    #if CMake links statically, it will not link in hwloc which causes undefined refs downstream
    #even though hwloc is really "private" and doesn't need to be public I have to link publicly
    TARGET_LINK_LIBRARIES(${LIBRARY_NAME} PUBLIC Kokkos::hwloc)
    #what I don't want is the headers to be propagated downstream
    TARGET_INCLUDE_DIRECTORIES(${LIBRARY_NAME} PRIVATE ${HWLOC_INCLUDE_DIR})
  ENDIF()
  
  IF (KOKKOS_ENABLE_LIBNUMA)
    #this is really annoying that I have to do this
    #if CMake links statically, it will not link in hwloc which causes undefined refs downstream
    #even though hwloc is really "private" and doesn't need to be public I have to link publicly
    TARGET_LINK_LIBRARIES(${LIBRARY_NAME} PUBLIC Kokkos::libnuma)
    #what I don't want is the headers to be propagated downstream
    TARGET_INCLUDE_DIRECTORIES(${LIBRARY_NAME} PRIVATE ${LIBNUMA_INCLUDE_DIR})
  ENDIF()

  IF (KOKKOS_ENABLE_LIBRT)
    #this is really annoying that I have to do this
    #if CMake links statically, it will not link in librt which causes undefined refs downstream
    #even though librt is really "private" and doesn't need to be public I have to link publicly
    TARGET_LINK_LIBRARIES(${LIBRARY_NAME} PRIVATE Kokkos::librt)
    #what I don't want is the headers to be propagated downstream
    TARGET_INCLUDE_DIRECTORIES(${LIBRARY_NAME} PRIVATE ${LIBRT_INCLUDE_DIR})
  ENDIF()

  IF (KOKKOS_ENABLE_MEMKIND)
    TARGET_LINK_LIBRARIES(${LIBRARY_NAME} PRIVATE Kokkos::memkind)
  ENDIF()

  #dlfcn.h is in header files and needs to propagate
  TARGET_LINK_LIBRARIES(${LIBRARY_NAME} PUBLIC Kokkos::libdl)

ENDFUNCTION()

FUNCTION(KOKKOS_INTERNAL_ADD_LIBRARY LIBRARY_NAME)
  CMAKE_PARSE_ARGUMENTS(PARSE 
    "STATIC;SHARED"
    ""
    "HEADERS;SOURCES"
    ${ARGN})

  IF(PARSE_HEADERS)
    LIST(REMOVE_DUPLICATES PARSE_HEADERS)
  ENDIF()
  IF(PARSE_SOURCES)
    LIST(REMOVE_DUPLICATES PARSE_SOURCES)
  ENDIF()

  IF (KOKKOS_SEPARATE_LIBS)
    ADD_LIBRARY(
      ${LIBRARY_NAME}
      ${PARSE_HEADERS}
      ${PARSE_SOURCES}
    )
    KOKKOS_LINK_TPLS(${LIBRARY_NAME})
    IF(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.13")
      #great, this works the "right" way
      TARGET_LINK_OPTIONS(
        ${LIBRARY_NAME} PUBLIC ${KOKKOS_LINK_OPTIONS}
      )
    ELSE()
      #well, have to do it the wrong way for now
      TARGET_LINK_LIBRARIES(
        ${LIBRARY_NAME} PUBLIC ${KOKKOS_LINK_OPTIONS}
      )
    ENDIF()
  ELSE()
    ADD_LIBRARY(
      ${LIBRARY_NAME}
      OBJECT
      ${PARSE_HEADERS}
      ${PARSE_SOURCES}
    )
    IF(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.13")
      #great, this works the "right" way
      TARGET_LINK_OPTIONS(
        ${LIBRARY_NAME} PUBLIC ${KOKKOS_LINK_OPTIONS}
      )
      #I can go ahead and link the TPLs here
      KOKKOS_LINK_TPLS(${LIBRARY_NAME})
    ELSEIF(${CMAKE_VERSION} VERSION_LESS "3.12" OR MSVC)
      #nothing works yet for object libraries
      #we will need to hack this later for libkokkos
      #I also can't link the TPLs here - also must be delayed
    ELSE()
      TARGET_LINK_LIBRARIES(
        ${LIBRARY_NAME} PUBLIC ${KOKKOS_LINK_OPTIONS}
      )
      #I can go ahead and link the TPLs here
      KOKKOS_LINK_TPLS(${LIBRARY_NAME})
    ENDIF()
  ENDIF()

  TARGET_COMPILE_OPTIONS(
    ${LIBRARY_NAME}
    PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${KOKKOS_COMPILE_OPTIONS}>
  )

  IF (KOKKOS_ENABLE_CUDA)
    TARGET_COMPILE_OPTIONS(
      ${LIBRARY_NAME}
      PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${KOKKOS_CUDA_OPTIONS}>
    )
    SET(NODEDUP_CUDAFE_OPTIONS)
    FOREACH(OPT ${KOKKOS_CUDAFE_OPTIONS})
      LIST(APPEND NODEDUP_CUDAFE_OPTIONS -Xcudafe ${OPT})
    ENDFOREACH()
    TARGET_COMPILE_OPTIONS(
      ${LIBRARY_NAME} 
      PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${NODEDUP_CUDAFE_OPTIONS}>
    )
  ENDIF()

  LIST(LENGTH KOKKOS_XCOMPILER_OPTIONS XOPT_LENGTH)
  IF (XOPT_LENGTH GREATER 1)
    MESSAGE(FATAL_ERROR "CMake deduplication does not allow multiple -Xcompiler flags (${KOKKOS_XCOMPILER_OPTIONS}): will require Kokkos to upgrade to minimum 3.12")
  ENDIF()
  IF(KOKKOS_XCOMPILER_OPTIONS)
    SET(NODEDUP_XCOMPILER_OPTIONS)
    FOREACH(OPT ${KOKKOS_XCOMPILER_OPTIONS})
      #I have to do this for now because we can't guarantee 3.12 support
      #I really should do this with the shell option 
      LIST(APPEND NODEDUP_XCOMPILER_OPTIONS -Xcompiler) 
      LIST(APPEND NODEDUP_XCOMPILER_OPTIONS ${OPT}) 
    ENDFOREACH()
    TARGET_COMPILE_OPTIONS(
      ${LIBRARY_NAME} 
      PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${NODEDUP_XCOMPILER_OPTIONS}>
    )
  ENDIF()



  TARGET_INCLUDE_DIRECTORIES(
    ${LIBRARY_NAME}
    PUBLIC ${KOKKOS_TPL_INCLUDE_DIRS}
  )


  IF (KOKKOS_CXX_STANDARD_FEATURE)
    #GREAT! I can't do this the right way
    TARGET_COMPILE_FEATURES(${LIBRARY_NAME} PUBLIC ${KOKKOS_CXX_STANDARD_FEATURE})
    IF (NOT KOKKOS_USE_CXX_EXTENSIONS)
      SET_TARGET_PROPERTIES(${LIBRARY_NAME} PROPERTIES CXX_EXTENSIONS OFF)
    ENDIF()
  ELSE()
    #OH, Well, no choice but the wrong way
    ## commenting this out because it seems redundant (if it is left in there are two -std=c++11 entries) ...
    ## TARGET_COMPILE_OPTIONS(${LIBRARY_NAME} PUBLIC ${KOKKOS_CXX_STANDARD_FLAG})
  ENDIF()

  IF (KOKKOS_SEPARATE_LIBS OR ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.12")
    #Even if separate libs and these are object libraries
    #We still need to install them for transitive flags and deps
    KOKKOS_INTERNAL_ADD_LIBRARY_INSTALL(${LIBRARY_NAME})
  ELSE()
    #this is an object library and cmake <3.12 doesn't do this correctly
    #so.... do nothing
  ENDIF()

  INSTALL(
    FILES  ${PARSE_HEADERS}
    DESTINATION include
    COMPONENT ${PACKAGE_NAME}
  )

ENDFUNCTION()

FUNCTION(KOKKOS_ADD_LIBRARY LIBRARY_NAME)
  if (KOKKOS_HAS_TRILINOS)
    TRIBITS_ADD_LIBRARY(${LIBRARY_NAME} ${ARGN})
  else()
    KOKKOS_INTERNAL_ADD_LIBRARY(
      ${LIBRARY_NAME} ${ARGN})
  endif()
ENDFUNCTION()

FUNCTION(KOKKOS_ADD_INTERFACE_LIBRARY NAME)
IF (KOKKOS_HAS_TRILINOS)
  TRIBITS_ADD_LIBRARY(${NAME} ${ARGN})
ELSE()
  CMAKE_PARSE_ARGUMENTS(PARSE
    ""
    ""
    "HEADERS;SOURCES"
    ${ARGN}
  )

  ADD_LIBRARY(${NAME} INTERFACE)
  KOKKOS_INTERNAL_ADD_LIBRARY_INSTALL(${NAME})

  INSTALL(
    FILES  ${PARSE_HEADERS}
    DESTINATION include
  )

  INSTALL(
    FILES  ${PARSE_HEADERS}
    DESTINATION include
    COMPONENT ${PACKAGE_NAME}
  )
ENDIF()
ENDFUNCTION()

FUNCTION(KOKKOS_LIB_INCLUDE_DIRECTORIES TARGET)
  IF(KOKKOS_HAS_TRILINOS)
    #ignore the target, tribits doesn't do anything directly with targets
    TRIBITS_INCLUDE_DIRECTORIES(${ARGN})
  ELSE() #append to a list for later
    KOKKOS_LIB_TYPE(${TARGET} INCTYPE) 
    FOREACH(DIR ${ARGN})
      TARGET_INCLUDE_DIRECTORIES(${TARGET} ${INCTYPE} $<BUILD_INTERFACE:${DIR}>)
    ENDFOREACH()
  ENDIF()
ENDFUNCTION()

FUNCTION(KOKKOS_LIB_COMPILE_OPTIONS TARGET)
  IF(KOKKOS_HAS_TRILINOS)
    #don't trust tribits to do this correctly
    KOKKOS_TARGET_COMPILE_OPTIONS(${TARGET} ${ARGN})
  ELSE()
    KOKKOS_LIB_TYPE(${TARGET} INCTYPE)
    KOKKOS_TARGET_COMPILE_OPTIONS(${${PROJECT_NAME}_LIBRARY_NAME_PREFIX}${TARGET} ${INCTYPE} ${ARGN})
  ENDIF()
ENDFUNCTION()

MACRO(KOKKOS_ADD_TEST_DIRECTORIES)
  IF (KOKKOS_HAS_TRILINOS)
    TRIBITS_ADD_TEST_DIRECTORIES(${ARGN})
  ELSE()
    IF(KOKKOS_ENABLE_TESTS)
      FOREACH(TEST_DIR ${ARGN})
        ADD_SUBDIRECTORY(${TEST_DIR})
      ENDFOREACH()
    ENDIF()
  ENDIF()
ENDMACRO()

MACRO(KOKKOS_ADD_EXAMPLE_DIRECTORIES)
  if (KOKKOS_HAS_TRILINOS)
    TRIBITS_ADD_EXAMPLE_DIRECTORIES(${ARGN})
  else()
    IF(KOKKOS_ENABLE_EXAMPLES)
      FOREACH(EXAMPLE_DIR ${ARGN})
        ADD_SUBDIRECTORY(${EXAMPLE_DIR})
      ENDFOREACH()
    ENDIF()
  endif()
ENDMACRO()
