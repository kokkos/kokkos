#These are tribits wrappers only ever called by Kokkos itself

INCLUDE(CMakeParseArguments)
INCLUDE(CTest)
INCLUDE(GNUInstallDirs)

MESSAGE(STATUS "The project name is: ${PROJECT_NAME}")

FUNCTION(VERIFY_EMPTY CONTEXT)
  if(${ARGN})
    MESSAGE(FATAL_ERROR "Kokkos does not support all of Tribits. Unhandled arguments in ${CONTEXT}:\n${ARGN}")
  endif()
ENDFUNCTION()

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
  TARGET_INCLUDE_DIRECTORIES(${LIBRARY_NAME} ${INCTYPE} $<INSTALL_INTERFACE:${KOKKOS_HEADER_DIR}>)

  INSTALL(
    TARGETS ${LIBRARY_NAME}
    EXPORT ${PROJECT_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT ${PACKAGE_NAME}
  )

  INSTALL(
    TARGETS ${LIBRARY_NAME}
    EXPORT KokkosTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )

  VERIFY_EMPTY(KOKKOS_ADD_LIBRARY ${PARSE_UNPARSED_ARGUMENTS})
ENDMACRO()

FUNCTION(KOKKOS_ADD_EXECUTABLE ROOT_NAME)
  if (KOKKOS_HAS_TRILINOS)
    TRIBITS_ADD_EXECUTABLE(${ROOT_NAME} ${ARGN})
  else()
    CMAKE_PARSE_ARGUMENTS(PARSE
      "TESTONLY"
      ""
      "SOURCES;TESTONLYLIBS"
      ${ARGN})

    SET(EXE_NAME ${PACKAGE_NAME}_${ROOT_NAME})
    ADD_EXECUTABLE(${EXE_NAME} ${PARSE_SOURCES})
    IF (PARSE_TESTONLYLIBS)
      TARGET_LINK_LIBRARIES(${EXE_NAME} PRIVATE ${PARSE_TESTONLYLIBS})
    ENDIF()
    VERIFY_EMPTY(KOKKOS_ADD_EXECUTABLE ${PARSE_UNPARSED_ARGUMENTS})
    #All executables must link to all the kokkos targets
    #This is just private linkage because exe is final
    TARGET_LINK_LIBRARIES(${EXE_NAME} PRIVATE Kokkos::kokkos)
  endif()
ENDFUNCTION()

FUNCTION(KOKKOS_ADD_EXECUTABLE_AND_TEST ROOT_NAME)
CMAKE_PARSE_ARGUMENTS(PARSE
  ""
  ""
  "SOURCES;CATEGORIES;ARGS"
  ${ARGN})
VERIFY_EMPTY(KOKKOS_ADD_EXECUTABLE_AND_TEST ${PARSE_UNPARSED_ARGUMENTS})

IF (KOKKOS_HAS_TRILINOS)
  IF(DEFINED PARSE_ARGS)
    STRING(REPLACE ";" " " PARSE_ARGS "${PARSE_ARGS}")
  ENDIF()
  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    ${ROOT_NAME}
    SOURCES ${PARSE_SOURCES}
    TESTONLYLIBS kokkos_gtest
    NUM_MPI_PROCS 1
    COMM serial mpi
    ARGS ${PARSE_ARGS}
    CATEGORIES ${PARSE_CATEGORIES}
    SOURCES ${PARSE_SOURCES}
    FAIL_REGULAR_EXPRESSION "  FAILED  "
    ARGS ${PARSE_ARGS}
  )
ELSE()
  KOKKOS_ADD_TEST_EXECUTABLE(${ROOT_NAME}
    SOURCES ${PARSE_SOURCES}
  )
  KOKKOS_ADD_TEST(NAME ${ROOT_NAME}
    EXE ${ROOT_NAME}
    FAIL_REGULAR_EXPRESSION "  FAILED  "
    ARGS ${PARSE_ARGS}
  )
ENDIF()
ENDFUNCTION()

FUNCTION(KOKKOS_SET_EXE_PROPERTY ROOT_NAME)
  SET(TARGET_NAME ${PACKAGE_NAME}_${ROOT_NAME})
  IF (NOT TARGET ${TARGET_NAME})
    MESSAGE(SEND_ERROR "No target ${TARGET_NAME} exists - cannot set target properties")
  ENDIF()
  SET_PROPERTY(TARGET ${TARGET_NAME} PROPERTY ${ARGN})
ENDFUNCTION()

MACRO(KOKKOS_SETUP_BUILD_ENVIRONMENT)
  # This is needed for both regular build and install tests
  INCLUDE(${KOKKOS_SRC_PATH}/cmake/kokkos_compiler_id.cmake)
  #set an internal option, if not already set
  SET(Kokkos_INSTALL_TESTING OFF CACHE INTERNAL "Whether to build tests and examples against installation")
  IF (Kokkos_INSTALL_TESTING)
    SET(KOKKOS_ENABLE_TESTS ON)
    SET(KOKKOS_ENABLE_EXAMPLES ON)
    # This looks a little weird, but what we are doing
    # is to NOT build Kokkos but instead look for an
    # installed Kokkos - then build examples and tests
    # against that installed Kokkos
    FIND_PACKAGE(Kokkos REQUIRED)
    # Just grab the configuration from the installation
    FOREACH(DEV ${Kokkos_DEVICES})
      SET(KOKKOS_ENABLE_${DEV} ON)
    ENDFOREACH()
    FOREACH(OPT ${Kokkos_OPTIONS})
      SET(KOKKOS_ENABLE_${OPT} ON)
    ENDFOREACH()
    FOREACH(TPL ${Kokkos_TPLS})
      SET(KOKKOS_ENABLE_${TPL} ON)
    ENDFOREACH()
    FOREACH(ARCH ${Kokkos_ARCH})
      SET(KOKKOS_ARCH_${ARCH} ON)
    ENDFOREACH()
  ELSE()
    INCLUDE(${KOKKOS_SRC_PATH}/cmake/kokkos_enable_devices.cmake)
    INCLUDE(${KOKKOS_SRC_PATH}/cmake/kokkos_enable_options.cmake)
    INCLUDE(${KOKKOS_SRC_PATH}/cmake/kokkos_test_cxx_std.cmake)
    INCLUDE(${KOKKOS_SRC_PATH}/cmake/kokkos_arch.cmake)
    IF (NOT KOKKOS_HAS_TRILINOS)
      SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${Kokkos_SOURCE_DIR}/cmake/Modules/")
    ENDIF()
    INCLUDE(${KOKKOS_SRC_PATH}/cmake/kokkos_tpls.cmake)
    INCLUDE(${KOKKOS_SRC_PATH}/cmake/kokkos_corner_cases.cmake)
  ENDIF()
ENDMACRO()

MACRO(KOKKOS_ADD_TEST_EXECUTABLE ROOT_NAME)
  CMAKE_PARSE_ARGUMENTS(PARSE
    ""
    ""
    "SOURCES"
    ${ARGN})
  KOKKOS_ADD_EXECUTABLE(${ROOT_NAME}
    SOURCES ${PARSE_SOURCES}
    ${PARSE_UNPARSED_ARGUMENTS}
    TESTONLYLIBS kokkos_gtest
  )
  SET(EXE_NAME ${PACKAGE_NAME}_${ROOT_NAME})
ENDMACRO()

MACRO(KOKKOS_PACKAGE_POSTPROCESS)
  if (KOKKOS_HAS_TRILINOS)
    TRIBITS_PACKAGE_POSTPROCESS()
  endif()
ENDMACRO()

## KOKKOS_CONFIGURE_CORE  Configure/Generate header files for core content based
##                        on enabled backends.
##                        KOKKOS_FWD is the forward declare set
##                        KOKKOS_SETUP  is included in Kokkos_Macros.hpp and include prefix includes/defines
##                        KOKKOS_DECLARE is the declaration set
##                        KOKKOS_POST_INCLUDE is included at the end of Kokkos_Core.hpp
MACRO(KOKKOS_CONFIGURE_CORE)
   SET(FWD_BACKEND_LIST)
   FOREACH(MEMSPACE ${KOKKOS_MEMSPACE_LIST})
      LIST(APPEND FWD_BACKEND_LIST ${MEMSPACE})
   ENDFOREACH()
   FOREACH(BACKEND_ ${KOKKOS_ENABLED_DEVICES})
      IF( ${BACKEND_} STREQUAL "PTHREAD")
         LIST(APPEND FWD_BACKEND_LIST THREADS)
      ELSE()
         LIST(APPEND FWD_BACKEND_LIST ${BACKEND_})
      ENDIF()
   ENDFOREACH()
   MESSAGE(STATUS "Kokkos Devices: ${KOKKOS_ENABLED_DEVICES}, Kokkos Backends: ${FWD_BACKEND_LIST}")
   KOKKOS_CONFIG_HEADER( KokkosCore_Config_HeaderSet.in KokkosCore_Config_FwdBackend.hpp "KOKKOS_FWD" "fwd/Kokkos_Fwd" "${FWD_BACKEND_LIST}")
   KOKKOS_CONFIG_HEADER( KokkosCore_Config_HeaderSet.in KokkosCore_Config_SetupBackend.hpp "KOKKOS_SETUP" "setup/Kokkos_Setup" "${DEVICE_SETUP_LIST}")
   KOKKOS_CONFIG_HEADER( KokkosCore_Config_HeaderSet.in KokkosCore_Config_DeclareBackend.hpp "KOKKOS_DECLARE" "decl/Kokkos_Declare" "${FWD_BACKEND_LIST}")
   KOKKOS_CONFIG_HEADER( KokkosCore_Config_HeaderSet.in KokkosCore_Config_PostInclude.hpp "KOKKOS_POST_INCLUDE" "Kokkos_Post_Include" "${KOKKOS_BACKEND_POST_INCLUDE_LIST}")
   SET(_DEFAULT_HOST_MEMSPACE "::Kokkos::HostSpace")
   KOKKOS_OPTION(DEFAULT_DEVICE_MEMORY_SPACE "" STRING "Override default device memory space")
   KOKKOS_OPTION(DEFAULT_HOST_MEMORY_SPACE "" STRING "Override default host memory space")
   KOKKOS_OPTION(DEFAULT_DEVICE_EXECUTION_SPACE "" STRING "Override default device execution space")
   KOKKOS_OPTION(DEFAULT_HOST_PARALLEL_EXECUTION_SPACE "" STRING "Override default host parallel execution space")
   IF (NOT Kokkos_DEFAULT_DEVICE_EXECUTION_SPACE STREQUAL "")
      SET(_DEVICE_PARALLEL ${Kokkos_DEFAULT_DEVICE_EXECUTION_SPACE})
      MESSAGE(STATUS "Override default device execution space: ${_DEVICE_PARALLEL}")
      SET(KOKKOS_DEVICE_SPACE_ACTIVE ON)
   ELSE()
      IF (_DEVICE_PARALLEL STREQUAL "NoTypeDefined")
         SET(KOKKOS_DEVICE_SPACE_ACTIVE OFF)
      ELSE()
         SET(KOKKOS_DEVICE_SPACE_ACTIVE ON)
      ENDIF()
   ENDIF()
   IF (NOT Kokkos_DEFAULT_HOST_PARALLEL_EXECUTION_SPACE STREQUAL "")
      SET(_HOST_PARALLEL ${Kokkos_DEFAULT_HOST_PARALLEL_EXECUTION_SPACE})
      MESSAGE(STATUS "Override default host parallel execution space: ${_HOST_PARALLEL}")
      SET(KOKKOS_HOSTPARALLEL_SPACE_ACTIVE ON)
   ELSE()
      IF (_HOST_PARALLEL STREQUAL "NoTypeDefined")
         SET(KOKKOS_HOSTPARALLEL_SPACE_ACTIVE OFF)
      ELSE()
         SET(KOKKOS_HOSTPARALLEL_SPACE_ACTIVE ON)
      ENDIF()
   ENDIF()
   #We are ready to configure the header
   CONFIGURE_FILE(cmake/KokkosCore_config.h.in KokkosCore_config.h @ONLY)
ENDMACRO()

## KOKKOS_INSTALL_ADDITIONAL_FILES - instruct cmake to install files in target destination.
##                        Includes generated header files, scripts such as nvcc_wrapper and hpcbind,
##                        as well as other files provided through plugins.
MACRO(KOKKOS_INSTALL_ADDITIONAL_FILES)
  # kokkos_launch_compiler is used by Kokkos to prefix compiler commands so that they forward to nvcc_wrapper
  INSTALL(PROGRAMS
          "${CMAKE_CURRENT_SOURCE_DIR}/bin/nvcc_wrapper"
          "${CMAKE_CURRENT_SOURCE_DIR}/bin/hpcbind"
          "${CMAKE_CURRENT_SOURCE_DIR}/bin/kokkos_launch_compiler"
          DESTINATION ${CMAKE_INSTALL_BINDIR})
  INSTALL(FILES
          "${CMAKE_CURRENT_BINARY_DIR}/KokkosCore_config.h"
          "${CMAKE_CURRENT_BINARY_DIR}/KokkosCore_Config_FwdBackend.hpp"
          "${CMAKE_CURRENT_BINARY_DIR}/KokkosCore_Config_SetupBackend.hpp"
          "${CMAKE_CURRENT_BINARY_DIR}/KokkosCore_Config_DeclareBackend.hpp"
          "${CMAKE_CURRENT_BINARY_DIR}/KokkosCore_Config_PostInclude.hpp"
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
ENDMACRO()

FUNCTION(KOKKOS_SET_LIBRARY_PROPERTIES LIBRARY_NAME)
  CMAKE_PARSE_ARGUMENTS(PARSE
    "PLAIN_STYLE"
    ""
    ""
    ${ARGN})

  IF(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.18")
    #I can use link options
    #check for CXX linkage using the simple 3.18 way
    TARGET_LINK_OPTIONS(
      ${LIBRARY_NAME} PUBLIC
      $<$<LINK_LANGUAGE:CXX>:${KOKKOS_LINK_OPTIONS}>
    )
  ELSEIF(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.13")
    #I can use link options
    #just assume CXX linkage
    TARGET_LINK_OPTIONS(
      ${LIBRARY_NAME} PUBLIC ${KOKKOS_LINK_OPTIONS}
    )
  ELSE()
    #assume CXX linkage, we have no good way to check otherwise
    IF (PARSE_PLAIN_STYLE)
      TARGET_LINK_LIBRARIES(
        ${LIBRARY_NAME} ${KOKKOS_LINK_OPTIONS}
      )
    ELSE()
      #well, have to do it the wrong way for now
      TARGET_LINK_LIBRARIES(
        ${LIBRARY_NAME} PUBLIC ${KOKKOS_LINK_OPTIONS}
      )
    ENDIF()
  ENDIF()

  TARGET_COMPILE_OPTIONS(
    ${LIBRARY_NAME} PUBLIC
    $<$<COMPILE_LANGUAGE:CXX>:${KOKKOS_COMPILE_OPTIONS}>
  )

  TARGET_COMPILE_DEFINITIONS(
    ${LIBRARY_NAME} PUBLIC
    $<$<COMPILE_LANGUAGE:CXX>:${KOKKOS_COMPILE_DEFINITIONS}>
  )

  TARGET_LINK_LIBRARIES(
    ${LIBRARY_NAME} PUBLIC ${KOKKOS_LINK_LIBRARIES}
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

  IF (KOKKOS_ENABLE_HIP)
    TARGET_COMPILE_OPTIONS(
      ${LIBRARY_NAME}
      PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${KOKKOS_AMDGPU_OPTIONS}>
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

  IF (KOKKOS_CXX_STANDARD_FEATURE)
    #GREAT! I can do this the right way
    TARGET_COMPILE_FEATURES(${LIBRARY_NAME} PUBLIC ${KOKKOS_CXX_STANDARD_FEATURE})
    IF (NOT KOKKOS_USE_CXX_EXTENSIONS)
      SET_TARGET_PROPERTIES(${LIBRARY_NAME} PROPERTIES CXX_EXTENSIONS OFF)
    ENDIF()
  ELSE()
    #OH, well, no choice but the wrong way
    TARGET_COMPILE_OPTIONS(${LIBRARY_NAME} PUBLIC ${KOKKOS_CXX_STANDARD_FLAG})
  ENDIF()
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

  IF(PARSE_STATIC)
    SET(LINK_TYPE STATIC)
  ENDIF()

  IF(PARSE_SHARED)
    SET(LINK_TYPE SHARED)
  ENDIF()

  # MSVC and other platforms want to have
  # the headers included as source files
  # for better dependency detection
  ADD_LIBRARY(
    ${LIBRARY_NAME}
    ${LINK_TYPE}
    ${PARSE_HEADERS}
    ${PARSE_SOURCES}
  )

  KOKKOS_INTERNAL_ADD_LIBRARY_INSTALL(${LIBRARY_NAME})

  #In case we are building in-tree, add an alias name
  #that matches the install Kokkos:: name
  ADD_LIBRARY(Kokkos::${LIBRARY_NAME} ALIAS ${LIBRARY_NAME})
ENDFUNCTION()

FUNCTION(KOKKOS_ADD_LIBRARY LIBRARY_NAME)
  CMAKE_PARSE_ARGUMENTS(PARSE
    "ADD_BUILD_OPTIONS"
    ""
    "HEADERS"
    ${ARGN}
  )
  IF (KOKKOS_HAS_TRILINOS)
    # We do not pass headers to trilinos. They would get installed
    # to the default include folder, but we want headers installed
    # preserving the directory structure, e.g. impl
    # If headers got installed in both locations, it breaks some
    # downstream packages
    TRIBITS_ADD_LIBRARY(${LIBRARY_NAME} ${PARSE_UNPARSED_ARGUMENTS})
    #Stolen from Tribits - it can add prefixes
    SET(TRIBITS_LIBRARY_NAME_PREFIX "${${PROJECT_NAME}_LIBRARY_NAME_PREFIX}")
    SET(TRIBITS_LIBRARY_NAME ${TRIBITS_LIBRARY_NAME_PREFIX}${LIBRARY_NAME})
    #Tribits has way too much techinical debt and baggage to even
    #allow PUBLIC target_compile_options to be used. It forces C++ flags on projects
    #as a giant blob of space-separated strings. We end up with duplicated
    #flags between the flags implicitly forced on Kokkos-dependent and those Kokkos
    #has in its public INTERFACE_COMPILE_OPTIONS.
    #These do NOT get de-deduplicated because Tribits
    #creates flags as a giant monolithic space-separated string
    #Do not set any transitive properties and keep everything working as before
    #KOKKOS_SET_LIBRARY_PROPERTIES(${TRIBITS_LIBRARY_NAME} PLAIN_STYLE)
  ELSE()
    # Forward the headers, we want to know about all headers
    # to make sure they appear correctly in IDEs
    KOKKOS_INTERNAL_ADD_LIBRARY(
      ${LIBRARY_NAME} ${PARSE_UNPARSED_ARGUMENTS} HEADERS ${PARSE_HEADERS})
    IF (PARSE_ADD_BUILD_OPTIONS)
      KOKKOS_SET_LIBRARY_PROPERTIES(${LIBRARY_NAME})
    ENDIF()
  ENDIF()
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
