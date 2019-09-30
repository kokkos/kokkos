
# We want to determine if options are given with the wrong case
# In order to detect which arguments are given to compare against
# the list of valid arguments, at the beginning here we need to
# form a list of all the given variables. If it begins with any 
# case of KoKkOS, we add it to the list.


GET_CMAKE_PROPERTY(_variableNames VARIABLES)
SET(KOKKOS_GIVEN_VARIABLES)
FOREACH (var ${_variableNames})
  STRING(TOUPPER ${var} UC_VAR)
  STRING(FIND ${UC_VAR} KOKKOS IDX)
  IF (${IDX} EQUAL 0)
    LIST(APPEND KOKKOS_GIVEN_VARIABLES ${var})
  ENDIF()
ENDFOREACH()

# Basic initialization (Used in KOKKOS_SETTINGS)
SET(Kokkos_SOURCE_DIR    ${CMAKE_CURRENT_SOURCE_DIR})
SET(KOKKOS_SOURCE_DIR    ${CMAKE_CURRENT_SOURCE_DIR})
SET(KOKKOS_SRC_PATH      ${Kokkos_SOURCE_DIR})
SET(KOKKOS_PATH          ${Kokkos_SOURCE_DIR})
SET(KOKKOS_TOP_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Needed to simplify syntax of if statements
CMAKE_POLICY(SET CMP0054 NEW)

# Is this a build as part of Trilinos?
IF(COMMAND TRIBITS_PACKAGE_DECL)
  SET(KOKKOS_HAS_TRILINOS ON)
ELSE()
  SET(KOKKOS_HAS_TRILINOS OFF)
ENDIF()

IF (NOT KOKKOS_HAS_TRILINOS)
  INCLUDE(${KOKKOS_SRC_PATH}/cmake/kokkos_preamble.cmake)
ENDIF()


SET(KOKKOS_ENABLED_OPTIONS)      #exported in config file
SET(KOKKOS_ENABLED_DEVICES)      #exported in config file
SET(KOKKOS_ENABLED_ARCH_LIST)    #exported in config file

#These are helper flags used for sanity checks during config
#Certain features should depend on other features being configured first
SET(KOKKOS_CFG_DAG_NONE              On) #sentinel to indicate no dependencies
SET(KOKKOS_CFG_DAG_DEVICES_DONE      Off)
SET(KOKKOS_CFG_DAG_OPTIONS_DONE      Off)
SET(KOKKOS_CFG_DAG_ARCH_DONE         Off)
SET(KOKKOS_CFG_DAG_CXX_STD_DONE      Off)
SET(KOKKOS_CFG_DAG_COMPILER_ID_DONE  Off)
FUNCTION(KOKKOS_CFG_DEPENDS SUCCESSOR PRECURSOR)
  SET(PRE_FLAG  KOKKOS_CFG_DAG_${PRECURSOR})
  SET(POST_FLAG KOKKOS_CFG_DAG_${SUCCESSOR})
  IF (NOT ${PRE_FLAG})
    MESSAGE(FATAL_ERROR "Bad CMake refactor: feature ${SUCCESSOR} cannot be configured until ${PRECURSOR} is configured")
  ENDIF()
  GLOBAL_SET(${POST_FLAG} On)
ENDFUNCTION()


list(APPEND CMAKE_MODULE_PATH cmake/Modules)

IF(NOT KOKKOS_HAS_TRILINOS)
  cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
  set(CMAKE_DISABLE_SOURCE_CHANGES ON)
  set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
  IF (Spack_WORKAROUND) 
    #if we are explicitly using Spack for development,
    #nuke the Spack compiler
    SEt(SPACK_CXX $ENV{SPACK_CXX})
    IF(SPACK_CXX)
      SET(CMAKE_CXX_COMPILER ${SPACK_CXX} CACHE STRING "the C++ compiler" FORCE)
      SET(ENV{CXX} ${SPACK_CXX})
    ENDIF()
  ENDif()
  IF(NOT DEFINED ${PROJECT_NAME})
    PROJECT(Kokkos CXX)
  ENDIF()
ENDIF()

set(Kokkos_VERSION_MAJOR 3)
set(Kokkos_VERSION_MINOR 0)
set(Kokkos_VERSION_PATCH 0)
set(Kokkos_VERSION "${Kokkos_VERSION_MAJOR}.${Kokkos_VERSION_MINOR}.${Kokkos_VERSION_PATCH}")

IF(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.12.0") 
  MESSAGE(STATUS "Setting policy CMP0074 to use <Package>_ROOT variables")
  CMAKE_POLICY(SET CMP0074 NEW)
ENDIF()

# Load either the real TriBITS or a TriBITS wrapper 
# for certain utility functions that are universal (like GLOBAL_SET)
INCLUDE(${KOKKOS_SRC_PATH}/cmake/fake_tribits.cmake)
GLOBAL_SET(KOKKOS_COMPILE_OPTIONS)
GLOBAL_SET(KOKKOS_LINK_OPTIONS)
GLOBAL_SET(KOKKOS_CUDA_OPTIONS)
GLOBAL_SET(KOKKOS_CUDAFE_OPTIONS)
GLOBAL_SET(KOKKOS_XCOMPILER_OPTIONS)


# Include a set of Kokkos-specific wrapper functions that
# will either call raw CMake or TriBITS
# These are functions like KOKKOS_INCLUDE_DIRECTORIES
INCLUDE(${KOKKOS_SRC_PATH}/cmake/kokkos_tribits.cmake)

# The build environment setup goes in the following steps
# 1) Check all the enable options. This includes checking Kokkos_DEVICES
# 2) Check the compiler ID (type and version)
# 3) Check the CXX standard and select important CXX flags
# 4) Check for any third-party libraries (TPLs) like hwloc
# 5) Check if optimizing for a particular architecture and add arch-specific flags
KOKKOS_SETUP_BUILD_ENVIRONMENT()

# Finish off the build
# 6) Recurse into subdirectories and configure individual libraries
# 7) Export and install targets

OPTION(BUILD_SHARED_LIBS "Build shared libraries" OFF)
# Workaround for building position independent code.
IF(BUILD_SHARED_LIBS)
  SET(CMAKE_POSITION_INDEPENDENT_CODE ON)
ENDIF()

IF (NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  MESSAGE(FATAL_ERROR "Kokkos assumes a 64-bit build; i.e., 8-byte pointers, but found ${CMAKE_SIZEOF_VOID_P}-byte pointers instead")
ENDIF()

IF(Kokkos_SEPARATE_LIBS)
 SET(KOKKOS_EXT_LIBRARIES Kokkos::kokkoscore Kokkos::kokkoscontainers Kokkos::kokkosalgorithms)
 SET(KOKKOS_INT_LIBRARIES kokkoscore kokkoscontainers kokkosalgorithms)
ELSE()
 SET(KOKKOS_EXT_LIBRARIES Kokkos::kokkos)
 SET(KOKKOS_INT_LIBRARIES kokkos)
ENDIF()
SET_PROPERTY(GLOBAL PROPERTY KOKKOS_INT_LIBRARIES ${KOKKOS_INT_LIBRARIES})

GET_DIRECTORY_PROPERTY(HAS_PARENT PARENT_DIRECTORY)
IF (KOKKOS_HAS_TRILINOS)
  SET(TRILINOS_INCDIR ${CMAKE_INSTALL_PREFIX}/${${PROJECT_NAME}_INSTALL_INCLUDE_DIR})
  SET(KOKKOS_HEADER_DIR ${TRILINOS_INCDIR})
  SET(KOKKOS_IS_SUBDIRECTORY TRUE)
ELSEIF(HAS_PARENT)
  SET(KOKKOS_HEADER_DIR "include/kokkos")
  SET(KOKKOS_IS_SUBDIRECTORY TRUE)
ELSE()
  SET(KOKKOS_HEADER_DIR "${CMAKE_INSTALL_INCLUDEDIR}")
  SET(KOKKOS_IS_SUBDIRECTORY FALSE)
ENDIF()



#------------------------------------------------------------------------------
#
# A) Forward declare the package so that certain options are also defined for
# subpackages
#

KOKKOS_PACKAGE_DECL()


#------------------------------------------------------------------------------
#
# D) Process the subpackages (subdirectories) for Kokkos
#
KOKKOS_PROCESS_SUBPACKAGES()


#------------------------------------------------------------------------------
#
# E) If Kokkos itself is enabled, process the Kokkos package
#

KOKKOS_PACKAGE_DEF()
KOKKOS_EXCLUDE_AUTOTOOLS_FILES()
KOKKOS_PACKAGE_POSTPROCESS()

IF (NOT KOKKOS_HAS_TRILINOS)
  IF (NOT KOKKOS_SEPARATE_LIBS)
    KOKKOS_MAKE_LIBKOKKOS()
  ENDIF()
  include(${KOKKOS_SRC_PATH}/cmake/kokkos_install.cmake)
ENDIF()

# nvcc_wrapper is Kokkos' wrapper for NVIDIA's NVCC CUDA compiler.
# Kokkos needs nvcc_wrapper in order to build.  Other libraries and
# executables also need nvcc_wrapper.  Thus, we need to install it.
# If the argument of DESTINATION is a relative path, CMake computes it
# as relative to ${CMAKE_INSTALL_PATH}.
INSTALL(PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/bin/nvcc_wrapper DESTINATION ${CMAKE_INSTALL_BINDIR})
INSTALL(FILES "${CMAKE_BINARY_DIR}/KokkosCore_config.h" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})


#  Finally - if we are a subproject - make sure the enabled devices are visible
IF (HAS_PARENT)
  FOREACH(DEV Kokkos_ENABLED_DEVICES)
    #I would much rather not make these cache variables or global properties, but I can't
    #make any guarantees on whether PARENT_SCOPE is good enough to make 
    #these variables visible where I need them
    SET(Kokkos_ENABLE_${DEV} ON PARENT_SCOPE)
    SET_PROPERTY(GLOBAL PROPERTY Kokkos_ENABLE_${DEV} ON)
  ENDFOREACH()
ENDIF()


FUNCTION(TARGET_LINK_KOKKOS TARGET)
  CMAKE_PARSE_ARGUMENTS(
    PARSE
    "PRIVATE;PUBLIC;INTERFACE"
    ""
    ""
    ${ARGN}
  )

  SET(LINK_TYPE)
  IF (PARSE_PRIVATE)
    SET(LINK_TYPE PRIVATE)
  ELSEIF(PARSE_PUBLIC)
    SET(LINK_TYPE PUBLIC)
  ELSEIF(PARSE_INTERFACE)
    SET(LINK_TYPE INTERFACE)
  ENDIF()

  GET_PROPERTY(KOKKOS_LIBRARIES GLOBAL PROPERTY KOKKOS_INT_LIBRARIES)

  FOREACH(LIB ${KOKKOS_LIBRARIES})
    TARGET_LINK_LIBRARIES(${TARGET} ${LINK_TYPE} ${LIB})
  ENDFOREACH()
ENDFUNCTION()
