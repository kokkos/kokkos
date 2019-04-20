# Is this a build as part of Trilinos?

IF(COMMAND TRIBITS_PACKAGE_DECL)
  SET(KOKKOS_HAS_TRILINOS ON CACHE BOOL "")
ELSE()
  SET(KOKKOS_HAS_TRILINOS OFF CACHE BOOL "")
ENDIF()

# Basic initialization (Used in KOKKOS_SETTINGS)
set(Kokkos_SOURCE_DIR    ${CMAKE_CURRENT_SOURCE_DIR})
set(KOKKOS_SOURCE_DIR    ${CMAKE_CURRENT_SOURCE_DIR})
set(KOKKOS_SRC_PATH      ${Kokkos_SOURCE_DIR})
set(KOKKOS_PATH          ${Kokkos_SOURCE_DIR})
set(KOKKOS_TOP_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})

OPTION(KOKKOS_ENABLE_EXPLICIT_INSTANATION "Whether to enable explicit template instantiation"  OFF)
OPTION(KOKKOS_ENABLE_ETI                  "Whether to enable explicit template instantiation"  OFF)
OPTION(KOKKOS_ENABLE_Cuda_RDC             "Whether to enable relocatable device code for CUDA" OFF)

IF(NOT KOKKOS_HAS_TRILINOS)
  cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
  IF(NOT DEFINED ${PROJECT_NAME})
    PROJECT(Kokkos CXX)
  ENDIF()
  set (Kokkos_VERSION_MAJOR 2)
  set (Kokkos_VERSION_MINOR 7)
  set (Kokkos_VERSION_PATCH 4)
ENDIF()

SET(KOKKOS_CXX_STANDARD "" CACHE STRING "The C++ standard for Kokkos to use: c++11, c++14, or c++17")
SET(KOKKOS_CXX_FEATURES "" CACHE STRING "The list of C++ features for Kokkos to enable")
SET(CXX_STANDARD_TEST)

IF (KOKKOS_CXX_STANDARD AND CMAKE_CXX_STANDARD)
  #make sure these are consistent
  IF (${KOKKOS_CXX_STANDARD} STREQUAL "c++11")
    IF(NOT ${CMAKE_CXX_STANDARD} STREQUAL "11")
      SET(CXX_STD_ERROR ON)
    ENDIF()
  ELSEIF (${KOKKOS_CXX_STANDARD} STREQUAL "c++14")
    IF(NOT ${CMAKE_CXX_STANDARD} STREQUAL "14")
      SET(CXX_STD_ERROR ON)
    ENDIF()
  ELSEIF (${KOKKOS_CXX_STANDARD} STREQUAL "c++17")
    IF(NOT ${CMAKE_CXX_STANDARD} STREQUAL "17")
      SET(CXX_STD_ERROR ON)
    ENDIF()
  ELSE()
    #KOKKOS_CXX_STANDARD is something else, which means definitely invalid
    SET(CXX_STD_ERROR ON)
  ENDIF()
  IF (CXX_STD_ERROR)
    MESSAGE(FATAL_ERROR "Specified both CMAKE_CXX_STANDARD and KOKKOS_CXX_STANDARD, but they don't match")
  ENDIF()
ENDIF()

IF (NOT KOKKOS_CXX_STANDARD AND NOT CMAKE_CXX_STANDARD)
  SET(KOKKOS_CXX_STANDARD "c++11")
ENDIF()

IF (KOKKOS_CXX_STANDARD)
  IF (${KOKKOS_CXX_STANDARD} STREQUAL "c++11")
    LIST(APPEND KOKKOS_CXX_FEATURES cxx_std_11)
  ELSEIF(${KOKKOS_CXX_STANDARD} STREQUAL "c++14")
    LIST(APPEND KOKKOS_CXX_FEATURES cxx_std_14)
  ELSEIF(${KOKKOS_CXX_STANDARD} STREQUAL "c++17")
    LIST(APPEND KOKKOS_CXX_FEATURES cxx_std_17)
  ENDIF()
ENDIF()

IF (CMAKE_CXX_STANDARD)
  IF (${CMAKE_CXX_STANDARD} STREQUAL "11")
    LIST(APPEND KOKKOS_CXX_FEATURES cxx_std_11)
  ELSEIF(${CMAKE_CXX_STANDARD} STREQUAL "14")
    LIST(APPEND KOKKOS_CXX_FEATURES cxx_std_14)
  ELSEIF(${CMAKE_CXX_STANDARD} STREQUAL "17")
    LIST(APPEND KOKKOS_CXX_FEATURES cxx_std_17)
  ENDIF()
ENDIF()




INCLUDE(${KOKKOS_SRC_PATH}/cmake/kokkos_tribits.cmake)
KOKKOS_SETUP_BUILD_ENVIRONMENT()

SET_PROPERTY(GLOBAL PROPERTY KOKKOS_LIBRARIES_NAMES)
SET_PROPERTY(GLOBAL PROPERTY KOKKOS_ALL_SOURCES)
SET_PROPERTY(GLOBAL PROPERTY KOKKOS_BUILD_INTERFACE_INCLUDES)
SET_PROPERTY(GLOBAL PROPERTY KOKKOS_COMPILE_DEFINITIONS)
IF (NOT KOKKOS_HAS_TRILINOS AND NOT KOKKOS_SEPARATE_LIBS)
  SET_PROPERTY(GLOBAL PROPERTY KOKKOS_LIBRARIES_NAMES kokkos)
ENDIF()


GET_DIRECTORY_PROPERTY(HAS_PARENT PARENT_DIRECTORY)
IF (KOKKOS_HAS_TRILINOS)
  SET(TRILINOS_INCDIR ${CMAKE_INSTALL_PREFIX}/${${PROJECT_NAME}_INSTALL_INCLUDE_DIR})
  SET(KOKKOS_HEADER_DIR ${TRILINOS_INCDIR})
ELSEIF(HAS_PARENT)
  SET(KOKKOS_HEADER_DIR "include/kokkos")
ELSE()
  SET(KOKKOS_HEADER_DIR "include")
ENDIF()


#------------ NOW BUILD ------------------------------------------------------
#include(${KOKKOS_SRC_PATH}/cmake/kokkos_build.cmake)

#TRIBITS_PACKAGE_DECL(Kokkos)
#ADD_SUBDIRECTORY(core)
#ADD_SUBDIRECTORY(containers)
#ADD_SUBDIRECTORY(algorithms)

#------------------------------------------------------------------------------
#
# A) Forward declare the package so that certain options are also defined for
# subpackages
#

KOKKOS_PACKAGE_DECL()


#------------------------------------------------------------------------------
#
# B) Install Kokkos' build files
#
# If using the Makefile-generated files, then need to set things up.
# Here, assume that TriBITS has been run from ProjectCompilerPostConfig.cmake
# and already generated KokkosCore_config.h and kokkos_generated_settings.cmake
# in the previously define Kokkos_GEN_DIR
# We need to copy them over to the correct place and source the cmake file

set(Kokkos_GEN_DIR ${CMAKE_BINARY_DIR})
file(COPY "${Kokkos_GEN_DIR}/KokkosCore_config.h"
  DESTINATION "${CMAKE_CURRENT_BINARY_DIR}" USE_SOURCE_PERMISSIONS)
install(FILES "${Kokkos_GEN_DIR}/KokkosCore_config.h"
  DESTINATION include)
file(COPY "${Kokkos_GEN_DIR}/kokkos_generated_settings.cmake"
  DESTINATION "${CMAKE_CURRENT_BINARY_DIR}" USE_SOURCE_PERMISSIONS)

include(${CMAKE_CURRENT_BINARY_DIR}/kokkos_generated_settings.cmake)
# Sources come from makefile-generated kokkos_generated_settings.cmake file
# Enable using the individual sources if needed
set_kokkos_srcs(KOKKOS_SRC ${KOKKOS_SRC})

#------------------------------------------------------------------------------
#
# C) Install Kokkos' executable scripts
#

# nvcc_wrapper is Kokkos' wrapper for NVIDIA's NVCC CUDA compiler.
# Kokkos needs nvcc_wrapper in order to build.  Other libraries and
# executables also need nvcc_wrapper.  Thus, we need to install it.
# If the argument of DESTINATION is a relative path, CMake computes it
# as relative to ${CMAKE_INSTALL_PATH}.

INSTALL(PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/bin/nvcc_wrapper DESTINATION bin)


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



