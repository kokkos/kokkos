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

SET(KOKKOS_COMPILE_OPTIONS)
SET(KOKKOS_LINK_OPTIONS)
SET(KOKKOS_CUDA_OPTIONS)
SET(KOKKOS_CUDAFE_OPTIONS)
SET(KOKKOS_XCOMPILER_OPTIONS)


IF(NOT KOKKOS_HAS_TRILINOS)
  cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
  IF(NOT DEFINED ${PROJECT_NAME})
    PROJECT(Kokkos CXX)
  ENDIF()
  set (Kokkos_VERSION_MAJOR 2)
  set (Kokkos_VERSION_MINOR 7)
  set (Kokkos_VERSION_PATCH 4)
ENDIF()

INCLUDE(${KOKKOS_SRC_PATH}/cmake/fake_tribits.cmake)
INCLUDE(${KOKKOS_SRC_PATH}/cmake/kokkos_tribits.cmake)
KOKKOS_SETUP_BUILD_ENVIRONMENT()

IF (NOT KOKKOS_HAS_TRILINOS AND NOT KOKKOS_SEPARATE_LIBS)
  SET(KOKKOS_LIBRARIES_NAMES kokkos)
ELSE()
  SET(KOKKOS_LIBRARIES_NAMES kokkoscore kokkoscontainers kokkosalgorithms)
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
#file(COPY "${Kokkos_GEN_DIR}/kokkos_generated_settings.cmake"
#  DESTINATION "${CMAKE_CURRENT_BINARY_DIR}" USE_SOURCE_PERMISSIONS)
#include(${CMAKE_CURRENT_BINARY_DIR}/kokkos_generated_settings.cmake)

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
  #just always do it
  KOKKOS_MAKE_LIBKOKKOS()
  include(${KOKKOS_SRC_PATH}/cmake/kokkos_install.cmake)
ENDIF()



