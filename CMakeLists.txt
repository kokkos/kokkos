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


list(APPEND CMAKE_MODULE_PATH cmake/Modules)

IF(NOT KOKKOS_HAS_TRILINOS)
  cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
  IF(NOT DEFINED ${PROJECT_NAME})
    PROJECT(Kokkos CXX)
  ENDIF()
  set (Kokkos_VERSION_MAJOR 3)
  set (Kokkos_VERSION_MINOR 0)
  set (Kokkos_VERSION_PATCH 0)
ENDIF()

IF(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.12.0") 
  MESSAGE(STATUS "Setting policy CMP0074 to use <Package>_ROOT variables")
  CMAKE_POLICY(SET CMP0074 NEW)
ENDIF()

INCLUDE(${KOKKOS_SRC_PATH}/cmake/fake_tribits.cmake)
GLOBAL_SET(KOKKOS_COMPILE_OPTIONS)
GLOBAL_SET(KOKKOS_LINK_OPTIONS)
GLOBAL_SET(KOKKOS_CUDA_OPTIONS)
GLOBAL_SET(KOKKOS_CUDAFE_OPTIONS)
GLOBAL_SET(KOKKOS_XCOMPILER_OPTIONS)



INCLUDE(${KOKKOS_SRC_PATH}/cmake/kokkos_tribits.cmake)
KOKKOS_SETUP_BUILD_ENVIRONMENT()

GET_DIRECTORY_PROPERTY(HAS_PARENT PARENT_DIRECTORY)
IF (KOKKOS_HAS_TRILINOS)
  SET(TRILINOS_INCDIR ${CMAKE_INSTALL_PREFIX}/${${PROJECT_NAME}_INSTALL_INCLUDE_DIR})
  SET(KOKKOS_HEADER_DIR ${TRILINOS_INCDIR})
  SET(KOKKOS_IS_SUBDIRECTORY TRUE)
ELSEIF(HAS_PARENT)
  SET(KOKKOS_HEADER_DIR "include/kokkos")
  SET(KOKKOS_IS_SUBDIRECTORY TRUE)
ELSE()
  SET(KOKKOS_HEADER_DIR "include")
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
  #just always do it - no more separate libs stuff
  KOKKOS_MAKE_LIBKOKKOS()
  include(${KOKKOS_SRC_PATH}/cmake/kokkos_install.cmake)
ENDIF()

# nvcc_wrapper is Kokkos' wrapper for NVIDIA's NVCC CUDA compiler.
# Kokkos needs nvcc_wrapper in order to build.  Other libraries and
# executables also need nvcc_wrapper.  Thus, we need to install it.
# If the argument of DESTINATION is a relative path, CMake computes it
# as relative to ${CMAKE_INSTALL_PATH}.
INSTALL(PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/bin/nvcc_wrapper DESTINATION bin)
INSTALL(FILES "${CMAKE_BINARY_DIR}/KokkosCore_config.h" DESTINATION include)

