IF(COMMAND TRIBITS_PACKAGE_DECL)
  SET(KOKKOS_HAS_TRILINOS ON CACHE BOOL "")
ELSE()
  SET(KOKKOS_HAS_TRILINOS OFF CACHE BOOL "")
ENDIF()

IF(NOT KOKKOS_HAS_TRILINOS)
  cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
  project(Kokkos CXX)

  # Basic initialization (Used in KOKKOS_SETTINGS)
  set(KOKKOS_SRC_PATH ${Kokkos_SOURCE_DIR})
  set(KOKKOS_PATH ${KOKKOS_SRC_PATH})

  #------------ COMPILER AND FEATURE CHECKS ------------------------------------
  include(${KOKKOS_SRC_PATH}/cmake/kokkos_functions.cmake)
  set_kokkos_cxx_compiler()
  set_kokkos_compiler_standard()
  
  #------------ GET OPTIONS AND KOKKOS_SETTINGS --------------------------------
  # Add Kokkos' modules to CMake's module path.
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${Kokkos_SOURCE_DIR}/cmake/Modules/")

  set(KOKKOS_CMAKE_VERBOSE True)
  include(${KOKKOS_SRC_PATH}/cmake/kokkos_options.cmake)

  include(${KOKKOS_SRC_PATH}/cmake/kokkos_settings.cmake)

  #------------ GENERATE HEADER AND SOURCE FILES -------------------------------
  execute_process(
    COMMAND ${KOKKOS_SETTINGS} make -f ${KOKKOS_SRC_PATH}/core/src/Makefile build-makefile-cmake-kokkos
    WORKING_DIRECTORY "${Kokkos_BINARY_DIR}"
    OUTPUT_FILE ${Kokkos_BINARY_DIR}/core_src_make.out
    RESULT_VARIABLE res
  )
  include(${Kokkos_BINARY_DIR}/gen_kokkos.cmake)

  #------------ NOW BUILD ------------------------------------------------------
  include(${KOKKOS_SRC_PATH}/cmake/kokkos_build.cmake)

ELSE()
#------------------------------------------------------------------------------
#
# A) Forward delcare the package so that certain options are also defined for
# subpackages
#

TRIBITS_PACKAGE_DECL(Kokkos) # ENABLE_SHADOWING_WARNINGS)


#------------------------------------------------------------------------------
#
# B) Install Kokkos' build files
#
# If using the Makefile-generated files, then need to set things up.
# Here, assume that TriBITS has been run from ProjectCompilerPostConfig.cmake
# and already generated KokkosCore_config.h and gen_cmake.kokkos
# in the previously define Kokkos_GEN_DIR
# We need to copy them over to the correct place and source the cmake file

if(NOT KOKKOS_LEGACY_TRIBITS)
  set(Kokkos_GEN_DIR ${CMAKE_BINARY_DIR})
  file(COPY "${Kokkos_GEN_DIR}/KokkosCore_config.h"
    DESTINATION "${CMAKE_CURRENT_BINARY_DIR}" USE_SOURCE_PERMISSIONS)
  file(COPY "${Kokkos_GEN_DIR}/gen_kokkos.cmake"
    DESTINATION "${CMAKE_CURRENT_BINARY_DIR}" USE_SOURCE_PERMISSIONS)

  include(${CMAKE_CURRENT_BINARY_DIR}/gen_kokkos.cmake)
  # Sources come from makefile-generated gen_kokkos.cmake file
  # Enable using the individual sources if needed
  set_kokkos_srcs(KOKKOS_SRC ${KOKKOS_SRC})
endif ()


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
# D) Process the subpackages for Kokkos
#

TRIBITS_PROCESS_SUBPACKAGES()


#------------------------------------------------------------------------------
#
# E) If Kokkos itself is enabled, process the Kokkos package
#

TRIBITS_PACKAGE_DEF()

TRIBITS_EXCLUDE_AUTOTOOLS_FILES()

TRIBITS_EXCLUDE_FILES(
  classic/doc
  classic/LinAlg/doc/CrsRefactorNotesMay2012
  )

TRIBITS_PACKAGE_POSTPROCESS()

ENDIF()
