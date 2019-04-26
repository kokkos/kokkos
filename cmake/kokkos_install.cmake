
SET(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")
SET(INSTALL_BIN_DIR bin CACHE PATH "Installation directory for executables")
SET(INSTALL_INCLUDE_DIR ${KOKKOS_HEADER_DIR} CACHE PATH
  "Installation directory for header files")

#Set all the variables needed for kokkosConfig.cmake
GET_PROPERTY(KOKKOS_PROP_LIBS GLOBAL PROPERTY KOKKOS_LIBRARIES_NAMES)
SET(KOKKOS_LIBRARIES ${KOKKOS_PROP_LIBS})

SET(DEF_INSTALL_CMAKE_DIR)
IF(WIN32 AND NOT CYGWIN)
  LIST(APPEND DEF_INSTALL_CMAKE_DIR CMake)
ELSE()
  LIST(APPEND DEF_INSTALL_CMAKE_DIR lib/CMake/Kokkos)
  #also add the totally normal place for it to be
  LIST(APPEND DEF_INSTALL_CMAKE_DIR lib/cmake)
ENDIF()
SET(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR} CACHE PATH
    "Installation directory for CMake files")

# Make relative paths absolute (needed later on)
FOREACH(p LIB BIN INCLUDE CMAKE)
  SET(var INSTALL_${p}_DIR)
  IF(NOT IS_ABSOLUTE "${${var}}")
    SET(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
  ENDIF()
ENDFOREACH()

INSTALL (FILES
  ${Kokkos_BINARY_DIR}/KokkosCore_config.h
  DESTINATION ${KOKKOS_HEADER_DIR}
)

# Add all targets to the build-tree export set
#export(TARGETS ${Kokkos_LIBRARIES_NAMES}
#  FILE "${Kokkos_BINARY_DIR}/KokkosTargets.cmake")

# Export the package for use from the build-tree
# (this registers the build-tree with a global CMake-registry)
#export(PACKAGE Kokkos)

# Create the KokkosConfig.cmake and KokkosConfigVersion files
file(RELATIVE_PATH REL_INCLUDE_DIR "${INSTALL_CMAKE_DIR}"
   "${INSTALL_INCLUDE_DIR}")
# ... for the build tree
set(CONF_INCLUDE_DIRS "${Kokkos_SOURCE_DIR}" "${Kokkos_BINARY_DIR}")
include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/KokkosConfig.cmake.in
  "${Kokkos_BINARY_DIR}/KokkosConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake)
write_basic_package_version_file("${Kokkos_BINARY_DIR}/KokkosConfigVersion.cmake"
      VERSION "${Kokkos_VERSION_MAJOR}.${Kokkos_VERSION_MINOR}.${Kokkos_VERSION_PATCH}"
      COMPATIBILITY SameMajorVersion)

# Install the KokkosConfig.cmake and KokkosConfigVersion.cmake
FOREACH(DIR ${INSTALL_CMAKE_DIR})
  install(FILES
    "${Kokkos_BINARY_DIR}/KokkosConfig.cmake"
    DESTINATION ${DIR})
  install(FILES
    "${Kokkos_BINARY_DIR}/KokkosConfigVersion.cmake"
    DESTINATION ${DIR})

  # Install the export set for use with the install-tree
  INSTALL(EXPORT KokkosTargets DESTINATION ${DIR})
ENDFOREACH()

# build and install pkgconfig file
CONFIGURE_FILE(core/src/kokkos.pc.in kokkos.pc @ONLY)
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/kokkos.pc DESTINATION lib/pkgconfig)

CONFIGURE_FILE(cmake/KokkosCore_config.h.in KokkosCore_config.h @ONLY)
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/KokkosCore_config.h DESTINATION include)

