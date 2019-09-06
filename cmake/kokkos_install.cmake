
SET(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")
SET(INSTALL_BIN_DIR bin CACHE PATH "Installation directory for executables")
SET(INSTALL_INCLUDE_DIR ${KOKKOS_HEADER_DIR} CACHE PATH
  "Installation directory for header files")

#Set all the variables needed for kokkosConfig.cmake
GET_PROPERTY(KOKKOS_PROP_LIBS GLOBAL PROPERTY KOKKOS_LIBRARIES_NAMES)
SET(KOKKOS_LIBRARIES ${KOKKOS_PROP_LIBS})

# Make relative paths absolute (needed later on)
FOREACH(p LIB BIN INCLUDE CMAKE)
  SET(var INSTALL_${p}_DIR)
  IF(NOT IS_ABSOLUTE "${${var}}")
    SET(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
  ENDIF()
ENDFOREACH()

INCLUDE(CMakePackageConfigHelpers)
CONFIGURE_PACKAGE_CONFIG_FILE(cmake/KokkosConfig.cmake.in
  "${Kokkos_BINARY_DIR}/KokkosConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake)
WRITE_BASIC_PACKAGE_VERSION_FILE("${Kokkos_BINARY_DIR}/KokkosConfigVersion.cmake"
      VERSION "${Kokkos_VERSION}"
      COMPATIBILITY SameMajorVersion)

# Install the KokkosConfig.cmake and KokkosConfigVersion.cmake
set(right_place lib/cmake/Kokkos)
set(wrong_place lib/CMake/Kokkos)
#                   ^^ case-sensitive
install(FILES
  "${Kokkos_BINARY_DIR}/KokkosConfig.cmake"
  "${Kokkos_BINARY_DIR}/KokkosConfigVersion.cmake"
  DESTINATION ${right_place})
install(EXPORT KokkosTargets NAMESPACE Kokkos:: DESTINATION ${right_place})

if (NOT APPLE) #case insensitive, can only install one config
# For backward compatibility, export legacy target (not namespaced) to the old
# location that will not be discovered by CMake when Kokkos install prefix is
# added to CMAKE_PREFIX_PATH in user code.
install(FILES
  "${Kokkos_BINARY_DIR}/KokkosConfig.cmake"
  "${Kokkos_BINARY_DIR}/KokkosConfigVersion.cmake"
  DESTINATION ${wrong_place})
install(EXPORT KokkosTargets DESTINATION ${wrong_place})
endif()

# build and install pkgconfig file
CONFIGURE_FILE(core/src/kokkos.pc.in kokkos.pc @ONLY)
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/kokkos.pc DESTINATION lib/pkgconfig)

CONFIGURE_FILE(cmake/KokkosCore_config.h.in KokkosCore_config.h @ONLY)
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/KokkosCore_config.h DESTINATION ${KOKKOS_HEADER_DIR})

