
INCLUDE(GNUInstallDirs)

#Set all the variables needed for kokkosConfig.cmake
GET_PROPERTY(KOKKOS_PROP_LIBS GLOBAL PROPERTY KOKKOS_LIBRARIES_NAMES)
SET(KOKKOS_LIBRARIES ${KOKKOS_PROP_LIBS})

INCLUDE(CMakePackageConfigHelpers)
CONFIGURE_PACKAGE_CONFIG_FILE(cmake/KokkosConfig.cmake.in
  "${Kokkos_BINARY_DIR}/KokkosConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}/cmake)
WRITE_BASIC_PACKAGE_VERSION_FILE("${Kokkos_BINARY_DIR}/KokkosConfigVersion.cmake"
      VERSION "${Kokkos_VERSION}"
      COMPATIBILITY SameMajorVersion)

# Install the KokkosConfig.cmake and KokkosConfigVersion.cmake
install(FILES
  "${Kokkos_BINARY_DIR}/KokkosConfig.cmake"
  "${Kokkos_BINARY_DIR}/KokkosConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Kokkos)
install(EXPORT KokkosTargets NAMESPACE Kokkos:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Kokkos)

# build and install pkgconfig file
CONFIGURE_FILE(core/src/kokkos.pc.in kokkos.pc @ONLY)
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/kokkos.pc DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)

CONFIGURE_FILE(cmake/KokkosCore_config.h.in KokkosCore_config.h @ONLY)
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/KokkosCore_config.h DESTINATION ${KOKKOS_HEADER_DIR})

