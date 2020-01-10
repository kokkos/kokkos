# In case we need it for irreparably bad build systems
# Export CXXFLAGS in a variable
GET_TARGET_PROPERTY(KOKKOS_COMPILE_OPTIONS_LIST kokkoscore INTERFACE_COMPILE_OPTIONS)
LIST(APPEND KOKKOS_COMPILE_OPTIONS_LIST ${CMAKE_CXX${KOKKOS_CXX_STANDARD}_STANDARD_COMPILE_OPTION})
IF (NOT KOKKOS_HAS_TRILINOS)
  INCLUDE(GNUInstallDirs)

  #Set all the variables needed for KokkosConfig.cmake
  GET_PROPERTY(KOKKOS_PROP_LIBS GLOBAL PROPERTY KOKKOS_LIBRARIES_NAMES)
  SET(KOKKOS_LIBRARIES ${KOKKOS_PROP_LIBS})

  INCLUDE(CMakePackageConfigHelpers)
  CONFIGURE_PACKAGE_CONFIG_FILE(
    cmake/KokkosConfig.cmake.in
    "${Kokkos_BINARY_DIR}/KokkosConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}/cmake)

  INCLUDE(CMakePackageConfigHelpers)
  CONFIGURE_PACKAGE_CONFIG_FILE(
	  cmake/KokkosConfigCommon.cmake.in
	  "${Kokkos_BINARY_DIR}/KokkosConfigCommon.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}/cmake)

  WRITE_BASIC_PACKAGE_VERSION_FILE("${Kokkos_BINARY_DIR}/KokkosConfigVersion.cmake"
      VERSION "${Kokkos_VERSION}"
      COMPATIBILITY SameMajorVersion)

  # Install the KokkosConfig*.cmake files
  install(FILES
    "${Kokkos_BINARY_DIR}/KokkosConfig.cmake"
    "${Kokkos_BINARY_DIR}/KokkosConfigCommon.cmake"
    "${Kokkos_BINARY_DIR}/KokkosConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Kokkos)
  install(EXPORT KokkosTargets NAMESPACE Kokkos:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Kokkos)
ELSE()
  CONFIGURE_FILE(cmake/KokkosConfigCommon.cmake.in ${Kokkos_BINARY_DIR}/KokkosConfigCommon.cmake @ONLY)
  file(READ ${Kokkos_BINARY_DIR}/KokkosConfigCommon.cmake KOKKOS_CONFIG_COMMON)
  file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/KokkosConfig_install.cmake" "${KOKKOS_CONFIG_COMMON}")
ENDIF()

# build and install pkgconfig file
CONFIGURE_FILE(core/src/kokkos.pc.in kokkos.pc @ONLY)
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/kokkos.pc DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)

INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/KokkosCore_config.h DESTINATION ${KOKKOS_HEADER_DIR})

