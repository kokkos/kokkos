include(CMakePackageConfigHelpers)
if(NOT Kokkos_INSTALL_TESTING)
  include(GNUInstallDirs)

  #Set all the variables needed for KokkosConfig.cmake
  get_property(KOKKOS_PROP_LIBS GLOBAL PROPERTY KOKKOS_LIBRARIES_NAMES)
  set(KOKKOS_LIBRARIES ${KOKKOS_PROP_LIBS})

  include(CMakePackageConfigHelpers)
  configure_package_config_file(
    cmake/KokkosConfig.cmake.in "${Kokkos_BINARY_DIR}/cmake_packages/KokkosConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}/cmake
  )

  configure_package_config_file(
    cmake/KokkosConfigCommon.cmake.in "${Kokkos_BINARY_DIR}/cmake_packages/KokkosConfigCommon.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}/cmake
  )

  write_basic_package_version_file(
    "${Kokkos_BINARY_DIR}/cmake_packages/KokkosConfigVersion.cmake" VERSION "${Kokkos_VERSION}"
    COMPATIBILITY AnyNewerVersion
  )

  # Install the KokkosConfig*.cmake files
  install(
    FILES "${Kokkos_BINARY_DIR}/cmake_packages/KokkosConfig.cmake"
          "${Kokkos_BINARY_DIR}/cmake_packages/KokkosConfigCommon.cmake"
          "${Kokkos_BINARY_DIR}/cmake_packages/KokkosConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Kokkos
  )
  if(Kokkos_ENABLE_EXPERIMENTAL_CXX20_MODULES)
    install(
      EXPORT KokkosTargets
      NAMESPACE Kokkos::
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Kokkos
      CXX_MODULES_DIRECTORY .
    )
  else()
    install(EXPORT KokkosTargets NAMESPACE Kokkos:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Kokkos)
  endif()
  export(EXPORT KokkosTargets NAMESPACE Kokkos:: FILE ${Kokkos_BINARY_DIR}/cmake_packages/KokkosTargets.cmake)

else()
  configure_file(cmake/KokkosConfigCommon.cmake.in ${Kokkos_BINARY_DIR}/KokkosConfigCommon.cmake @ONLY)

  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/KokkosConfigVersion.cmake" VERSION "${Kokkos_VERSION}" COMPATIBILITY AnyNewerVersion
  )

  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/KokkosConfigVersion.cmake
          DESTINATION "${${PROJECT_NAME}_INSTALL_LIB_DIR}/cmake/Kokkos"
  )
endif()

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/KokkosCore_config.h DESTINATION ${KOKKOS_HEADER_DIR})
