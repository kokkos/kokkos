include(CMakePackageConfigHelpers)
if(NOT Kokkos_INSTALL_TESTING)
  include(GNUInstallDirs)

  #Set all the variables needed for KokkosConfig.cmake
  get_property(KOKKOS_PROP_LIBS GLOBAL PROPERTY KOKKOS_LIBRARIES_NAMES)
  set(KOKKOS_LIBRARIES ${KOKKOS_PROP_LIBS})

  include(CMakePackageConfigHelpers)
  configure_package_config_file(
    cmake/KokkosConfig.cmake.in "${Kokkos_BINARY_DIR}/KokkosConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}/cmake
  )

  configure_package_config_file(
    cmake/KokkosConfigCommon.cmake.in "${Kokkos_BINARY_DIR}/KokkosConfigCommon.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}/cmake
  )

  write_basic_package_version_file(
    "${Kokkos_BINARY_DIR}/KokkosConfigVersion.cmake" VERSION "${Kokkos_VERSION}" COMPATIBILITY AnyNewerVersion
  )

  # Install the KokkosConfig*.cmake files
  install(FILES "${Kokkos_BINARY_DIR}/KokkosConfig.cmake" "${Kokkos_BINARY_DIR}/KokkosConfigCommon.cmake"
                "${Kokkos_BINARY_DIR}/KokkosConfigVersion.cmake" DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Kokkos
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
  export(EXPORT KokkosTargets NAMESPACE Kokkos:: FILE ${Kokkos_BINARY_DIR}/KokkosTargets.cmake)

  # Required to be a TriBITS-compliant external package
  file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/cmake_packages/Kokkos)
  file(COPY ${Kokkos_BINARY_DIR}/KokkosConfig.cmake ${Kokkos_BINARY_DIR}/KokkosConfigCommon.cmake
            ${Kokkos_BINARY_DIR}/KokkosConfigVersion.cmake DESTINATION ${CMAKE_BINARY_DIR}/cmake_packages/Kokkos
  )
  file(WRITE ${CMAKE_BINARY_DIR}/cmake_packages/Kokkos/KokkosTargets.cmake
       "include(${Kokkos_BINARY_DIR}/KokkosTargets.cmake)"
  )

  # Create .pc files to be used by pkg-config
  if(Kokkos_ENABLE_HWLOC)
    list(APPEND DEPS_LIBS ${HWLOC_LIBRARY})
  endif()
  if(Kokkos_ENABLE_LIBDL)
    list(APPEND DEPS_LIBS ${dl_LOCATION})
  endif()
  if(Kokkos_ENABLE_OPENMP OR Kokkos_ENABLE_THREADS)
    if(Kokkos_ENABLE_OPENMP)
      set(OPENMP_FLAGS ${OpenMP_CXX_FLAGS})
    endif()
    foreach(OPENMP_LIBNAME ${OpenMP_CXX_LIB_NAMES})
      list(APPEND DEPS_LIBS "${OpenMP_${OPENMP_LIBNAME}_LIBRARY}")
    endforeach()
  endif()
  if(Kokkos_ENABLE_CUDA)
    list(APPEND DEPS_LIBS ${CUDA_cuda_driver_LIBRARY})
  endif()
  list(APPEND DEPS_LIBS "-lstdc++ -lm")
  if(BUILD_SHARED_LIBS)
    list(JOIN DEPS_PRIVATE_LIBS " " DEPS_LIBS)
    unset(DEPS_LIBS)
  else()
    list(JOIN DEPS_LIBS " " DEPS_LIBS)
  endif()
  configure_file(kokkos-core.pc.in kokkos-core.pc @ONLY)
  if(BUILD_SHARED_LIBS)
    set(DEPS_PRIVATE_LIBS "-lstdc++ -lm")
    unset(DEPS_LIBS)
  else()
    set(DEPS_LIBS "-lstdc++ -lm")
    unset(DEPS_PRIVATE_LIBS)
  endif()
  configure_file(kokkos-containers.pc.in kokkos-containers.pc @ONLY)
  install(FILES ${CMAKE_BINARY_DIR}/kokkos-core.pc
                ${CMAKE_BINARY_DIR}/kokkos-containers.pc
          DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig"
  )
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
