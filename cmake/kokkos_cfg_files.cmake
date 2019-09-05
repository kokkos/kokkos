
# build and install pkgconfig file
CONFIGURE_FILE(core/src/kokkos.pc.in kokkos.pc @ONLY)
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/kokkos.pc DESTINATION lib/pkgconfig)

CONFIGURE_FILE(cmake/KokkosCore_config.h.in KokkosCore_config.h @ONLY)
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/KokkosCore_config.h DESTINATION ${KOKKOS_HEADER_DIR})

